"""Calculate drag coefficients for aircraft from YAML files.

This module calculates drag polar coefficients (CD0, k, e) and landing gear drag
increment for aircraft based on their physical characteristics.

The drag polar equation: CD = CD0 + k * CL^2

Where:
    - CD0: Zero-lift drag coefficient (parasite drag)
    - k: Induced drag factor, k = 1 / (pi * AR * e)
    - e: Oswald efficiency factor
    - CL: Lift coefficient

For landing/takeoff phases, landing gear drag is added:
    CD = CD0 + k * CL^2 + CD_gear


FORMULAS AND REFERENCES
=======================

1. Skin Friction Coefficient (Cf)
---------------------------------
Empirical curve fit for turbulent flat plate skin friction:

    Cf = 0.00258 + 0.00102 * exp(-6.28e-9 * Re) + 0.00295 * exp(-2.01e-8 * Re)

This is a curve fit approximation that transitions smoothly across Reynolds
number ranges, suitable for typical transport aircraft cruise conditions.

Reference:
    - Raymer, D. P. (2018). Aircraft Design: A Conceptual Approach (6th ed.).
      AIAA Education Series. Chapter 12.


2. Zero-Lift Drag Coefficient (CD0)
-----------------------------------
Component buildup method using flat plate analogy:

    CD0 = sum(Cf_i * FF_i * Q_i * S_wet_i) / S_ref

Simplified for wing and fuselage:

    CD0_wing = Cf_wing * S_wet_wing / S_ref
    CD0_fuselage = Cf_fuselage * S_wet_fuselage / S_ref
    CD0_total = (CD0_wing + CD0_fuselage) * component_factor

Where:
    - S_wet_wing = 2 * S_ref (upper and lower surfaces)
    - S_wet_fuselage = pi * L_fus * (W_fus + H_fus) / 2 (elliptical cross-section)
    - component_factor = 1.05 (accounts for tail, nacelles, interference, etc.)

Reference:
    - Raymer, D. P. (2018). Aircraft Design: A Conceptual Approach (6th ed.).
      AIAA Education Series. Chapter 12, Eq. 12.27.
    - Torenbeek, E. (1982). Synthesis of Subsonic Airplane Design.
      Delft University Press. Appendix F.


3. Oswald Efficiency Factor (e)
-------------------------------
Obert method for transport aircraft:

    e = 1 / (1.05 + 0.007 * pi * AR)

Where AR is the wing aspect ratio (b^2 / S).

This empirical method accounts for non-elliptical lift distribution and
viscous effects on induced drag.

Reference:
    - Obert, E. (2009). Aerodynamic Design of Transport Aircraft.
      IOS Press. Chapter 4.
    - Nita, M., & Scholz, D. (2012). Estimating the Oswald Factor from
      Basic Aircraft Geometrical Parameters. DLRK 2012.


4. Induced Drag Factor (k)
--------------------------
From lifting line theory:

    k = 1 / (pi * AR * e)

This relates the induced drag coefficient to the lift coefficient squared:
    CD_induced = k * CL^2

Reference:
    - Anderson, J. D. (2017). Fundamentals of Aerodynamics (6th ed.).
      McGraw-Hill. Chapter 5.


5. Landing Gear Drag Coefficient (CD_gear)
------------------------------------------
Torenbeek G-66 statistical method:

    Original formula (G-66):
        (CD_uc)_basic = 7e-4 * (W_to)^0.785 / S

    With correction factor F_uc for local flow conditions (G-67):
        CD_uc = F_uc * (CD_uc)_basic

    Simplified formula with F_uc ~ 0.45 pre-applied for typical landing:
        CD_gear = 3.2e-4 * (MTOW)^0.785 / S

Where:
    - MTOW: Maximum takeoff weight in kg
    - S: Wing reference area in m^2
    - F_uc: Correction factor accounting for angle of attack and flap
      deflection effects on local flow velocity below the wing.
      F_uc < 1 when flaps are deployed (reduced local velocity).

The original G-66 constant is 7e-4 (SI units). The factor F_uc ~ 0.45 was
derived by comparing calculated values with validated aircraft data.

Reference:
    - Torenbeek, E. (1982). Synthesis of Subsonic Airplane Design.
      Delft University Press. Appendix G, Section G-6, Equations G-63 to G-67.
      ISBN: 978-90-247-2724-7.


6. Reynolds Number
------------------
Standard definition:

    Re = rho * V * L / mu

Where:
    - rho: Air density (kg/m^3)
    - V: True airspeed (m/s)
    - L: Characteristic length (m) - MAC for wing, length for fuselage
    - mu: Dynamic viscosity (Pa.s)


7. International Standard Atmosphere (ISA)
------------------------------------------
Temperature and density variation with altitude:

    Troposphere (h < 11000 m):
        T = T0 + beta * h
        rho = rho0 * (T / T0)^4.2568

    Stratosphere (h >= 11000 m):
        T = 216.65 K (constant)
        rho = rho_trop * exp(-dh / 6341.55)

Where:
    - T0 = 288.15 K (sea level temperature)
    - rho0 = 1.225 kg/m^3 (sea level density)
    - beta = -0.0065 K/m (lapse rate)

Reference:
    - ICAO (1993). Manual of the ICAO Standard Atmosphere (3rd ed.).
      Doc 7488/3.


Usage:
    python dragcoef.py       # Display values without updating (dry-run)
    python dragcoef.py --run # Update all YAML files
"""

import sys
from pathlib import Path

import numpy as np
import yaml

# Add src to path to import aerox from a source checkout.
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))


# =============================================================================
# ATMOSPHERIC FUNCTIONS (ISA Model)
# =============================================================================


def atmos(h, dT=0):
    """Compute pressure, density and temperature at a given altitude.

    Based on ICAO Standard Atmosphere (ISA) model.

    Args:
        h (float): Altitude (in meters).
        dT (float): Temperature shift from ISA (in K). Defaults to 0.

    Returns:
        tuple: Air pressure (Pa), density (kg/m3), and temperature (K).

    Reference:
        ICAO Doc 7488/3, Manual of the ICAO Standard Atmosphere.
    """
    # ISA constants
    T0 = 288.15  # K, sea level temperature
    rho0 = 1.225  # kg/m3, sea level density
    R = 287.05287  # J/(kg.K), specific gas constant for air
    beta = -0.0065  # K/m, temperature lapse rate in troposphere

    dT = np.maximum(-15, np.minimum(dT, 15))
    T0_shift = T0 + dT

    # Temperature (with tropopause limit at 216.65 K)
    T = np.maximum(T0_shift + beta * h, 216.65 + dT)

    # Density in troposphere
    rhotrop = rho0 * (T / T0_shift) ** 4.256848030018761

    # Stratospheric correction (exponential decay above 11 km)
    dhstrat = np.maximum(0.0, h - 11000.0)
    rho = rhotrop * np.exp(-dhstrat / 6341.552161)

    # Pressure from ideal gas law
    p = rho * R * T

    return p, rho, T


def density(h, dT=0):
    """Compute air density at a given altitude.

    Args:
        h (float): Altitude (in meters).
        dT (float): Temperature shift from ISA (in K). Defaults to 0.

    Returns:
        float: Air density (kg/m^3).
    """
    _, rho, _ = atmos(h, dT)
    return rho


def vsound(h, dT=0):
    """Compute speed of sound at a given altitude.

    Formula: a = sqrt(gamma * R * T)

    Args:
        h (float): Altitude (in meters).
        dT (float): Temperature shift from ISA (in K). Defaults to 0.

    Returns:
        float: Speed of sound (m/s).
    """
    gamma = 1.40  # Ratio of specific heats for air (cp/cv)
    R = 287.05287  # J/(kg.K), specific gas constant for air
    _, _, T = atmos(h, dT)
    a = np.sqrt(gamma * R * T)
    return a


def mach2tas(mach, h):
    """Convert Mach number to true airspeed.

    Formula: TAS = M * a

    Args:
        mach (float): Mach number
        h (float): Altitude (in meters)

    Returns:
        float: True airspeed (m/s)
    """
    a = vsound(h)
    return mach * a


# =============================================================================
# AERODYNAMIC FUNCTIONS
# =============================================================================


def calc_re(
    density: float, tas: float, dynamic_viscosity: float, characteristic_length: float
) -> float:
    """Calculate Reynolds number.

    Formula: Re = rho * V * L / mu

    Args:
        density: Air density (kg/m^3)
        tas: True airspeed (m/s)
        dynamic_viscosity: Dynamic viscosity (Pa.s)
        characteristic_length: Characteristic length (m)

    Returns:
        Reynolds number (dimensionless)
    """
    return density * tas * characteristic_length / dynamic_viscosity


def calc_cf(Re: float) -> float:
    """Calculate skin friction coefficient for turbulent flow.

    Empirical curve fit approximation:
        Cf = 0.00258 + 0.00102 * exp(-6.28e-9 * Re) + 0.00295 * exp(-2.01e-8 * Re)

    This provides a smooth transition across Reynolds number ranges typical
    of transport aircraft (Re ~ 10^6 to 10^8).

    Args:
        Re: Reynolds number

    Returns:
        Skin friction coefficient (dimensionless)

    Reference:
        Raymer, D. P. (2018). Aircraft Design: A Conceptual Approach.
        Chapter 12.
    """
    cf = 0.00258 + 0.00102 * np.exp(-6.28e-9 * Re) + 0.00295 * np.exp(-2.01e-8 * Re)
    return cf


def calc_gear_drag(mtow_kg: float, wing_area_m2: float) -> float:
    """Calculate landing gear drag coefficient using Torenbeek G-66 method.

    Torenbeek G-66 statistical formula for retractable undercarriage drag:

        Original: (CD_uc)_basic = 7e-4 * (W_to)^0.785 / S

    With F_uc correction factor (~0.45) pre-applied for typical landing
    configuration with flaps deployed:

        CD_gear = 3.2e-4 * (MTOW)^0.785 / S

    The F_uc factor accounts for:
        - Local flow velocity reduction below wing due to flap deflection
        - Interference effects between gear and wing/flap

    Args:
        mtow_kg: Maximum takeoff weight in kg
        wing_area_m2: Wing reference area in m^2

    Returns:
        Landing gear drag coefficient increment (dimensionless)

    Reference:
        Torenbeek, E. (1982). Synthesis of Subsonic Airplane Design.
        Delft University Press. Appendix G, Section G-6, Eq. G-66.
        ISBN: 978-90-247-2724-7.
    """
    # Torenbeek constant: 7e-4 (SI) * F_uc (~0.45) = 3.2e-4
    constant = 3.2e-4
    return constant * (mtow_kg**0.785) / wing_area_m2


def calc_oswald_efficiency(aspect_ratio: float) -> float:
    """Calculate Oswald efficiency factor using Obert method.

    Obert empirical formula:
        e = 1 / (1.05 + 0.007 * pi * AR)

    This accounts for:
        - Non-elliptical spanwise lift distribution
        - Viscous drag increase with lift
        - Wing-fuselage interference

    Args:
        aspect_ratio: Wing aspect ratio (b^2 / S)

    Returns:
        Oswald efficiency factor (dimensionless, typically 0.7-0.85)

    Reference:
        Obert, E. (2009). Aerodynamic Design of Transport Aircraft.
        IOS Press.
        Nita, M., & Scholz, D. (2012). Estimating the Oswald Factor from
        Basic Aircraft Geometrical Parameters. DLRK 2012.
    """
    return 1.0 / (1.05 + 0.007 * np.pi * aspect_ratio)


def calc_induced_drag_factor(aspect_ratio: float, oswald_e: float) -> float:
    """Calculate induced drag factor k.

    From lifting line theory:
        k = 1 / (pi * AR * e)

    Such that: CD_induced = k * CL^2

    Args:
        aspect_ratio: Wing aspect ratio (b^2 / S)
        oswald_e: Oswald efficiency factor

    Returns:
        Induced drag factor k (dimensionless)

    Reference:
        Anderson, J. D. (2017). Fundamentals of Aerodynamics (6th ed.).
        McGraw-Hill. Chapter 5.
    """
    return 1.0 / (np.pi * aspect_ratio * oswald_e)


# =============================================================================
# MAIN CALCULATION FUNCTION
# =============================================================================


def calc_drag_coefficients(
    aircraft,
    altitude: float = 35000,
    mach: float = 0.78,
    dynamic_viscosity: float = 1.444e-5,
    component_factor: float = 1.05,
):
    """Calculate complete drag polar coefficients for an aircraft.

    Uses component buildup method for CD0 and Obert method for Oswald factor.

    CD0 calculation (Raymer component buildup):
        CD0 = sum(Cf * S_wet / S_ref) * component_factor

    Oswald efficiency (Obert method):
        e = 1 / (1.05 + 0.007 * pi * AR)

    Induced drag factor:
        k = 1 / (pi * AR * e)

    Landing gear drag (Torenbeek G-66):
        CD_gear = 3.2e-4 * (MTOW)^0.785 / S

    Args:
        aircraft: Aircraft object with attributes:
            - wing.area: Wing reference area (m^2)
            - wing.span: Wing span (m)
            - wing.mac: Mean aerodynamic chord (m)
            - fuselage.length: Fuselage length (m)
            - fuselage.height: Fuselage height (m)
            - fuselage.width: Fuselage width (m)
            - mtow: Maximum takeoff weight (kg)
        altitude: Cruise altitude in feet (default: 35000 ft)
        mach: Cruise Mach number (default: 0.78)
        dynamic_viscosity: Air dynamic viscosity (Pa.s, default: 1.444e-5)
        component_factor: Multiplier for misc. components (default: 1.05)

    Returns:
        Tuple of (cd0, k, e, gears):
            - cd0: Zero-lift drag coefficient
            - k: Induced drag factor
            - e: Oswald efficiency factor
            - gears: Landing gear drag coefficient increment

    References:
        - Raymer (2018): CD0 component buildup method
        - Obert (2009): Oswald efficiency factor
        - Torenbeek (1982): Landing gear drag, Appendix G, Eq. G-66
    """
    # Check if required data is available
    if not aircraft.wing or not aircraft.fuselage:
        raise ValueError(f"Aircraft {aircraft.icao} missing wing or fuselage data")

    # Extract geometric parameters
    S_ref = aircraft.wing.area  # Reference wing area (m^2)
    wing_span = aircraft.wing.span  # Wing span (m)
    L_wing = aircraft.wing.mac  # Mean aerodynamic chord (m)
    L_fuselage = aircraft.fuselage.length  # Fuselage length (m)
    H_fuselage = aircraft.fuselage.height  # Fuselage height (m)
    W_fuselage = aircraft.fuselage.width  # Fuselage width (m)

    # Convert altitude to meters
    altitude_m = altitude * 0.3048

    # Calculate atmospheric properties at cruise altitude
    rho = density(altitude_m)
    tas = mach2tas(mach, altitude_m)

    # Calculate Reynolds numbers for wing and fuselage
    Re_wing = calc_re(rho, tas, dynamic_viscosity, L_wing)
    Re_fuselage = calc_re(rho, tas, dynamic_viscosity, L_fuselage)

    # Calculate skin friction coefficients
    cf_wing = calc_cf(Re_wing)
    cf_fuselage = calc_cf(Re_fuselage)

    # Calculate wetted areas
    # Wing: both upper and lower surfaces
    S_wet_wing = 2 * S_ref

    # Fuselage: approximate as elliptical cross-section cylinder
    S_wet_fuselage = np.pi * L_fuselage * (W_fuselage + H_fuselage) / 2

    # Calculate CD0 components (flat plate analogy)
    cd0_wing = cf_wing * S_wet_wing / S_ref
    cd0_fuselage = cf_fuselage * S_wet_fuselage / S_ref

    # Total CD0 with component factor for tail, nacelles, interference, etc.
    cd0_total = (cd0_fuselage + cd0_wing) * component_factor

    # Calculate aspect ratio
    aspect_ratio = wing_span**2 / S_ref

    # Calculate Oswald efficiency factor (Obert method)
    e = calc_oswald_efficiency(aspect_ratio)

    # Calculate induced drag factor
    k = calc_induced_drag_factor(aspect_ratio, e)

    # Calculate landing gear drag coefficient (Torenbeek G-66)
    gears = calc_gear_drag(aircraft.mtow, S_ref)

    return cd0_total, k, e, gears


# =============================================================================
# YAML UPDATE FUNCTION
# =============================================================================


def update_aircraft_yaml(icao: str, cd0: float, k: float, e: float, gears: float):
    """Update the aircraft YAML file with calculated drag coefficients.

    Args:
        icao: ICAO aircraft code
        cd0: Zero-lift drag coefficient
        k: Induced drag factor
        e: Oswald efficiency factor
        gears: Landing gear drag coefficient increment
    """
    # Path to YAML file
    yaml_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aerox"
        / "data"
        / "aircraft"
        / "yml"
    )
    yaml_file = None

    # Search in subdirectories
    for subdir in yaml_dir.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{icao.lower()}.yml"
            if candidate.exists():
                yaml_file = candidate
                break

    if not yaml_file:
        print(f"Warning: YAML file not found for {icao}")
        return

    # Read existing YAML
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Update drag section
    if "drag" not in data:
        data["drag"] = {}

    data["drag"]["cd0"] = round(cd0, 5)
    data["drag"]["k"] = round(k, 5)
    data["drag"]["e"] = round(e, 5)
    data["drag"]["gears"] = round(gears, 5)

    # Write back to YAML
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"Updated {icao}: cd0={cd0:.5f}, k={k:.5f}, e={e:.5f}, gears={gears:.5f}")
