"""Aircraft-independent guidance programs and selected-guidance compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from aerox.fms.selected_intent import (
    SelectedAltitudeIntent,
    SelectedConfigurationIntent,
    SelectedSpeedGuidanceIntent,
    SelectedSpeedIntent,
    SelectedThrustIntent,
    SelectedVerticalIntent,
)


class LateralMode(StrEnum):
    """Aircraft-independent lateral guidance families."""

    ROUTE = "ROUTE"
    TRACK_HOLD = "TRACK_HOLD"
    HEADING_HOLD = "HEADING_HOLD"


class VerticalMode(StrEnum):
    """Aircraft-independent vertical guidance families."""

    PATH = "PATH"
    ALTITUDE = "ALTITUDE"
    ALTITUDE_HOLD = "ALTITUDE_HOLD"
    VERTICAL_SPEED = "VERTICAL_SPEED"
    FLIGHT_PATH_ANGLE = "FLIGHT_PATH_ANGLE"


class SpeedMode(StrEnum):
    """Speed guidance, including directional CAS/Mach crossover modes."""

    PROFILE = "PROFILE"
    CAS = "CAS"
    MACH = "MACH"
    CAS_MACH = "CAS_MACH"
    MACH_CAS = "MACH_CAS"
    SPEED_HOLD = "SPEED_HOLD"


class SpeedSelectorMode(StrEnum):
    """Active member of a concurrently selected CAS/Mach pair."""

    CAS = "CAS"
    MACH = "MACH"


class ThrustMode(StrEnum):
    """High-level thrust guidance families, not manufacturer FMA labels."""

    AUTO = "AUTO"
    MAXIMUM_AVAILABLE = "MAXIMUM_AVAILABLE"
    MINIMUM_AVAILABLE = "MINIMUM_AVAILABLE"
    THRUST_HOLD = "THRUST_HOLD"
    THRUST_FRACTION = "THRUST_FRACTION"
    THRUST_CEILING_FRACTION = "THRUST_CEILING_FRACTION"


class ConfigurationMode(StrEnum):
    """Effective aerodynamic-configuration guidance families."""

    CLEAN = "CLEAN"
    DRAG_INCREMENT = "DRAG_INCREMENT"


class GuardMetric(StrEnum):
    """Observable quantities usable by sequential transition guards."""

    ELAPSED_TIME_S = "ELAPSED_TIME_S"
    PRESSURE_ALTITUDE_M = "PRESSURE_ALTITUDE_M"
    ALONG_PATH_M = "ALONG_PATH_M"
    DISTANCE_TO_GO_M = "DISTANCE_TO_GO_M"
    WAYPOINT_INDEX = "WAYPOINT_INDEX"
    ALTITUDE_ERROR_M = "ALTITUDE_ERROR_M"
    SPEED_ERROR_MPS = "SPEED_ERROR_MPS"


class GuardComparison(StrEnum):
    """Comparison applied to an observed guard metric."""

    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    ABS_LE = "ABS_LE"


class GuardLogic(StrEnum):
    """How multiple conditions in one transition are combined."""

    ALL = "ALL"
    ANY = "ANY"


def _finite(value: float, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _optional_finite(value: float | None, name: str) -> float | None:
    return None if value is None else _finite(value, name)


@dataclass(frozen=True)
class GuidanceAlternative:
    """One discrete alternative retained by an inverse-guidance extractor."""

    label: str
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("guidance-alternative label must be non-empty")
        probability = _finite(self.probability, "alternative probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("alternative probability must lie between zero and one")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True)
class GuidanceEvidence:
    """Confidence and provenance attached to an inferred rule or transition."""

    confidence: float = 1.0
    source_ids: tuple[str, ...] = ()
    alternatives: tuple[GuidanceAlternative, ...] = ()

    def __post_init__(self) -> None:
        confidence = _finite(self.confidence, "guidance confidence")
        sources = tuple(self.source_ids)
        alternatives = tuple(self.alternatives)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("guidance confidence must lie between zero and one")
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError("guidance evidence source IDs must be non-empty strings")
        if len(sources) != len(set(sources)):
            raise ValueError("guidance evidence source IDs must be unique")
        if any(not isinstance(item, GuidanceAlternative) for item in alternatives):
            raise TypeError("guidance alternatives must be GuidanceAlternative objects")
        labels = [item.label for item in alternatives]
        if len(labels) != len(set(labels)):
            raise ValueError("guidance-alternative labels must be unique")
        if sum(item.probability for item in alternatives) > 1.0 + 1e-12:
            raise ValueError("guidance-alternative probabilities must not exceed one")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "alternatives", alternatives)


@dataclass(frozen=True)
class LateralCommand:
    """One lateral mode and its mode-specific target."""

    mode: LateralMode
    waypoint_index: int | None = None
    track_rad: float | None = None
    heading_rad: float | None = None
    evidence: GuidanceEvidence = field(default_factory=GuidanceEvidence)

    def __post_init__(self) -> None:
        mode = LateralMode(self.mode)
        track = _optional_finite(self.track_rad, "track target")
        heading = _optional_finite(self.heading_rad, "heading target")
        if self.waypoint_index is not None and (
            not isinstance(self.waypoint_index, int) or self.waypoint_index < 0
        ):
            raise ValueError("waypoint index must be a non-negative integer")
        if mode == LateralMode.ROUTE:
            if track is not None or heading is not None:
                raise ValueError("ROUTE mode does not accept track/heading targets")
        elif mode == LateralMode.TRACK_HOLD:
            if track is None or heading is not None or self.waypoint_index is not None:
                raise ValueError("TRACK_HOLD requires only a track target")
        elif heading is None or track is not None or self.waypoint_index is not None:
            raise ValueError("HEADING_HOLD requires only a heading target")
        if not isinstance(self.evidence, GuidanceEvidence):
            raise TypeError("lateral-command evidence must be GuidanceEvidence")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "track_rad", track)
        object.__setattr__(self, "heading_rad", heading)


@dataclass(frozen=True)
class VerticalCommand:
    """One vertical mode and its mode-specific target."""

    mode: VerticalMode
    altitude_m: float | None = None
    vertical_speed_mps: float | None = None
    flight_path_angle_rad: float | None = None
    evidence: GuidanceEvidence = field(default_factory=GuidanceEvidence)

    def __post_init__(self) -> None:
        mode = VerticalMode(self.mode)
        altitude = _optional_finite(self.altitude_m, "altitude target")
        vertical_speed = _optional_finite(
            self.vertical_speed_mps, "vertical-speed target"
        )
        angle = _optional_finite(self.flight_path_angle_rad, "flight-path-angle target")
        populated = sum(
            value is not None for value in (altitude, vertical_speed, angle)
        )
        if mode == VerticalMode.PATH and populated:
            raise ValueError("PATH mode does not accept a vertical target")
        if mode in {VerticalMode.ALTITUDE, VerticalMode.ALTITUDE_HOLD} and (
            altitude is None or populated != 1
        ):
            raise ValueError(f"{mode} requires only an altitude target")
        if mode == VerticalMode.VERTICAL_SPEED and (
            vertical_speed is None or populated != 1
        ):
            raise ValueError("VERTICAL_SPEED requires only a vertical-speed target")
        if mode == VerticalMode.FLIGHT_PATH_ANGLE and (angle is None or populated != 1):
            raise ValueError(
                "FLIGHT_PATH_ANGLE requires only a flight-path-angle target"
            )
        if angle is not None and abs(angle) >= np.deg2rad(30.0):
            raise ValueError("flight-path-angle target must be below 30 degrees")
        if not isinstance(self.evidence, GuidanceEvidence):
            raise TypeError("vertical-command evidence must be GuidanceEvidence")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "altitude_m", altitude)
        object.__setattr__(self, "vertical_speed_mps", vertical_speed)
        object.__setattr__(self, "flight_path_angle_rad", angle)


@dataclass(frozen=True)
class SpeedCommand:
    """One speed mode, selected CAS/Mach pair, and optional active member."""

    mode: SpeedMode
    calibrated_airspeed_mps: float | None = None
    mach: float | None = None
    active_selector: SpeedSelectorMode | None = None
    evidence: GuidanceEvidence = field(default_factory=GuidanceEvidence)

    def __post_init__(self) -> None:
        mode = SpeedMode(self.mode)
        cas = _optional_finite(
            self.calibrated_airspeed_mps, "calibrated-airspeed target"
        )
        mach = _optional_finite(self.mach, "Mach target")
        selector = (
            None
            if self.active_selector is None
            else SpeedSelectorMode(self.active_selector)
        )
        if mode == SpeedMode.PROFILE:
            if cas is not None or mach is not None or selector is not None:
                raise ValueError("PROFILE speed mode does not accept selector targets")
        elif cas is None or mach is None:
            raise ValueError(f"{mode} requires a selected CAS/Mach pair")
        if mode in {SpeedMode.CAS, SpeedMode.MACH} and selector not in {None, mode}:
            raise ValueError(f"{mode} speed mode cannot select {selector}")
        if cas is not None and cas <= 1.0:
            raise ValueError("selected CAS must exceed 1 m/s")
        if mach is not None and not 0.05 < mach < 1.5:
            raise ValueError("selected Mach must lie between 0.05 and 1.5")
        if not isinstance(self.evidence, GuidanceEvidence):
            raise TypeError("speed-command evidence must be GuidanceEvidence")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "calibrated_airspeed_mps", cas)
        object.__setattr__(self, "mach", mach)
        object.__setattr__(self, "active_selector", selector)


@dataclass(frozen=True)
class ThrustCommand:
    """One high-level thrust mode and its optional target."""

    mode: ThrustMode = ThrustMode.AUTO
    net_thrust_n: float | None = None
    available_thrust_fraction: float | None = None
    evidence: GuidanceEvidence = field(default_factory=GuidanceEvidence)

    def __post_init__(self) -> None:
        mode = ThrustMode(self.mode)
        thrust = _optional_finite(self.net_thrust_n, "net-thrust target")
        fraction = _optional_finite(
            self.available_thrust_fraction, "available-thrust fraction"
        )
        if mode == ThrustMode.THRUST_HOLD:
            if thrust is None or thrust < 0.0 or fraction is not None:
                raise ValueError("THRUST_HOLD requires only non-negative net thrust")
        elif mode in {
            ThrustMode.THRUST_FRACTION,
            ThrustMode.THRUST_CEILING_FRACTION,
        }:
            if thrust is not None or fraction is None or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{mode} requires only a fraction from zero to one")
        elif thrust is not None or fraction is not None:
            raise ValueError(f"{mode} does not accept a thrust target")
        if not isinstance(self.evidence, GuidanceEvidence):
            raise TypeError("thrust-command evidence must be GuidanceEvidence")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "net_thrust_n", thrust)
        object.__setattr__(self, "available_thrust_fraction", fraction)


@dataclass(frozen=True)
class ConfigurationCommand:
    """One effective configuration mode and its added drag coefficient."""

    mode: ConfigurationMode = ConfigurationMode.CLEAN
    drag_coefficient_increment: float = 0.0
    evidence: GuidanceEvidence = field(default_factory=GuidanceEvidence)

    def __post_init__(self) -> None:
        mode = ConfigurationMode(self.mode)
        increment = _finite(
            self.drag_coefficient_increment,
            "drag coefficient increment",
        )
        if increment < 0.0:
            raise ValueError("drag coefficient increment must be non-negative")
        if mode == ConfigurationMode.CLEAN and not np.isclose(increment, 0.0):
            raise ValueError("CLEAN configuration requires zero drag increment")
        if mode == ConfigurationMode.DRAG_INCREMENT and increment <= 0.0:
            raise ValueError("DRAG_INCREMENT requires a positive drag increment")
        if not isinstance(self.evidence, GuidanceEvidence):
            raise TypeError("configuration-command evidence must be GuidanceEvidence")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "drag_coefficient_increment", increment)


@dataclass(frozen=True)
class GuidanceObservation:
    """Aircraft-independent state supplied when evaluating transition guards."""

    elapsed_time_s: float | None = None
    pressure_altitude_m: float | None = None
    along_path_m: float | None = None
    distance_to_go_m: float | None = None
    waypoint_index: int | None = None
    altitude_error_m: float | None = None
    speed_error_mps: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "elapsed_time_s",
            "pressure_altitude_m",
            "along_path_m",
            "distance_to_go_m",
            "altitude_error_m",
            "speed_error_mps",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, _optional_finite(value, name))
        if self.waypoint_index is not None and (
            not isinstance(self.waypoint_index, int) or self.waypoint_index < 0
        ):
            raise ValueError("observed waypoint index must be a non-negative integer")
        if self.elapsed_time_s is not None and self.elapsed_time_s < 0.0:
            raise ValueError("observed elapsed time must be non-negative")
        if self.along_path_m is not None and self.along_path_m < 0.0:
            raise ValueError("observed along-path distance must be non-negative")
        if self.distance_to_go_m is not None and self.distance_to_go_m < 0.0:
            raise ValueError("observed distance to go must be non-negative")

    def value(self, metric: GuardMetric) -> float:
        """Return a requested metric or fail when it was not observed."""

        attribute = {
            GuardMetric.ELAPSED_TIME_S: "elapsed_time_s",
            GuardMetric.PRESSURE_ALTITUDE_M: "pressure_altitude_m",
            GuardMetric.ALONG_PATH_M: "along_path_m",
            GuardMetric.DISTANCE_TO_GO_M: "distance_to_go_m",
            GuardMetric.WAYPOINT_INDEX: "waypoint_index",
            GuardMetric.ALTITUDE_ERROR_M: "altitude_error_m",
            GuardMetric.SPEED_ERROR_MPS: "speed_error_mps",
        }[GuardMetric(metric)]
        value = getattr(self, attribute)
        if value is None:
            raise ValueError(f"transition guard requires observation {attribute}")
        return float(value)


@dataclass(frozen=True)
class GuardCondition:
    """One scalar condition in a transition guard."""

    metric: GuardMetric
    comparison: GuardComparison
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", GuardMetric(self.metric))
        object.__setattr__(self, "comparison", GuardComparison(self.comparison))
        object.__setattr__(
            self, "threshold", _finite(self.threshold, "guard threshold")
        )

    def matches(self, observation: GuidanceObservation) -> bool:
        """Evaluate this condition against one observation."""

        value = observation.value(self.metric)
        if self.comparison == GuardComparison.LT:
            return value < self.threshold
        if self.comparison == GuardComparison.LE:
            return value <= self.threshold
        if self.comparison == GuardComparison.GT:
            return value > self.threshold
        if self.comparison == GuardComparison.GE:
            return value >= self.threshold
        return abs(value) <= self.threshold


@dataclass(frozen=True)
class TransitionGuard:
    """Conditions that advance a sequential guidance program by one segment."""

    conditions: tuple[GuardCondition, ...]
    logic: GuardLogic = GuardLogic.ALL
    evidence: GuidanceEvidence = field(default_factory=GuidanceEvidence)

    def __post_init__(self) -> None:
        conditions = tuple(self.conditions)
        if not conditions or any(
            not isinstance(condition, GuardCondition) for condition in conditions
        ):
            raise ValueError("transition guard requires GuardCondition objects")
        if not isinstance(self.evidence, GuidanceEvidence):
            raise TypeError("transition evidence must be GuidanceEvidence")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "logic", GuardLogic(self.logic))

    def matches(self, observation: GuidanceObservation) -> bool:
        """Evaluate the complete guard against one observation."""

        results = tuple(condition.matches(observation) for condition in self.conditions)
        return all(results) if self.logic == GuardLogic.ALL else any(results)


@dataclass(frozen=True)
class GuidanceSegment:
    """Concurrent lateral, vertical, speed, thrust, and configuration rules."""

    segment_id: str
    lateral: LateralCommand
    vertical: VerticalCommand
    speed: SpeedCommand
    thrust: ThrustCommand = field(default_factory=ThrustCommand)
    configuration: ConfigurationCommand = field(default_factory=ConfigurationCommand)
    transition: TransitionGuard | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValueError("guidance segment ID must be non-empty")
        for name, value, expected_type in (
            ("lateral", self.lateral, LateralCommand),
            ("vertical", self.vertical, VerticalCommand),
            ("speed", self.speed, SpeedCommand),
            ("thrust", self.thrust, ThrustCommand),
            ("configuration", self.configuration, ConfigurationCommand),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"segment {name} must be {expected_type.__name__}")
        if self.transition is not None and not isinstance(
            self.transition, TransitionGuard
        ):
            raise TypeError("segment transition must be TransitionGuard or None")


@dataclass(frozen=True)
class SpeedGuidanceStage:
    """One independently guarded speed command."""

    stage_id: str
    command: SpeedCommand
    transition: TransitionGuard | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("speed-guidance stage ID must be non-empty")
        if not isinstance(self.command, SpeedCommand):
            raise TypeError("speed-guidance stage command must be SpeedCommand")
        if self.transition is not None and not isinstance(
            self.transition, TransitionGuard
        ):
            raise TypeError(
                "speed-guidance stage transition must be TransitionGuard or None"
            )


@dataclass(frozen=True)
class VerticalGuidanceStage:
    """One independently guarded vertical command."""

    stage_id: str
    command: VerticalCommand
    transition: TransitionGuard | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("vertical-guidance stage ID must be non-empty")
        if not isinstance(self.command, VerticalCommand):
            raise TypeError("vertical-guidance stage command must be VerticalCommand")
        if self.transition is not None and not isinstance(
            self.transition, TransitionGuard
        ):
            raise TypeError(
                "vertical-guidance stage transition must be TransitionGuard or None"
            )


@dataclass(frozen=True)
class GuidanceProgram:
    """A guidance program with joint rules and optional independent channels."""

    program_id: str
    segments: tuple[GuidanceSegment, ...]
    format_version: int = 1
    metadata: Mapping[str, str] = field(default_factory=dict)
    speed_stages: tuple[SpeedGuidanceStage, ...] = ()
    vertical_stages: tuple[VerticalGuidanceStage, ...] = ()

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not isinstance(self.program_id, str) or not self.program_id.strip():
            raise ValueError("guidance program ID must be non-empty")
        if self.format_version not in {1, 2, 3}:
            raise ValueError("unsupported guidance-program format version")
        if not segments or any(
            not isinstance(segment, GuidanceSegment) for segment in segments
        ):
            raise ValueError("guidance program requires GuidanceSegment objects")
        identifiers = [segment.segment_id for segment in segments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("guidance segment IDs must be unique")
        if any(segment.transition is None for segment in segments[:-1]):
            raise ValueError("every non-final segment requires a transition guard")
        if segments[-1].transition is not None:
            raise ValueError("the final guidance segment must not have a transition")
        speed_stages = tuple(self.speed_stages)
        if any(not isinstance(stage, SpeedGuidanceStage) for stage in speed_stages):
            raise ValueError(
                "independent speed guidance requires SpeedGuidanceStage objects"
            )
        if speed_stages and self.format_version not in {2, 3}:
            raise ValueError(
                "independent speed guidance requires program format version 2"
            )
        stage_identifiers = [stage.stage_id for stage in speed_stages]
        if len(stage_identifiers) != len(set(stage_identifiers)):
            raise ValueError("speed-guidance stage IDs must be unique")
        if speed_stages and any(
            stage.transition is None for stage in speed_stages[:-1]
        ):
            raise ValueError("every non-final speed stage requires a transition guard")
        if speed_stages and speed_stages[-1].transition is not None:
            raise ValueError(
                "the final speed-guidance stage must not have a transition"
            )
        vertical_stages = tuple(self.vertical_stages)
        if any(
            not isinstance(stage, VerticalGuidanceStage) for stage in vertical_stages
        ):
            raise ValueError(
                "independent vertical guidance requires VerticalGuidanceStage objects"
            )
        if vertical_stages and self.format_version != 3:
            raise ValueError(
                "independent vertical guidance requires program format version 3"
            )
        vertical_identifiers = [stage.stage_id for stage in vertical_stages]
        if len(vertical_identifiers) != len(set(vertical_identifiers)):
            raise ValueError("vertical-guidance stage IDs must be unique")
        if vertical_stages and any(
            stage.transition is None for stage in vertical_stages[:-1]
        ):
            raise ValueError(
                "every non-final vertical stage requires a transition guard"
            )
        if vertical_stages and vertical_stages[-1].transition is not None:
            raise ValueError(
                "the final vertical-guidance stage must not have a transition"
            )
        metadata = dict(self.metadata)
        if any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("guidance metadata must map non-empty strings to strings")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "speed_stages", speed_stages)
        object.__setattr__(self, "vertical_stages", vertical_stages)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def advance(self, segment_index: int, observation: GuidanceObservation) -> int:
        """Advance a stateful executor while the active transition is satisfied."""

        if not isinstance(segment_index, int) or not 0 <= segment_index < len(
            self.segments
        ):
            raise ValueError("active guidance segment index is out of range")
        index = segment_index
        while index < len(self.segments) - 1:
            transition = self.segments[index].transition
            if transition is None or not transition.matches(observation):
                break
            index += 1
        return index

    def to_dict(self) -> dict[str, Any]:
        """Return a structure containing only YAML-safe primitive values."""

        result = {
            "format_version": self.format_version,
            "program_id": self.program_id,
            "metadata": dict(self.metadata),
            "segments": [_segment_to_dict(segment) for segment in self.segments],
        }
        if self.speed_stages:
            result["speed_stages"] = [
                _speed_stage_to_dict(stage) for stage in self.speed_stages
            ]
        if self.vertical_stages:
            result["vertical_stages"] = [
                _vertical_stage_to_dict(stage) for stage in self.vertical_stages
            ]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GuidanceProgram:
        """Validate and construct a program from YAML-safe primitive values."""

        data = _mapping(value, "guidance program")
        _reject_unknown(
            data,
            {
                "format_version",
                "program_id",
                "metadata",
                "segments",
                "speed_stages",
                "vertical_stages",
            },
            "guidance program",
        )
        if "program_id" not in data or "segments" not in data:
            raise ValueError("guidance program requires program_id and segments")
        raw_segments = data["segments"]
        if not isinstance(raw_segments, list):
            raise TypeError("guidance program segments must be a list")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("guidance program metadata must be a mapping")
        raw_speed_stages = data.get("speed_stages", [])
        if not isinstance(raw_speed_stages, list):
            raise TypeError("guidance program speed_stages must be a list")
        raw_vertical_stages = data.get("vertical_stages", [])
        if not isinstance(raw_vertical_stages, list):
            raise TypeError("guidance program vertical_stages must be a list")
        return cls(
            program_id=data["program_id"],
            segments=tuple(_segment_from_dict(item) for item in raw_segments),
            format_version=data.get("format_version", 1),
            metadata=metadata,
            speed_stages=tuple(
                _speed_stage_from_dict(item) for item in raw_speed_stages
            ),
            vertical_stages=tuple(
                _vertical_stage_from_dict(item) for item in raw_vertical_stages
            ),
        )

    def to_yaml(self) -> str:
        """Serialize the program with safe, deterministic YAML primitives."""

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, value: str) -> GuidanceProgram:
        """Deserialize a program using PyYAML's safe loader."""

        if not isinstance(value, str):
            raise TypeError("guidance-program YAML must be a string")
        loaded = yaml.safe_load(value)
        if not isinstance(loaded, Mapping):
            raise ValueError("guidance-program YAML must contain one mapping")
        return cls.from_dict(loaded)

    def compile_selected_guidance(self) -> CompiledGuidanceProgram:
        """Compile the executable timed-target guidance subset."""

        return compile_guidance_program(self)

    def build_intent_guidance(self, lateral_guidance: Any, **settings: Any) -> Any:
        """Bind this program to intent-driven trajectory generation.

        Unlike :meth:`compile_selected_guidance`, this adapter evaluates the
        program's transition guards against the generated aircraft state. It
        therefore preserves spatial, altitude-capture, and distance-to-go
        semantics instead of lowering them to an elapsed-time schedule.
        """

        from aerox.guidance.program import IntentDrivenGuidance

        return IntentDrivenGuidance(
            program=self,
            lateral_guidance=lateral_guidance,
            **settings,
        )

    def build_online_guidance(self, lateral_guidance: Any, **settings: Any) -> Any:
        """Compatibility alias for :meth:`build_intent_guidance`."""

        return self.build_intent_guidance(lateral_guidance, **settings)


@dataclass(frozen=True)
class GuidanceTransitionEvent:
    """One transition taken by a stateful guidance-program executor."""

    from_segment_index: int
    to_segment_index: int
    observation: GuidanceObservation

    def __post_init__(self) -> None:
        if (
            not isinstance(self.from_segment_index, int)
            or not isinstance(self.to_segment_index, int)
            or self.from_segment_index < 0
            or self.to_segment_index != self.from_segment_index + 1
        ):
            raise ValueError("guidance transition indices must be consecutive")
        if not isinstance(self.observation, GuidanceObservation):
            raise TypeError("transition event requires a GuidanceObservation")


@dataclass
class GuidanceProgramExecutor:
    """Advance joint and independent guidance channels from observations."""

    program: GuidanceProgram
    _segment_index: int = field(default=0, init=False, repr=False)
    _transition_events: list[GuidanceTransitionEvent] = field(
        default_factory=list, init=False, repr=False
    )
    _speed_stage_index: int = field(default=0, init=False, repr=False)
    _speed_transition_events: list[GuidanceTransitionEvent] = field(
        default_factory=list, init=False, repr=False
    )
    _vertical_stage_index: int = field(default=0, init=False, repr=False)
    _vertical_transition_events: list[GuidanceTransitionEvent] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.program, GuidanceProgram):
            raise TypeError("executor program must be a GuidanceProgram")

    @property
    def segment_index(self) -> int:
        """Return the zero-based active segment index."""

        return self._segment_index

    @property
    def current_segment(self) -> GuidanceSegment:
        """Return the active concurrent guidance commands."""

        return self.program.segments[self._segment_index]

    @property
    def speed_stage_index(self) -> int:
        """Return the zero-based active independent speed-stage index."""

        return self._speed_stage_index

    @property
    def current_speed(self) -> SpeedCommand:
        """Return the independent speed command or the joint-segment command."""

        if self.program.speed_stages:
            return self.program.speed_stages[self._speed_stage_index].command
        return self.current_segment.speed

    @property
    def vertical_stage_index(self) -> int:
        """Return the zero-based active independent vertical-stage index."""

        return self._vertical_stage_index

    @property
    def current_vertical(self) -> VerticalCommand:
        """Return the independent vertical command or joint-segment command."""

        if self.program.vertical_stages:
            return self.program.vertical_stages[self._vertical_stage_index].command
        return self.current_segment.vertical

    @property
    def transition_events(self) -> tuple[GuidanceTransitionEvent, ...]:
        """Return an immutable snapshot of transitions taken since reset."""

        return tuple(self._transition_events)

    @property
    def speed_transition_events(self) -> tuple[GuidanceTransitionEvent, ...]:
        """Return transitions taken by the independent speed channel."""

        return tuple(self._speed_transition_events)

    @property
    def vertical_transition_events(self) -> tuple[GuidanceTransitionEvent, ...]:
        """Return transitions taken by the independent vertical channel."""

        return tuple(self._vertical_transition_events)

    def reset(self, segment_index: int = 0) -> None:
        """Reset execution to one program segment and clear event history."""

        if not isinstance(segment_index, int) or not 0 <= segment_index < len(
            self.program.segments
        ):
            raise ValueError("active guidance segment index is out of range")
        self._segment_index = segment_index
        self._speed_stage_index = 0
        self._vertical_stage_index = 0
        self._transition_events.clear()
        self._speed_transition_events.clear()
        self._vertical_transition_events.clear()

    def update(self, observation: GuidanceObservation) -> GuidanceSegment:
        """Apply all satisfied sequential guards and return the active segment."""

        if not isinstance(observation, GuidanceObservation):
            raise TypeError("executor update requires a GuidanceObservation")
        while self._segment_index < len(self.program.segments) - 1:
            transition = self.current_segment.transition
            if transition is None or not transition.matches(observation):
                break
            previous = self._segment_index
            self._segment_index += 1
            self._transition_events.append(
                GuidanceTransitionEvent(previous, self._segment_index, observation)
            )
        while (
            self.program.speed_stages
            and self._speed_stage_index < len(self.program.speed_stages) - 1
        ):
            transition = self.program.speed_stages[self._speed_stage_index].transition
            if transition is None or not transition.matches(observation):
                break
            previous = self._speed_stage_index
            self._speed_stage_index += 1
            self._speed_transition_events.append(
                GuidanceTransitionEvent(
                    previous,
                    self._speed_stage_index,
                    observation,
                )
            )
        while (
            self.program.vertical_stages
            and self._vertical_stage_index < len(self.program.vertical_stages) - 1
        ):
            transition = self.program.vertical_stages[
                self._vertical_stage_index
            ].transition
            if transition is None or not transition.matches(observation):
                break
            previous = self._vertical_stage_index
            self._vertical_stage_index += 1
            self._vertical_transition_events.append(
                GuidanceTransitionEvent(
                    previous,
                    self._vertical_stage_index,
                    observation,
                )
            )
            next_transition = self.program.vertical_stages[
                self._vertical_stage_index
            ].transition
            if next_transition is not None and any(
                condition.metric
                in {GuardMetric.ALTITUDE_ERROR_M, GuardMetric.SPEED_ERROR_MPS}
                for condition in next_transition.conditions
            ):
                # Command-relative errors in ``observation`` belong to the stage
                # that was active when the observation was built. Re-evaluate a
                # newly active command-relative guard on the next guidance update.
                break
        return self.current_segment


@dataclass(frozen=True)
class CompiledGuidanceProgram:
    """A program lowered to timed aerox guidance command channels."""

    source: GuidanceProgram
    effective_time_s: np.ndarray
    speed_effective_time_s: np.ndarray
    vertical_effective_time_s: np.ndarray
    altitude_intent: SelectedAltitudeIntent | None
    vertical_intent: SelectedVerticalIntent | None
    speed_intent: SelectedSpeedIntent
    speed_guidance_intent: SelectedSpeedGuidanceIntent
    thrust_intent: SelectedThrustIntent
    configuration_intent: SelectedConfigurationIntent

    def build_guidance(self, lateral_guidance: Any, **settings: Any) -> Any:
        """Bind compiled commands to an existing SpatialPathGuidance instance."""

        forbidden = {
            "lateral_guidance",
            "altitude_intent",
            "speed_intent",
            "speed_guidance_intent",
            "thrust_intent",
            "configuration_intent",
            "vertical_intent",
        }.intersection(settings)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise TypeError(f"compiled guidance owns these settings: {names}")
        from aerox.guidance.selected_altitude import SelectedAltitudeGuidance

        return SelectedAltitudeGuidance(
            lateral_guidance=lateral_guidance,
            altitude_intent=self.altitude_intent,
            vertical_intent=self.vertical_intent,
            speed_intent=self.speed_intent,
            speed_guidance_intent=self.speed_guidance_intent,
            thrust_intent=self.thrust_intent,
            configuration_intent=self.configuration_intent,
            **settings,
        )


def _absolute_transition_time(transition: TransitionGuard) -> float:
    if transition.logic != GuardLogic.ALL or len(transition.conditions) != 1:
        raise ValueError(
            "selected-guidance compilation requires one elapsed-time condition per "
            "transition"
        )
    condition = transition.conditions[0]
    if (
        condition.metric != GuardMetric.ELAPSED_TIME_S
        or condition.comparison != GuardComparison.GE
        or condition.threshold <= 0.0
    ):
        raise ValueError(
            "selected-guidance compilation requires ELAPSED_TIME_S GE transitions"
        )
    return condition.threshold


def compile_guidance_program(program: GuidanceProgram) -> CompiledGuidanceProgram:
    """Lower a timed-target program into the current aerox guidance interfaces.

    The schema permits state guards and lateral modes that the timed adapter
    cannot represent. This compiler rejects them instead of changing their
    meaning.
    """

    if not isinstance(program, GuidanceProgram):
        raise TypeError("program must be a GuidanceProgram")
    times = [0.0]
    for segment in program.segments[:-1]:
        if segment.transition is None:
            raise ValueError("non-final segment is missing a transition")
        times.append(_absolute_transition_time(segment.transition))
    if any(second <= first for first, second in zip(times, times[1:], strict=False)):
        raise ValueError(
            "compiled elapsed-time transitions must be strictly increasing"
        )

    for segment in program.segments:
        if segment.lateral.mode != LateralMode.ROUTE:
            raise ValueError(
                "selected-guidance compilation currently supports ROUTE lateral mode"
            )
        if segment.speed.mode == SpeedMode.PROFILE:
            raise ValueError(
                "selected-guidance compilation requires explicit CAS/Mach selectors"
            )
        if segment.thrust.mode not in {
            ThrustMode.AUTO,
            ThrustMode.THRUST_FRACTION,
            ThrustMode.THRUST_CEILING_FRACTION,
        }:
            raise ValueError(
                "selected-guidance compilation supports AUTO and normalized "
                "thrust-fraction modes"
            )
    if any(stage.command.mode == SpeedMode.PROFILE for stage in program.speed_stages):
        raise ValueError(
            "selected-guidance compilation requires explicit speed-stage selectors"
        )

    if program.vertical_stages:
        vertical_times = [0.0]
        for stage in program.vertical_stages[:-1]:
            if stage.transition is None:
                raise ValueError("non-final vertical stage is missing a transition")
            vertical_times.append(_absolute_transition_time(stage.transition))
        if any(
            second <= first
            for first, second in zip(vertical_times, vertical_times[1:], strict=False)
        ):
            raise ValueError(
                "compiled vertical-stage transitions must be strictly increasing"
            )
        vertical_commands = tuple(stage.command for stage in program.vertical_stages)
        vertical_effective_time = np.asarray(vertical_times, dtype=float)
    else:
        vertical_commands = tuple(segment.vertical for segment in program.segments)
        vertical_effective_time = np.asarray(times, dtype=float)

    vertical_modes = tuple(command.mode.value for command in vertical_commands)
    direct_vertical = any(
        command.mode
        in {
            VerticalMode.PATH,
            VerticalMode.VERTICAL_SPEED,
            VerticalMode.FLIGHT_PATH_ANGLE,
        }
        for command in vertical_commands
    )
    altitude = np.asarray(
        [
            np.nan if command.altitude_m is None else command.altitude_m
            for command in vertical_commands
        ],
        dtype=float,
    )
    vertical_speed = np.asarray(
        [
            np.nan if command.vertical_speed_mps is None else command.vertical_speed_mps
            for command in vertical_commands
        ],
        dtype=float,
    )
    flight_path_angle = np.asarray(
        [
            np.nan
            if command.flight_path_angle_rad is None
            else command.flight_path_angle_rad
            for command in vertical_commands
        ],
        dtype=float,
    )
    thrust_modes = tuple(segment.thrust.mode.value for segment in program.segments)
    thrust_fractions = np.asarray(
        [
            np.nan
            if segment.thrust.available_thrust_fraction is None
            else segment.thrust.available_thrust_fraction
            for segment in program.segments
        ],
        dtype=float,
    )
    configuration_modes = tuple(
        segment.configuration.mode.value for segment in program.segments
    )
    drag_increments = np.asarray(
        [
            segment.configuration.drag_coefficient_increment
            for segment in program.segments
        ],
        dtype=float,
    )
    effective_time = np.asarray(times, dtype=float)
    if program.speed_stages:
        speed_times = [0.0]
        for stage in program.speed_stages[:-1]:
            if stage.transition is None:
                raise ValueError("non-final speed stage is missing a transition")
            speed_times.append(_absolute_transition_time(stage.transition))
        if any(
            second <= first
            for first, second in zip(speed_times, speed_times[1:], strict=False)
        ):
            raise ValueError(
                "compiled speed-stage transitions must be strictly increasing"
            )
        speed_commands = tuple(stage.command for stage in program.speed_stages)
        speed_effective_time = np.asarray(speed_times, dtype=float)
    else:
        speed_commands = tuple(segment.speed for segment in program.segments)
        speed_effective_time = effective_time
    speed_cas = np.asarray(
        [command.calibrated_airspeed_mps for command in speed_commands],
        dtype=float,
    )
    speed_mach = np.asarray(
        [command.mach for command in speed_commands],
        dtype=float,
    )
    independent_speed_modes = tuple(command.mode.value for command in speed_commands)
    independent_speed_selectors = tuple(
        None if command.active_selector is None else command.active_selector.value
        for command in speed_commands
    )
    return CompiledGuidanceProgram(
        source=program,
        effective_time_s=effective_time,
        speed_effective_time_s=speed_effective_time,
        vertical_effective_time_s=vertical_effective_time,
        altitude_intent=(
            None
            if direct_vertical
            else SelectedAltitudeIntent(vertical_effective_time, altitude)
        ),
        vertical_intent=(
            SelectedVerticalIntent(
                vertical_effective_time,
                vertical_modes,
                altitude,
                vertical_speed,
                flight_path_angle,
            )
            if direct_vertical
            else None
        ),
        speed_intent=SelectedSpeedIntent(
            speed_effective_time,
            speed_cas,
            speed_mach,
        ),
        speed_guidance_intent=SelectedSpeedGuidanceIntent(
            speed_effective_time,
            independent_speed_modes,
            independent_speed_selectors,
        ),
        thrust_intent=SelectedThrustIntent(
            effective_time, thrust_modes, thrust_fractions
        ),
        configuration_intent=SelectedConfigurationIntent(
            effective_time,
            configuration_modes,
            drag_increments,
        ),
    )


def _evidence_to_dict(evidence: GuidanceEvidence) -> dict[str, Any]:
    return {
        "confidence": evidence.confidence,
        "source_ids": list(evidence.source_ids),
        "alternatives": [
            {"label": item.label, "probability": item.probability}
            for item in evidence.alternatives
        ],
    }


def _evidence_from_dict(value: Any) -> GuidanceEvidence:
    data = _mapping(value, "guidance evidence")
    _reject_unknown(data, {"confidence", "source_ids", "alternatives"}, "evidence")
    raw_sources = data.get("source_ids", [])
    raw_alternatives = data.get("alternatives", [])
    if not isinstance(raw_sources, list) or not isinstance(raw_alternatives, list):
        raise TypeError("evidence sources and alternatives must be lists")
    alternatives = []
    for item in raw_alternatives:
        alternative = _mapping(item, "guidance alternative")
        _reject_unknown(alternative, {"label", "probability"}, "alternative")
        if "label" not in alternative or "probability" not in alternative:
            raise ValueError("guidance alternative requires label and probability")
        alternatives.append(
            GuidanceAlternative(alternative["label"], alternative["probability"])
        )
    return GuidanceEvidence(
        confidence=data.get("confidence", 1.0),
        source_ids=tuple(raw_sources),
        alternatives=tuple(alternatives),
    )


def _command_to_dict(command: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "mode": command.mode.value,
        "evidence": _evidence_to_dict(command.evidence),
    }
    for name in (
        "waypoint_index",
        "track_rad",
        "heading_rad",
        "altitude_m",
        "vertical_speed_mps",
        "flight_path_angle_rad",
        "calibrated_airspeed_mps",
        "mach",
        "active_selector",
        "net_thrust_n",
        "available_thrust_fraction",
        "drag_coefficient_increment",
    ):
        if hasattr(command, name) and getattr(command, name) is not None:
            value = getattr(command, name)
            data[name] = value.value if isinstance(value, StrEnum) else value
    return data


def _command_from_dict(value: Any, command_type: type[Any]) -> Any:
    data = _mapping(value, f"{command_type.__name__}")
    fields = {
        LateralCommand: {"mode", "waypoint_index", "track_rad", "heading_rad"},
        VerticalCommand: {
            "mode",
            "altitude_m",
            "vertical_speed_mps",
            "flight_path_angle_rad",
        },
        SpeedCommand: {
            "mode",
            "calibrated_airspeed_mps",
            "mach",
            "active_selector",
        },
        ThrustCommand: {"mode", "net_thrust_n", "available_thrust_fraction"},
        ConfigurationCommand: {"mode", "drag_coefficient_increment"},
    }[command_type]
    _reject_unknown(data, fields | {"evidence"}, command_type.__name__)
    if "mode" not in data:
        raise ValueError(f"{command_type.__name__} requires mode")
    arguments = {name: data[name] for name in fields if name in data}
    arguments["evidence"] = _evidence_from_dict(data.get("evidence", {}))
    return command_type(**arguments)


def _transition_to_dict(transition: TransitionGuard) -> dict[str, Any]:
    return {
        "logic": transition.logic.value,
        "conditions": [
            {
                "metric": condition.metric.value,
                "comparison": condition.comparison.value,
                "threshold": condition.threshold,
            }
            for condition in transition.conditions
        ],
        "evidence": _evidence_to_dict(transition.evidence),
    }


def _transition_from_dict(value: Any) -> TransitionGuard:
    data = _mapping(value, "transition guard")
    _reject_unknown(data, {"logic", "conditions", "evidence"}, "transition guard")
    raw_conditions = data.get("conditions")
    if not isinstance(raw_conditions, list):
        raise TypeError("transition conditions must be a list")
    conditions = []
    for item in raw_conditions:
        condition = _mapping(item, "guard condition")
        _reject_unknown(
            condition,
            {"metric", "comparison", "threshold"},
            "guard condition",
        )
        if not {"metric", "comparison", "threshold"}.issubset(condition):
            raise ValueError("guard condition requires metric, comparison, threshold")
        conditions.append(
            GuardCondition(
                metric=condition["metric"],
                comparison=condition["comparison"],
                threshold=condition["threshold"],
            )
        )
    return TransitionGuard(
        conditions=tuple(conditions),
        logic=data.get("logic", GuardLogic.ALL),
        evidence=_evidence_from_dict(data.get("evidence", {})),
    )


def _segment_to_dict(segment: GuidanceSegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "lateral": _command_to_dict(segment.lateral),
        "vertical": _command_to_dict(segment.vertical),
        "speed": _command_to_dict(segment.speed),
        "thrust": _command_to_dict(segment.thrust),
        "configuration": _command_to_dict(segment.configuration),
        "transition": (
            None
            if segment.transition is None
            else _transition_to_dict(segment.transition)
        ),
    }


def _segment_from_dict(value: Any) -> GuidanceSegment:
    data = _mapping(value, "guidance segment")
    required = {"segment_id", "lateral", "vertical", "speed"}
    _reject_unknown(
        data,
        required | {"thrust", "configuration", "transition"},
        "guidance segment",
    )
    if not required.issubset(data):
        raise ValueError("guidance segment requires ID and all primary axis commands")
    transition = data.get("transition")
    return GuidanceSegment(
        segment_id=data["segment_id"],
        lateral=_command_from_dict(data["lateral"], LateralCommand),
        vertical=_command_from_dict(data["vertical"], VerticalCommand),
        speed=_command_from_dict(data["speed"], SpeedCommand),
        thrust=_command_from_dict(
            data.get("thrust", {"mode": ThrustMode.AUTO.value}), ThrustCommand
        ),
        configuration=_command_from_dict(
            data.get("configuration", {"mode": ConfigurationMode.CLEAN.value}),
            ConfigurationCommand,
        ),
        transition=(None if transition is None else _transition_from_dict(transition)),
    )


def _speed_stage_to_dict(stage: SpeedGuidanceStage) -> dict[str, Any]:
    return {
        "stage_id": stage.stage_id,
        "command": _command_to_dict(stage.command),
        "transition": (
            None if stage.transition is None else _transition_to_dict(stage.transition)
        ),
    }


def _speed_stage_from_dict(value: Any) -> SpeedGuidanceStage:
    data = _mapping(value, "speed-guidance stage")
    _reject_unknown(
        data,
        {"stage_id", "command", "transition"},
        "speed-guidance stage",
    )
    if "stage_id" not in data or "command" not in data:
        raise ValueError("speed-guidance stage requires ID and command")
    transition = data.get("transition")
    return SpeedGuidanceStage(
        stage_id=data["stage_id"],
        command=_command_from_dict(data["command"], SpeedCommand),
        transition=(None if transition is None else _transition_from_dict(transition)),
    )


def _vertical_stage_to_dict(stage: VerticalGuidanceStage) -> dict[str, Any]:
    return {
        "stage_id": stage.stage_id,
        "command": _command_to_dict(stage.command),
        "transition": (
            None if stage.transition is None else _transition_to_dict(stage.transition)
        ),
    }


def _vertical_stage_from_dict(value: Any) -> VerticalGuidanceStage:
    data = _mapping(value, "vertical-guidance stage")
    _reject_unknown(
        data,
        {"stage_id", "command", "transition"},
        "vertical-guidance stage",
    )
    if "stage_id" not in data or "command" not in data:
        raise ValueError("vertical-guidance stage requires ID and command")
    transition = data.get("transition")
    return VerticalGuidanceStage(
        stage_id=data["stage_id"],
        command=_command_from_dict(data["command"], VerticalCommand),
        transition=(None if transition is None else _transition_from_dict(transition)),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {name} fields: {fields}")


__all__ = [
    "CompiledGuidanceProgram",
    "ConfigurationCommand",
    "ConfigurationMode",
    "GuardComparison",
    "GuardCondition",
    "GuardLogic",
    "GuardMetric",
    "GuidanceAlternative",
    "GuidanceEvidence",
    "GuidanceObservation",
    "GuidanceProgram",
    "GuidanceProgramExecutor",
    "GuidanceSegment",
    "GuidanceTransitionEvent",
    "LateralCommand",
    "LateralMode",
    "SpeedCommand",
    "SpeedGuidanceStage",
    "SpeedMode",
    "SpeedSelectorMode",
    "ThrustCommand",
    "ThrustMode",
    "TransitionGuard",
    "VerticalCommand",
    "VerticalGuidanceStage",
    "VerticalMode",
    "compile_guidance_program",
]
