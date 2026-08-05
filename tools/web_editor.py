#!/usr/bin/env python3
"""
Simple Aircraft Data Editor - Flask Web App

A lightweight, no-nonsense web interface for editing aircraft YAML files.

Usage:
    python simple_app.py
    or: uv run --no-project --with flask --with pyyaml python simple_app.py

Then open: http://localhost:5000
"""

import csv
import json
import locale
import sys
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

# Import drag calculation functions in both script and package-test contexts.
try:
    import dragcoef
except ModuleNotFoundError:  # pragma: no cover - exercised by package-style import
    from tools import dragcoef
import numpy as np
import yaml
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

app = Flask(__name__)
YAML_BASE_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "aerox" / "data" / "aircraft" / "yml"
)
ENGINE_TYPES = ["turbofan", "turboprop", "piston"]
DEFAULT_ENGINE_TYPE = "turbofan"
TURBOFAN_CSV = Path(__file__).parent / "turbofan.csv"


def load_aircraft_classification(icao: str) -> dict:
    """Load aircraft classification from turbofan.csv."""
    if not TURBOFAN_CSV.exists():
        return {}

    icao_lower = icao.lower()
    with open(TURBOFAN_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["icao"] == icao_lower:
                return {
                    "civil": row["civil"] == "1",
                    "military": row["military"] == "1",
                    "business": row["business"] == "1",
                    "edb": row["edb"] == "1",
                    "flaps": row["flaps"] == "1",
                }
    return {}

# Field schema for configuration-driven parsing and validation
FIELD_SCHEMA = {
    "simple": {
        "aircraft": {"type": str},
        "icao": {"type": str},
        "iata": {"type": str},
        "mtow": {"type": float},
        "mlw": {"type": float},
        "oew": {"type": float},
        "mfc": {"type": float},
        "vmo": {"type": float},
        "mmo": {"type": float},
        "ceiling": {"type": float},
        "icao_wtc": {"type": str},
        "eu_wtc": {"type": str},
        "faa_wtc": {"type": str},
        "parking_area_m2": {"type": float},
    },
    "nested": {
        "pax": {
            "fields": ["max", "high", "low"],
            "types": {"max": int, "high": int, "low": int},
        },
        "fuselage": {
            "fields": ["length", "height", "width"],
            "types": {"length": float, "height": float, "width": float},
        },
        "wing": {
            "fields": [
                "area", "span", "mac", "sweep", "t/c", "root_chord",
                "tip_chord", "taper_ratio", "aspect_ratio", "dihedral",
            ],
            "types": {
                "area": float,
                "span": float,
                "mac": float,
                "sweep": float,
                "t/c": float,
                "root_chord": float,
                "tip_chord": float,
                "taper_ratio": float,
                "aspect_ratio": float,
                "dihedral": float,
            },
        },
        "horizontal_tail": {
            "fields": [
                "area", "span", "mac", "root_chord", "tip_chord",
                "taper_ratio", "sweep", "dihedral", "arm",
                "elevator_area_ratio",
            ],
            "types": {
                "area": float,
                "span": float,
                "mac": float,
                "root_chord": float,
                "tip_chord": float,
                "taper_ratio": float,
                "sweep": float,
                "dihedral": float,
                "arm": float,
                "elevator_area_ratio": float,
            },
        },
        "vertical_tail": {
            "fields": [
                "area", "height", "mac", "root_chord", "tip_chord",
                "taper_ratio", "sweep", "arm", "rudder_area_ratio",
            ],
            "types": {
                "area": float,
                "height": float,
                "mac": float,
                "root_chord": float,
                "tip_chord": float,
                "taper_ratio": float,
                "sweep": float,
                "arm": float,
                "rudder_area_ratio": float,
            },
        },
        "flaps": {
            "fields": ["type", "area", "bf/b", "lambda_f", "cf/c", "Sf/S"],
            "types": {
                "type": str,
                "area": float,
                "bf/b": float,
                "lambda_f": float,
                "cf/c": float,
                "Sf/S": float,
            },
        },
        "engine": {
            "fields": ["type", "mount", "number", "default", "options"],
            "types": {
                "type": str,
                "mount": str,
                "number": int,
                "default": str,
                "options": list,
            },
        },
        "drag": {
            "fields": ["cd0", "k", "e", "gears"],
            "types": {"cd0": float, "k": float, "e": float, "gears": float},
        },
        "kinematic": {
            "fields": [
                "takeoff_distance_m",
                "takeoff_v2",
                "climb_to_fl50_ias",
                "climb_to_fl50_vs",
                "climb_to_fl150_ias",
                "climb_to_fl150_vs",
                "climb_to_fl240_ias",
                "climb_to_fl240_vs",
                "mach_climb_mach",
                "mach_climb_vs",
                "cruise_mach",
                "cruise_range_nm",
                "cruise_tas",
                "descent_to_fl240_mach",
                "descent_to_fl240_vs",
                "descent_to_fl100_ias",
                "descent_to_fl100_vs",
                "approach_ias",
                "approach_vs",
                "landing_distance_m",
                "landing_ias",
            ],
            "types": {
                "takeoff_distance_m": int,
                "takeoff_v2": int,
                "climb_to_fl50_ias": int,
                "climb_to_fl50_vs": int,
                "climb_to_fl150_ias": int,
                "climb_to_fl150_vs": int,
                "climb_to_fl240_ias": int,
                "climb_to_fl240_vs": int,
                "mach_climb_mach": float,
                "mach_climb_vs": int,
                "cruise_mach": float,
                "cruise_range_nm": int,
                "cruise_tas": int,
                "descent_to_fl240_mach": float,
                "descent_to_fl240_vs": int,
                "descent_to_fl100_ias": int,
                "descent_to_fl100_vs": int,
                "approach_ias": int,
                "approach_vs": int,
                "landing_distance_m": int,
                "landing_ias": int,
            },
        },
    },
}


# HTML Template - Simple and clean
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Aircraft Data Editor</title>
    <meta charset="UTF-8">
    <script src="https://code.jquery.com/jquery-3.7.1.min.js" integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo=" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 10px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 5px;
            font-size: 22px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 10px;
            font-size: 12px;
        }
        .nav {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
            align-items: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .nav.sticky {
            background: white;
            border-bottom: 2px solid #dee2e6;
        }
        .nav-group {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .nav-group label {
            font-size: 12px;
            color: #666;
            font-weight: 500;
        }
        .aircraft-name {
            font-size: 14px;
            font-weight: 600;
            color: #333;
            margin-left: 8px;
            padding-left: 8px;
            border-left: 2px solid #dee2e6;
        }
        .btn-save {
            background: #28a745;
            margin-left: auto;
            font-weight: 600;
        }
        .btn-save:hover {
            background: #218838;
        }
        .btn-calculate {
            background: #17a2b8;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 500;
        }
        .btn-calculate:hover {
            background: #138496;
        }
        .btn-calculate:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .aircraft-tags {
            display: inline-flex;
            gap: 6px;
            margin-left: 12px;
        }
        .aircraft-tag {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 500;
            text-transform: uppercase;
        }
        .tag-civil { background: #d4edda; color: #155724; }
        .tag-military { background: #f8d7da; color: #721c24; }
        .tag-business { background: #cce5ff; color: #004085; }
        .tag-yes { background: #d4edda; color: #155724; }
        .tag-no { background: #f8d7da; color: #721c24; }
        .drag-preview {
            background: #e7f3ff;
            border: 2px solid #17a2b8;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .drag-preview.error {
            background: #f8d7da;
            border-color: #dc3545;
        }
        .preview-content {
            margin-bottom: 10px;
        }
        .preview-content strong {
            display: block;
            margin-bottom: 8px;
            color: #333;
        }
        .preview-values {
            font-family: monospace;
            font-size: 13px;
            color: #495057;
            line-height: 1.6;
        }
        .preview-error {
            color: #721c24;
            font-size: 13px;
            line-height: 1.6;
        }
        .btn-apply {
            background: #28a745;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 500;
            width: 100%;
        }
        .btn-apply:hover {
            background: #218838;
        }
        select, input, textarea, button {
            padding: 5px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 12px;
        }
        select {
            flex: 1;
            max-width: 200px;
        }
        #engine-select {
            max-width: 120px;
            font-weight: 500;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 500;
        }
        button:hover { background: #0056b3; }
        .btn-secondary {
            background: #6c757d;
            padding: 5px 10px;
        }
        .btn-secondary:hover { background: #545b62; }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        @media (max-width: 1400px) {
            .form-grid { grid-template-columns: repeat(3, 1fr); }
        }
        @media (max-width: 1100px) {
            .form-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 768px) {
            .form-grid { grid-template-columns: 1fr; }
            .nav { flex-direction: column; align-items: stretch; }
            .nav-group { justify-content: center; }
        }
        .form-section {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
        }
        .form-section h3 {
            color: #495057;
            margin-bottom: 8px;
            font-size: 13px;
            padding-bottom: 5px;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
        }
        .field {
            margin-bottom: 3px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .field label {
            color: #495057;
            font-size: 12px;
            white-space: nowrap;
            min-width: 70px;
            text-align: right;
        }
        .field input, .field textarea {
            flex: 1;
            min-width: 0;
        }
        textarea {
            min-height: 50px;
            font-family: monospace;
            resize: vertical;
        }
        .field.textarea-field {
            flex-direction: column;
            align-items: stretch;
        }
        .field.textarea-field label {
            text-align: left;
            margin-bottom: 3px;
        }
        .save-bar {
            position: sticky;
            bottom: 0;
            background: white;
            padding: 20px 0;
            border-top: 2px solid #dee2e6;
            margin-top: 30px;
        }
        .save-bar button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            background: #28a745;
        }
        .save-bar button:hover {
            background: #218838;
        }
        .message {
            padding: 12px 20px;
            border-radius: 4px;
            font-weight: 500;
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            min-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        .success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .input-group {
            display: flex;
            flex: 1;
            min-width: 0;
        }
        .input-group input {
            flex: 1;
            min-width: 0;
            border-top-right-radius: 0;
            border-bottom-right-radius: 0;
            border-right: none;
        }
        .input-append {
            background: #e9ecef;
            border: 1px solid #ddd;
            border-left: none;
            border-radius: 0 4px 4px 0;
            padding: 5px 8px;
            font-size: 11px;
            color: #6c757d;
            white-space: nowrap;
            display: flex;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" style="float: right; padding: 6px 12px; background: #6c757d; color: white; text-decoration: none; border-radius: 4px; font-size: 12px; font-weight: 500;"><i class="fa-solid fa-list"></i> Overview</a>
        <h1><i class="fa-solid fa-plane"></i> Aircraft Data Editor</h1>
        <p class="subtitle">Simple web interface for editing aircraft YAML files</p>
        
        {% if message %}
        <div class="message {{ message_type }}">{{ message }}</div>
        {% endif %}
        
        <div class="nav">
            <div class="nav-group">
                <label>Engine:</label>
                <select id="engine-select" onchange="changeEngineType(this.value)">
                    {% for et in engine_types %}
                    <option value="{{ et }}" {% if et == current_engine_type %}selected{% endif %}>{{ et }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="nav-group">
                <label>Aircraft:</label>
                <select id="aircraft-select" onchange="loadAircraft(this.value)">
                    <option value="">Select...</option>
                    {% for icao in aircraft_by_type[current_engine_type] %}
                    <option value="{{ icao }}" {% if icao == current_icao %}selected{% endif %}>{{ icao }}</option>
                    {% endfor %}
                </select>
                <span id="aircraft-counter" style="color: #6c757d; font-size: 11px;"></span>
            </div>
            <div class="nav-group">
                <button onclick="prevAircraft()" class="btn-secondary"><i class="fa-solid fa-chevron-left"></i></button>
                <button onclick="nextAircraft()" class="btn-secondary"><i class="fa-solid fa-chevron-right"></i></button>
            </div>
            {% if data %}
            <span class="aircraft-name">{{ data.aircraft or current_icao }}</span>
            {% if classification %}
            <span class="aircraft-tags">
                {% if classification.civil %}<span class="aircraft-tag tag-civil">Civil</span>{% endif %}
                {% if classification.military %}<span class="aircraft-tag tag-military">Military</span>{% endif %}
                {% if classification.business %}<span class="aircraft-tag tag-business">Business</span>{% endif %}
                {% if classification.edb %}<span class="aircraft-tag tag-yes">EDB <i class="fa-solid fa-check"></i></span>{% else %}<span class="aircraft-tag tag-no">EDB <i class="fa-solid fa-xmark"></i></span>{% endif %}
                {% if classification.flaps %}<span class="aircraft-tag tag-yes">Flaps <i class="fa-solid fa-check"></i></span>{% else %}<span class="aircraft-tag tag-no">Flaps <i class="fa-solid fa-xmark"></i></span>{% endif %}
            </span>
            {% endif %}
            <input type="hidden" name="current_engine_type" form="edit-form" value="{{ current_engine_type }}">
            <button type="submit" form="edit-form" class="btn-save"><i class="fa-solid fa-floppy-disk"></i> Save</button>
            {% endif %}
        </div>
        
        {% if data %}
        <form method="POST" action="{{ url_for('save', icao=current_icao) }}" id="edit-form">
            <div class="form-grid">
                <!-- Column 1: Basic Info & Weight -->
                <div>
                    <div class="form-section">
                        <h3><i class="fa-solid fa-clipboard-list"></i> Basic Information</h3>
                        <div class="field">
                            <label>Aircraft Name</label>
                            <input type="text" name="aircraft" value="{{ data.aircraft or '' }}">
                        </div>
                        <div class="field">
                            <label>ICAO Code</label>
                            <input type="text" name="icao" value="{{ data.icao or '' }}" required>
                        </div>
                        <div class="field">
                            <label>IATA Code</label>
                            <input type="text" name="iata" value="{{ data.iata or '' }}">
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-weight-hanging"></i> Weight</h3>
                        <div class="field">
                            <label>MTOW</label>
                            <div class="input-group">
                                <input type="number" step="any" name="mtow" value="{{ data.mtow or '' }}">
                                <span class="input-append">kg</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>MLW</label>
                            <div class="input-group">
                                <input type="number" step="any" name="mlw" value="{{ data.mlw or '' }}">
                                <span class="input-append">kg</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>OEW</label>
                            <div class="input-group">
                                <input type="number" step="any" name="oew" value="{{ data.oew or '' }}">
                                <span class="input-append">kg</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>MFC</label>
                            <div class="input-group">
                                <input type="number" step="any" name="mfc" value="{{ data.mfc or '' }}">
                                <span class="input-append">L</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-gauge-high"></i> Performance</h3>
                        <div class="field">
                            <label>VMO</label>
                            <div class="input-group">
                                <input type="number" step="any" name="vmo" value="{{ data.vmo or '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>MMO</label>
                            <div class="input-group">
                                <input type="number" step="0.01" name="mmo" value="{{ data.mmo or '' }}">
                                <span class="input-append">M</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Ceiling</label>
                            <div class="input-group">
                                <input type="number" step="any" name="ceiling" value="{{ data.ceiling or '' }}">
                                <span class="input-append">ft</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-tags"></i> Classification</h3>
                        <div class="field">
                            <label>ICAO WTC</label>
                            <select name="icao_wtc">
                                <option value="">-</option>
                                {% for option in field_options.icao_wtc %}
                                <option value="{{ option }}" {% if data.icao_wtc == option %}selected{% endif %}>{{ option }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="field">
                            <label>EU WTC</label>
                            <select name="eu_wtc">
                                <option value="">-</option>
                                {% for option in field_options.eu_wtc %}
                                <option value="{{ option }}" {% if data.eu_wtc == option %}selected{% endif %}>{{ option }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="field">
                            <label>FAA WTC</label>
                            <select name="faa_wtc">
                                <option value="">-</option>
                                {% for option in field_options.faa_wtc %}
                                <option value="{{ option }}" {% if data.faa_wtc == option %}selected{% endif %}>{{ option }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="field">
                            <label>Parking Area</label>
                            <div class="input-group">
                                <input type="number" step="any" name="parking_area_m2" value="{{ data.parking_area_m2 or '' }}">
                                <span class="input-append">m²</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Column 2: Geometry -->
                <div>
                    <div class="form-section">
                        <h3><i class="fa-solid fa-users"></i> Passengers</h3>
                        <div class="field">
                            <label>Max</label>
                            <input type="number" name="pax__max" value="{{ data.pax.max if data.pax else '' }}">
                        </div>
                        <div class="field">
                            <label>High</label>
                            <input type="number" name="pax__high" value="{{ data.pax.high if data.pax else '' }}">
                        </div>
                        <div class="field">
                            <label>Low</label>
                            <input type="number" name="pax__low" value="{{ data.pax.low if data.pax else '' }}">
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-ruler-combined"></i> Fuselage</h3>
                        <div class="field">
                            <label>Length</label>
                            <div class="input-group">
                                <input type="number" step="0.01" name="fuselage__length" value="{{ data.fuselage.length if data.fuselage else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Height</label>
                            <div class="input-group">
                                <input type="number" step="0.01" name="fuselage__height" value="{{ data.fuselage.height if data.fuselage and data.fuselage.height else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Width</label>
                            <div class="input-group">
                                <input type="number" step="0.01" name="fuselage__width" value="{{ data.fuselage.width if data.fuselage and data.fuselage.width else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-plane"></i> Wing</h3>
                        <div class="field">
                            <label>Area</label>
                            <div class="input-group">
                                <input type="number" step="any" name="wing__area" value="{{ data.wing.area if data.wing and data.wing.area else '' }}">
                                <span class="input-append">m²</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Span</label>
                            <div class="input-group">
                                <input type="number" step="any" name="wing__span" value="{{ data.wing.span if data.wing and data.wing.span else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>MAC</label>
                            <div class="input-group">
                                <input type="number" step="any" name="wing__mac" value="{{ data.wing.mac if data.wing and data.wing.mac else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Sweep</label>
                            <div class="input-group">
                                <input type="number" step="any" name="wing__sweep" value="{{ data.wing.sweep if data.wing and data.wing.sweep is not none else '' }}">
                                <span class="input-append">°</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>t/c</label>
                            <input type="number" step="0.001" name="wing__t/c" value="{% if data.wing and 't/c' in data.wing and data.wing['t/c'] != None %}{{ '{:.3f}'.format(data.wing['t/c']) }}{% endif %}">
                        </div>
                        {% for field, label, unit in [
                            ('root_chord', 'Root chord', 'm'),
                            ('tip_chord', 'Tip chord', 'm'),
                            ('taper_ratio', 'Taper ratio', ''),
                            ('aspect_ratio', 'Aspect ratio', ''),
                            ('dihedral', 'Dihedral', '°')
                        ] %}
                        <div class="field">
                            <label>{{ label }}</label>
                            <div class="input-group">
                                <input type="number" step="any" name="wing__{{ field }}" value="{{ data.wing[field] if data.wing and data.wing[field] is not none else '' }}">
                                {% if unit %}<span class="input-append">{{ unit }}</span>{% endif %}
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                    <div class="form-section">
                        <h3><i class="fa-solid fa-plane"></i> Horizontal Tail</h3>
                        {% for field, label, unit in [
                            ('area', 'Area', 'm²'), ('span', 'Span', 'm'),
                            ('mac', 'MAC', 'm'), ('root_chord', 'Root chord', 'm'),
                            ('tip_chord', 'Tip chord', 'm'),
                            ('taper_ratio', 'Taper ratio', ''),
                            ('sweep', 'Quarter-chord sweep', '°'),
                            ('dihedral', 'Dihedral', '°'), ('arm', 'Moment arm', 'm'),
                            ('elevator_area_ratio', 'Elevator area ratio', '')
                        ] %}
                        <div class="field">
                            <label>{{ label }}</label>
                            <div class="input-group">
                                <input type="number" step="any" name="horizontal_tail__{{ field }}" value="{{ data.horizontal_tail[field] if data.horizontal_tail and data.horizontal_tail[field] is not none else '' }}">
                                {% if unit %}<span class="input-append">{{ unit }}</span>{% endif %}
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                    <div class="form-section">
                        <h3><i class="fa-solid fa-plane-up"></i> Vertical Tail</h3>
                        {% for field, label, unit in [
                            ('area', 'Area', 'm²'), ('height', 'Reference height', 'm'),
                            ('mac', 'MAC', 'm'), ('root_chord', 'Root chord', 'm'),
                            ('tip_chord', 'Tip chord', 'm'),
                            ('taper_ratio', 'Taper ratio', ''),
                            ('sweep', 'Quarter-chord sweep', '°'),
                            ('arm', 'Moment arm', 'm'),
                            ('rudder_area_ratio', 'Rudder area ratio', '')
                        ] %}
                        <div class="field">
                            <label>{{ label }}</label>
                            <div class="input-group">
                                <input type="number" step="any" name="vertical_tail__{{ field }}" value="{{ data.vertical_tail[field] if data.vertical_tail and data.vertical_tail[field] is not none else '' }}">
                                {% if unit %}<span class="input-append">{{ unit }}</span>{% endif %}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-sliders"></i> Flaps</h3>
                        <div class="field">
                            <label>Type</label>
                            <select name="flaps__type">
                                <option value="">-</option>
                                {% for option in field_options.flaps_type %}
                                <option value="{{ option }}" {% if data.flaps and data.flaps.type == option %}selected{% endif %}>{{ option }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="field">
                            <label>Area</label>
                            <div class="input-group">
                                <input type="number" step="0.01" name="flaps__area" value="{{ data.flaps.area if data.flaps and data.flaps.area else '' }}">
                                <span class="input-append">m²</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>bf/b</label>
                            <input type="number" step="0.01" name="flaps__bf/b" value="{{ data.flaps['bf/b'] if data.flaps and 'bf/b' in data.flaps and data.flaps['bf/b'] else '' }}">
                        </div>
                        <div class="field">
                            <label>lambda_f</label>
                            <input type="number" step="0.01" name="flaps__lambda_f" value="{{ data.flaps.lambda_f if data.flaps and data.flaps.lambda_f else '' }}">
                        </div>
                        <div class="field">
                            <label>cf/c</label>
                            <input type="number" step="0.01" name="flaps__cf/c" value="{{ data.flaps['cf/c'] if data.flaps and 'cf/c' in data.flaps and data.flaps['cf/c'] else '' }}">
                        </div>
                        <div class="field">
                            <label>Sf/S</label>
                            <input type="number" step="0.01" name="flaps__Sf/S" value="{{ data.flaps['Sf/S'] if data.flaps and 'Sf/S' in data.flaps and data.flaps['Sf/S'] else '' }}">
                        </div>
                    </div>
                </div>
                
                <!-- Column 3: Engine & Drag -->
                <div>
                    <div class="form-section">
                        <h3><i class="fa-solid fa-gear"></i> Engine</h3>
                        <div class="field">
                            <label>Type</label>
                            <select name="engine__type">
                                <option value="">-</option>
                                {% for option in field_options.engine_type %}
                                <option value="{{ option }}" {% if data.engine and data.engine.type == option %}selected{% endif %}>{{ option }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="field">
                            <label>Mount</label>
                            <input type="text" name="engine__mount" value="{{ data.engine.mount if data.engine else '' }}">
                        </div>
                        <div class="field">
                            <label>Number</label>
                            <input type="number" name="engine__number" value="{{ data.engine.number if data.engine else '' }}">
                        </div>
                        <div class="field">
                            <label>Default</label>
                            <input type="text" name="engine__default" value="{{ data.engine.default if data.engine else '' }}">
                        </div>
                        <div class="field textarea-field">
                            <label>Options <small style="font-weight: normal; color: #999;">(one per line)</small></label>
                            <textarea name="engine__options" rows="8">{% if data.engine and data.engine.options %}{{ data.engine.options | join('\n') }}{% endif %}</textarea>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-wind"></i> Drag Coefficients</h3>
                        <div class="field">
                            <button type="button" onclick="calculateDrag('{{ current_icao }}')" class="btn-calculate" style="width: 100%; margin-bottom: 10px;">
                                <i class="fa-solid fa-calculator"></i> Calculate Coefficients
                            </button>
                        </div>
                        <div id="drag-preview" class="drag-preview" style="display: none;">
                            <div class="preview-content">
                                <strong>Calculated Values:</strong>
                                <div class="preview-values">
                                    <div>CD0: <span id="preview_cd0">-</span></div>
                                    <div>K: <span id="preview_k">-</span></div>
                                    <div>e: <span id="preview_e">-</span></div>
                                    <div>Gears: <span id="preview_gears">-</span></div>
                                </div>
                            </div>
                            <button type="button" onclick="applyDragValues()" class="btn-apply">
                                <i class="fa-solid fa-check"></i> Apply to Fields
                            </button>
                        </div>
                        <div class="field">
                            <label>CD0</label>
                            <input type="number" step="0.00001" name="drag__cd0" id="drag_cd0" value="{{ data.drag.cd0 if data.drag and data.drag.cd0 else '' }}">
                        </div>
                        <div class="field">
                            <label>K</label>
                            <input type="number" step="0.00001" name="drag__k" id="drag_k" value="{{ data.drag.k if data.drag and data.drag.k else '' }}">
                        </div>
                        <div class="field">
                            <label>e</label>
                            <input type="number" step="0.00001" name="drag__e" id="drag_e" value="{{ data.drag.e if data.drag and data.drag.e else '' }}">
                        </div>
                        <div class="field">
                            <label>Gears</label>
                            <input type="number" step="0.00001" name="drag__gears" id="drag_gears" value="{{ data.drag.gears if data.drag and data.drag.gears else '' }}">
                        </div>
                    </div>
                </div>
                
                <!-- Column 4: Kinematic -->
                <div>
                    <div class="form-section">
                        <h3><i class="fa-solid fa-plane-departure"></i> Takeoff</h3>
                        <div class="field">
                            <label>Distance</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__takeoff_distance_m" value="{{ data.kinematic.takeoff_distance_m if data.kinematic else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>V2</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__takeoff_v2" value="{{ data.kinematic.takeoff_v2 if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-arrow-trend-up"></i> Climb</h3>
                        <div class="field">
                            <label>To FL50 IAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__climb_to_fl50_ias" value="{{ data.kinematic.climb_to_fl50_ias if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL50 VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__climb_to_fl50_vs" value="{{ data.kinematic.climb_to_fl50_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL150 IAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__climb_to_fl150_ias" value="{{ data.kinematic.climb_to_fl150_ias if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL150 VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__climb_to_fl150_vs" value="{{ data.kinematic.climb_to_fl150_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL240 IAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__climb_to_fl240_ias" value="{{ data.kinematic.climb_to_fl240_ias if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL240 VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__climb_to_fl240_vs" value="{{ data.kinematic.climb_to_fl240_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Mach Climb Mach</label>
                            <input type="number" step="0.01" name="kinematic__mach_climb_mach" value="{{ data.kinematic.mach_climb_mach if data.kinematic and data.kinematic.mach_climb_mach else '' }}">
                        </div>
                        <div class="field">
                            <label>Mach Climb VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__mach_climb_vs" value="{{ data.kinematic.mach_climb_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-plane"></i> Cruise</h3>
                        <div class="field">
                            <label>Cruise Mach</label>
                            <input type="number" step="0.01" name="kinematic__cruise_mach" value="{{ data.kinematic.cruise_mach if data.kinematic and data.kinematic.cruise_mach else '' }}">
                        </div>
                        <div class="field">
                            <label>Range</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__cruise_range_nm" value="{{ data.kinematic.cruise_range_nm if data.kinematic else '' }}">
                                <span class="input-append">nm</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>TAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__cruise_tas" value="{{ data.kinematic.cruise_tas if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-arrow-trend-down"></i> Descent</h3>
                        <div class="field">
                            <label>To FL240 Mach</label>
                            <input type="number" step="0.01" name="kinematic__descent_to_fl240_mach" value="{{ data.kinematic.descent_to_fl240_mach if data.kinematic and data.kinematic.descent_to_fl240_mach else '' }}">
                        </div>
                        <div class="field">
                            <label>To FL240 VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__descent_to_fl240_vs" value="{{ data.kinematic.descent_to_fl240_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL100 IAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__descent_to_fl100_ias" value="{{ data.kinematic.descent_to_fl100_ias if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>To FL100 VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__descent_to_fl100_vs" value="{{ data.kinematic.descent_to_fl100_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fa-solid fa-plane-arrival"></i> Approach & Landing</h3>
                        <div class="field">
                            <label>Approach IAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__approach_ias" value="{{ data.kinematic.approach_ias if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Approach VS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__approach_vs" value="{{ data.kinematic.approach_vs if data.kinematic else '' }}">
                                <span class="input-append">fpm</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Landing Dist.</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__landing_distance_m" value="{{ data.kinematic.landing_distance_m if data.kinematic else '' }}">
                                <span class="input-append">m</span>
                            </div>
                        </div>
                        <div class="field">
                            <label>Landing IAS</label>
                            <div class="input-group">
                                <input type="number" name="kinematic__landing_ias" value="{{ data.kinematic.landing_ias if data.kinematic else '' }}">
                                <span class="input-append">kts</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </form>
        {% else %}
        <p style="text-align: center; color: #666; padding: 40px;">
            <i class="fa-solid fa-arrow-left"></i> Select an aircraft from the dropdown to start editing
        </p>
        {% endif %}
    </div>
    
    <script>
        const aircraftByType = {{ aircraft_by_type | tojson }};
        let currentEngineType = '{{ current_engine_type }}';

        // Change engine type and update aircraft list
        function changeEngineType(engineType) {
            currentEngineType = engineType;
            const $aircraftSelect = $('#aircraft-select');
            const aircraftList = aircraftByType[engineType] || [];

            // Clear and repopulate aircraft dropdown
            $aircraftSelect.empty();
            $aircraftSelect.append('<option value="">Select...</option>');
            aircraftList.forEach(icao => {
                $aircraftSelect.append(`<option value="${icao}">${icao}</option>`);
            });

            // Update counter
            updateCounter();
        }

        // Update aircraft counter
        function updateCounter() {
            const $select = $('#aircraft-select');
            const $counter = $('#aircraft-counter');
            if ($counter.length) {
                const currentIndex = $select.prop('selectedIndex');
                const totalCount = $select.find('option').length - 1; // -1 for "Select..." option
                if (currentIndex > 0) {
                    $counter.text(`${currentIndex}/${totalCount}`);
                }
            }
        }
        
        // Auto-dismiss success message after 3 seconds
        $(document).ready(function() {
            const $message = $('.message.success');
            if ($message.length) {
                setTimeout(function() {
                    $message.fadeOut(300, function() {
                        $(this).remove();
                    });
                }, 3000);
            }
            
            // Initialize counter
            updateCounter();
        });
        
        // Save scroll position before navigating
        function saveScrollPosition() {
            sessionStorage.setItem('scrollPosition', $(window).scrollTop());
        }
        
        // Restore scroll position after page load
        $(window).on('load', function() {
            const savedPosition = sessionStorage.getItem('scrollPosition');
            if (savedPosition !== null) {
                $(window).scrollTop(parseInt(savedPosition));
                sessionStorage.removeItem('scrollPosition');
            }
        });
        
        function loadAircraft(icao) {
            if (icao) {
                saveScrollPosition();
                window.location.href = '/' + icao;
            }
        }
        
        function prevAircraft() {
            const $select = $('#aircraft-select');
            const currentIndex = $select.prop('selectedIndex');
            if (currentIndex > 1) { // > 1 because first option is "Select..."
                $select.prop('selectedIndex', currentIndex - 1);
                saveScrollPosition();
                updateCounter();
                loadAircraft($select.val());
            }
        }
        
        function nextAircraft() {
            const $select = $('#aircraft-select');
            const currentIndex = $select.prop('selectedIndex');
            if (currentIndex < $select.find('option').length - 1) {
                $select.prop('selectedIndex', currentIndex + 1);
                saveScrollPosition();
                updateCounter();
                loadAircraft($select.val());
            }
        }
        
        // Keyboard shortcuts
        $(document).on('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 's' || e.key === 'S') {
                    // Ctrl+S or Cmd+S to save
                    e.preventDefault();
                    const $form = $('#edit-form');
                    if ($form.length) {
                        $form.submit();
                    }
                }
            }
        });
        
        // Add sticky class when scrolling
        $(window).on('scroll', function() {
            const $nav = $('.nav');
            if ($nav.length) {
                if ($(window).scrollTop() > 100) {
                    $nav.addClass('sticky');
                } else {
                    $nav.removeClass('sticky');
                }
            }
        });
        
        // Convert number inputs with decimal steps to text inputs to avoid locale issues
        $(document).ready(function() {
            $('input[type="number"]').each(function() {
                const $input = $(this);
                const step = $input.attr('step');
                // If it has a decimal step (not an integer field), convert to text input
                if (step && (step.includes('.') || parseFloat(step) < 1)) {
                    $input.attr('type', 'text');
                    $input.attr('inputMode', 'decimal');
                    $input.attr('pattern', '[0-9]*[.]?[0-9]*');
                    
                    // Add validation on blur
                    $input.on('blur', function() {
                        // Replace comma with dot
                        let value = $(this).val().replace(',', '.');
                        $(this).val(value);
                        
                        // Validate it's a valid number
                        if (value && isNaN(parseFloat(value))) {
                            $(this).css('borderColor', 'red');
                        } else {
                            $(this).css('borderColor', '');
                        }
                    });
                }
            });
        });
        
        // Store calculated drag values
        let calculatedDragValues = null;
        
        // Calculate drag coefficients
        async function calculateDrag(icao) {
            const $button = $(event.target);
            const $previewDiv = $('#drag-preview');
            
            $button.prop('disabled', true);
            $button.html('<i class="fa-solid fa-spinner fa-spin"></i> Calculating...');
            $previewDiv.hide();
            
            try {
                const response = await $.ajax({
                    url: `/${icao}/calculate-drag`,
                    method: 'POST',
                    contentType: 'application/json',
                });
                
                if (response.error) {
                    // Show error in preview div
                    $previewDiv.removeClass('error').addClass('error');
                    $previewDiv.html(`
                        <div class="preview-content">
                            <strong><i class="fa-solid fa-circle-exclamation"></i> Error:</strong>
                            <div class="preview-error">${response.error}</div>
                        </div>
                    `);
                    $previewDiv.show();
                } else {
                    // Store calculated values
                    calculatedDragValues = response;
                    
                    // Update preview display with success
                    $previewDiv.removeClass('error');
                    $previewDiv.html(`
                        <div class="preview-content">
                            <strong>Calculated Values:</strong>
                            <div class="preview-values">
                                <div>CD0: <span id="preview_cd0">${response.cd0.toFixed(5)}</span></div>
                                <div>K: <span id="preview_k">${response.k.toFixed(5)}</span></div>
                                <div>e: <span id="preview_e">${response.e.toFixed(5)}</span></div>
                                <div>Gears: <span id="preview_gears">${response.gears.toFixed(5)}</span></div>
                            </div>
                        </div>
                        <button type="button" onclick="applyDragValues()" class="btn-apply">
                            <i class="fa-solid fa-check"></i> Apply to Fields
                        </button>
                    `);
                    $previewDiv.show();
                }
            } catch (error) {
                // Show error in preview div for network/other errors
                $previewDiv.removeClass('error').addClass('error');
                const errorMessage = error.responseJSON?.error || error.message || 'Unknown error occurred';
                $previewDiv.html(`
                    <div class="preview-content">
                        <strong><i class="fa-solid fa-circle-exclamation"></i> Error:</strong>
                        <div class="preview-error">${errorMessage}</div>
                    </div>
                `);
                $previewDiv.show();
            } finally {
                $button.prop('disabled', false);
                $button.html('<i class="fa-solid fa-calculator"></i> Calculate Coefficients');
            }
        }
        
        // Apply calculated values to form fields
        function applyDragValues() {
            if (calculatedDragValues) {
                $('#drag_cd0').val(calculatedDragValues.cd0.toFixed(5));
                $('#drag_k').val(calculatedDragValues.k.toFixed(5));
                $('#drag_e').val(calculatedDragValues.e.toFixed(5));
                $('#drag_gears').val(calculatedDragValues.gears.toFixed(5));
                
                // Optional: Hide preview after applying
                // $('#drag-preview').hide();
            }
        }
    </script>
</body>
</html>
"""


def safe_convert(value, target_type, field_name=None):
    """Safely convert value to target type with error handling."""
    if not value or value == "":
        return None
    try:
        if target_type == list:
            # For list types (like engine options)
            if isinstance(value, list):
                return value
            return [line.strip() for line in str(value).split("\n") if line.strip()]
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for {field_name}: {value}") from e


def find_aircraft_file(icao):
    """Find aircraft YAML file in any engine type subfolder."""
    for engine_type in ENGINE_TYPES:
        yaml_file = YAML_BASE_DIR / engine_type / f"{icao.lower()}.yml"
        if yaml_file.exists():
            return yaml_file, engine_type
    return None, None


def load_aircraft_data(icao):
    """Load aircraft data from YAML file."""
    yaml_file, engine_type = find_aircraft_file(icao)
    if not yaml_file:
        return None, None

    with open(yaml_file, "r") as f:
        document = yaml.safe_load(f)
    return document, engine_type


def format_value(value, field_name=None):
    """Format a value for YAML output with proper precision.

    Ensures integers stay as integers and floats have appropriate decimal places.
    """
    if value is None or value == "":
        return None

    # If it's already an int, return as is
    if isinstance(value, int):
        return value

    # If it's a float
    if isinstance(value, float):
        # Check if it's actually a whole number
        if value == int(value):
            return int(value)

        # For t/c ratio, always use 3 decimal places
        if field_name == "t/c":
            return f"{value:.3f}"
        # Preserve certification-grade mean-aerodynamic-chord precision.
        if field_name == "mac":
            return round(value, 4)
        # For drag coefficients, use 5 decimal precision
        if field_name in ["cd0", "k", "e", "gears"]:
            return round(value, 5)
        # For Mach numbers
        elif field_name and "mach" in field_name.lower():
            return round(value, 2)
        # For flaps ratios
        elif field_name in ["bf/b", "lambda_f", "cf/c", "Sf/S"]:
            return round(value, 2)
        # For dimensions (length, height, width, span, mac)
        else:
            return round(value, 2)

    return value


def save_aircraft_data(icao, data, engine_type=None):
    """Update editable parameters without reordering or rounding the YAML."""

    if engine_type:
        yaml_file = YAML_BASE_DIR / engine_type / f"{icao.lower()}.yml"
    else:
        yaml_file, _ = find_aircraft_file(icao)
        if not yaml_file:
            yaml_file = YAML_BASE_DIR / DEFAULT_ENGINE_TYPE / f"{icao.lower()}.yml"

    editable_fields = set(FIELD_SCHEMA["simple"]) | set(FIELD_SCHEMA["nested"])
    if yaml_file.exists():
        with open(yaml_file, "r", encoding="utf-8") as stream:
            existing_data = yaml.safe_load(stream) or {}
    else:
        existing_data = {}

    merged = dict(existing_data)
    for field_name in editable_fields:
        if field_name in data:
            merged[field_name] = data[field_name]

    sidecar_path = (
        YAML_BASE_DIR.parent
        / "provenance"
        / yaml_file.parent.name
        / f"{yaml_file.stem}.txt"
    )
    rendered_sidecar = None
    if all(
        section in merged for section in ("wing", "horizontal_tail", "vertical_tail")
    ):
        try:
            from update_aircraft_geometry import (
                GEOMETRY_UNITS,
                derive_geometry,
                render_geometry_section,
            )
            from migrate_aircraft_data import parse_old_provenance, validate_simple
        except ModuleNotFoundError:  # package-style import used by tests
            from tools.update_aircraft_geometry import (
                GEOMETRY_UNITS,
                derive_geometry,
                render_geometry_section,
            )
            from tools.migrate_aircraft_data import parse_old_provenance, validate_simple

        old_sidecar = (
            sidecar_path.read_text(encoding="utf-8") if sidecar_path.exists() else ""
        )
        metadata = {
            key: dict(value)
            for key, value in parse_old_provenance(old_sidecar).get("fields", {}).items()
            if key in GEOMETRY_UNITS
        }
        for path, unit in GEOMETRY_UNITS.items():
            section, field_name = path.split(".", 1)
            previous = (existing_data.get(section) or {}).get(field_name)
            replacement = (merged.get(section) or {}).get(field_name)
            if replacement != previous:
                record = {
                    "status": "legacy-unverified",
                    "source": "aerox_web_editor_manual_entry",
                    "notes": "Value entered manually through the Aerox web editor; source qualification is required.",
                }
                if previous is not None:
                    record["previous_value"] = f"{previous} {unit}"
                    record["correction_reason"] = "Manual web-editor change."
                metadata[path] = record
        derive_geometry(merged, metadata)
        issues = validate_simple(merged)
        if issues:
            raise ValueError("Invalid aircraft data: " + "; ".join(issues))
        if old_sidecar:
            rendered_sidecar = render_geometry_section(merged, old_sidecar, metadata)

    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            merged,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=1000,
        )
    if rendered_sidecar is not None:
        sidecar_path.write_text(rendered_sidecar, encoding="utf-8")


def get_aircraft_list(engine_type=None):
    """Get list of aircraft ICAO codes, optionally filtered by engine type."""
    if engine_type:
        folder = YAML_BASE_DIR / engine_type
        if folder.exists():
            return sorted([f.stem.upper() for f in folder.glob("*.yml")])
        return []
    else:
        # Get all aircraft from all subfolders
        all_aircraft = []
        for et in ENGINE_TYPES:
            folder = YAML_BASE_DIR / et
            if folder.exists():
                all_aircraft.extend([f.stem.upper() for f in folder.glob("*.yml")])
        return sorted(all_aircraft)


def get_aircraft_by_engine_type():
    """Get dictionary of aircraft lists grouped by engine type."""
    result = {}
    for engine_type in ENGINE_TYPES:
        result[engine_type] = get_aircraft_list(engine_type)
    return result


@lru_cache(maxsize=1)
def get_unique_field_values():
    """Get unique values for dropdown fields from all YAML files (cached)."""
    unique_values = {
        "icao_wtc": set(),
        "eu_wtc": set(),
        "faa_wtc": set(),
        "flaps_type": set(),
        "engine_type": set(),
    }

    for engine_type in ENGINE_TYPES:
        folder = YAML_BASE_DIR / engine_type
        if not folder.exists():
            continue
        for yaml_file in folder.glob("*.yml"):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)

                if data:
                    # Top-level fields
                    if data.get("icao_wtc"):
                        unique_values["icao_wtc"].add(data["icao_wtc"])
                    if data.get("eu_wtc"):
                        unique_values["eu_wtc"].add(data["eu_wtc"])
                    if data.get("faa_wtc"):
                        unique_values["faa_wtc"].add(data["faa_wtc"])

                    # Nested fields
                    if data.get("flaps") and data["flaps"].get("type"):
                        unique_values["flaps_type"].add(data["flaps"]["type"])
                    if data.get("engine") and data["engine"].get("type"):
                        unique_values["engine_type"].add(data["engine"]["type"])
            except Exception:
                continue

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in unique_values.items()}


def parse_form_data(form):
    """Convert form data back to nested YAML structure using schema-driven approach."""
    data = {}

    # Parse simple top-level fields using schema
    for field, config in FIELD_SCHEMA["simple"].items():
        value = form.get(field, "").strip()
        data[field] = safe_convert(value, config["type"], field) if value else None

    # Parse nested sections using schema
    for section, config in FIELD_SCHEMA["nested"].items():
        section_data = {}
        has_data = False

        for field in config["fields"]:
            key = f"{section}__{field}"
            value = form.get(key, "").strip()

            if value:
                has_data = True
                target_type = config["types"].get(field, str)

                section_data[field] = safe_convert(
                    value, target_type, f"{section}.{field}"
                )
            else:
                section_data[field] = None

        data[section] = section_data if has_data else None

    return data


OVERVIEW_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Aircraft Overview</title>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;600&display=swap">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Roboto Condensed', 'Arial Narrow', sans-serif;
            background: #f5f5f5;
            padding: 10px;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 5px;
            font-size: 22px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 15px;
            font-size: 12px;
        }
        .nav {
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
            align-items: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .nav-group {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .nav-group label {
            font-size: 12px;
            color: #666;
            font-weight: 500;
        }
        select {
            padding: 5px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        .th-select {
            padding: 0;
            border: none;
            background: transparent;
            font-size: 11px;
            font-weight: 600;
            color: #495057;
            cursor: pointer;
            text-decoration: underline;
            text-decoration-style: dotted;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
        }
        .th-select option {
            padding: 8px 12px;
            font-size: 12px;
            background: white;
        }
        .count-badge {
            background: #6c757d;
            color: white;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 8px;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 1px 0;
            font-size: 12px;
        }
        th, td {
            padding: 3px 6px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
            position: sticky;
            top: 0;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .icao-link {
            color: #007bff;
            text-decoration: none;
            font-weight: 600;
        }
        .icao-link:hover {
            text-decoration: underline;
        }
        .tag {
            font-size: 10px;
            padding: 1px 6px;
            border-radius: 10px;
            font-weight: 500;
            display: inline-block;
            margin-right: 2px;
        }
        .tag-civil { background: #d4edda; color: #155724; }
        .tag-military { background: #f8d7da; color: #721c24; }
        .tag-business { background: #cce5ff; color: #004085; }
        .tag-yes { background: #d4edda; color: #155724; }
        .tag-no { background: #f8d7da; color: #721c24; }
        .num { text-align: right; }
        .missing { color: #dc3545; }
        .engine-name {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        /* Section background colors - alternating */
        .sec-a { background: #ffffff; }
        .sec-b { background: #f5f5f5; }
        tr:hover td { filter: brightness(0.95); }
    </style>
</head>
<body>
    <div class="container">
        <a href="/editor" style="float: right; padding: 6px 12px; background: #6c757d; color: white; text-decoration: none; border-radius: 4px; font-size: 12px; font-weight: 500;"><i class="fa-solid fa-edit"></i> Editor</a>
        <a href="/clear-cache" style="float: right; margin-right: 8px; padding: 6px 12px; background: #17a2b8; color: white; text-decoration: none; border-radius: 4px; font-size: 12px; font-weight: 500;"><i class="fa-solid fa-refresh"></i> Refresh</a>
        <h1><i class="fa-solid fa-list"></i> Aircraft Overview</h1>
        <p class="subtitle">Overview of all aircraft data by engine category</p>

        <div class="nav">
            <div class="nav-group">
                <label>Engine:</label>
                <select id="engine-select" onchange="changeEngineType(this.value)">
                    {% for et in engine_types %}
                    <option value="{{ et }}" {% if et == current_engine_type %}selected{% endif %}>{{ et }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="nav-group">
                <label>Search:</label>
                <input type="text" id="search-input" placeholder="Filter..." oninput="filterTable()" style="width: 150px;">
            </div>
            <span class="count-badge"><span id="visible-count">{{ aircraft_list | length }}</span> / {{ aircraft_list | length }} aircraft</span>
        </div>

        <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th class="sec-a">ICAO</th>
                    <th class="sec-a" style="min-width: 120px;">Aircraft Name</th>
                    <th class="sec-a"><select id="type-filter" onchange="filterTable()" class="th-select"><option value="">Type</option><option value="civil">Civil</option><option value="military">Mil</option><option value="business">Biz</option></select></th>
                    <th class="sec-a"><select id="edb-filter" onchange="filterTable()" class="th-select"><option value="">EDB</option><option value="yes">Yes</option><option value="no">No</option></select></th>
                    <th class="sec-a"><select id="flap-filter" onchange="filterTable()" class="th-select"><option value="">Flap</option><option value="yes">Yes</option><option value="no">No</option></select></th>
                    <th class="num sec-b">MTOW</th>
                    <th class="num sec-b">MLW</th>
                    <th class="num sec-b">OEW</th>
                    <th class="num sec-b">MFC</th>
                    <th class="num sec-a">VMO</th>
                    <th class="num sec-a">MMO</th>
                    <th class="num sec-a">Ceiling</th>
                    <th class="sec-b"><select id="icao-wtc-filter" onchange="filterTable()" class="th-select"><option value="">ICAO WTC</option><option value="light">light</option><option value="medium">medium</option><option value="heavy">heavy</option><option value="super">super</option></select></th>
                    <th class="sec-b"><select id="eu-wtc-filter" onchange="filterTable()" class="th-select"><option value="">EU WTC</option><option value="light">light</option><option value="lower medium">lower med</option><option value="upper medium">upper med</option><option value="heavy">heavy</option><option value="super">super</option></select></th>
                    <th class="sec-b"><select id="faa-wtc-filter" onchange="filterTable()" class="th-select"><option value="">FAA WTC</option><option value="small">small</option><option value="large">large</option><option value="medium">medium</option><option value="heavy">heavy</option><option value="super">super</option></select></th>
                    <th class="num sec-b">Parking</th>
                    <th class="num sec-a">Pax Max</th>
                    <th class="num sec-a">Pax High</th>
                    <th class="num sec-a">Pax Low</th>
                    <th class="num sec-b" style="white-space:nowrap">Fus L</th>
                    <th class="num sec-b" style="white-space:nowrap">Fus H</th>
                    <th class="num sec-b" style="white-space:nowrap">Fus W</th>
                    <th class="num sec-a" style="white-space:nowrap">Wing S</th>
                    <th class="num sec-a">Span</th>
                    <th class="num sec-a">MAC</th>
                    <th class="num sec-a">Sweep</th>
                    <th class="num sec-a">t/c</th>
                    <th class="num sec-b">CD0</th>
                    <th class="num sec-b">K</th>
                    <th class="num sec-b">e</th>
                    <th class="num sec-b">Gears</th>
                    <th class="sec-a"><select id="eng-type-filter" onchange="filterTable()" class="th-select"><option value="">Eng Type</option><option value="turbofan">turbofan</option><option value="turboprop">turboprop</option><option value="piston">piston</option></select></th>
                    <th class="sec-a">Eng Mount</th>
                    <th class="num sec-a">N</th>
                    <th class="sec-a">Engine</th>
                </tr>
            </thead>
            <tbody>
                {% for ac in aircraft_list %}
                <tr data-civil="{{ 'yes' if ac.civil else 'no' }}" data-military="{{ 'yes' if ac.military else 'no' }}" data-business="{{ 'yes' if ac.business else 'no' }}" data-edb="{{ 'yes' if ac.edb else 'no' }}" data-flap="{{ 'yes' if ac.flaps_type else 'no' }}" data-icao-wtc="{{ ac.icao_wtc or '' }}" data-eu-wtc="{{ ac.eu_wtc or '' }}" data-faa-wtc="{{ ac.faa_wtc or '' }}" data-eng-type="{{ ac.engine_type or '' }}">
                    <td class="sec-a"><a href="/{{ ac.icao }}" class="icao-link">{{ ac.icao }}</a></td>
                    <td class="sec-a">{{ ac.name or '-' }}</td>
                    <td class="sec-a">
                        {% if ac.civil %}<span class="tag tag-civil">C</span>{% endif %}
                        {% if ac.military %}<span class="tag tag-military">M</span>{% endif %}
                        {% if ac.business %}<span class="tag tag-business">B</span>{% endif %}
                    </td>
                    <td class="sec-a">
                        {% if ac.edb %}<span class="tag tag-yes"><i class="fa-solid fa-check"></i></span>
                        {% else %}<span class="tag tag-no"><i class="fa-solid fa-xmark"></i></span>{% endif %}
                    </td>
                    <td class="sec-a">{% if ac.flaps_type %}<span class="tag tag-yes"><i class="fa-solid fa-check"></i></span>{% else %}<span class="tag tag-no"><i class="fa-solid fa-xmark"></i></span>{% endif %}</td>
                    <td class="num sec-b">{{ ac.mtow or '-' }}</td>
                    <td class="num sec-b">{{ ac.mlw or '-' }}</td>
                    <td class="num sec-b">{{ ac.oew or '-' }}</td>
                    <td class="num sec-b">{{ ac.mfc or '-' }}</td>
                    <td class="num sec-a">{{ ac.vmo or '-' }}</td>
                    <td class="num sec-a">{{ ac.mmo or '-' }}</td>
                    <td class="num sec-a">{{ ac.ceiling or '-' }}</td>
                    <td class="sec-b">{{ ac.icao_wtc or '-' }}</td>
                    <td class="sec-b">{{ ac.eu_wtc or '-' }}</td>
                    <td class="sec-b">{{ ac.faa_wtc or '-' }}</td>
                    <td class="num sec-b">{{ ac.parking_area_m2 or '-' }}</td>
                    <td class="num sec-a">{{ ac.pax_max or '-' }}</td>
                    <td class="num sec-a">{{ ac.pax_high or '-' }}</td>
                    <td class="num sec-a">{{ ac.pax_low or '-' }}</td>
                    <td class="num sec-b">{{ ac.fus_length or '-' }}</td>
                    <td class="num sec-b">{{ ac.fus_height or '-' }}</td>
                    <td class="num sec-b">{{ ac.fus_width or '-' }}</td>
                    <td class="num sec-a">{{ ac.wing_area or '-' }}</td>
                    <td class="num sec-a">{{ ac.wing_span or '-' }}</td>
                    <td class="num sec-a">{{ ac.wing_mac or '-' }}</td>
                    <td class="num sec-a">{{ ac.wing_sweep or '-' }}</td>
                    <td class="num sec-a">{{ ac.wing_tc or '-' }}</td>
                    <td class="num sec-b">{{ ac.drag_cd0 or '-' }}</td>
                    <td class="num sec-b">{{ ac.drag_k or '-' }}</td>
                    <td class="num sec-b">{{ ac.drag_e or '-' }}</td>
                    <td class="num sec-b">{{ ac.drag_gears or '-' }}</td>
                    <td class="sec-a">{{ ac.engine_type or '-' }}</td>
                    <td class="sec-a">{{ ac.engine_mount or '-' }}</td>
                    <td class="num sec-a">{{ ac.engine_number or '-' }}</td>
                    <td class="sec-a engine-name" title="{{ ac.engine_default or '' }}">{{ ac.engine_default or '-' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        </div>
    </div>

    <script>
        function changeEngineType(engineType) {
            window.location.href = '/?engine_type=' + engineType;
        }

        function filterTable() {
            const searchText = document.getElementById('search-input').value.toLowerCase();
            const typeFilter = document.getElementById('type-filter').value;
            const edbFilter = document.getElementById('edb-filter').value;
            const flapFilter = document.getElementById('flap-filter').value;
            const icaoWtcFilter = document.getElementById('icao-wtc-filter').value;
            const euWtcFilter = document.getElementById('eu-wtc-filter').value;
            const faaWtcFilter = document.getElementById('faa-wtc-filter').value;
            const engTypeFilter = document.getElementById('eng-type-filter').value;
            const rows = document.querySelectorAll('tbody tr');
            let visibleCount = 0;

            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                const matchesSearch = text.includes(searchText);

                let matchesType = true;
                if (typeFilter === 'civil') matchesType = row.dataset.civil === 'yes';
                else if (typeFilter === 'military') matchesType = row.dataset.military === 'yes';
                else if (typeFilter === 'business') matchesType = row.dataset.business === 'yes';

                const matchesEdb = !edbFilter || row.dataset.edb === edbFilter;
                const matchesFlap = !flapFilter || row.dataset.flap === flapFilter;
                const matchesIcaoWtc = !icaoWtcFilter || row.dataset.icaoWtc === icaoWtcFilter;
                const matchesEuWtc = !euWtcFilter || row.dataset.euWtc === euWtcFilter;
                const matchesFaaWtc = !faaWtcFilter || row.dataset.faaWtc === faaWtcFilter;
                const matchesEngType = !engTypeFilter || row.dataset.engType === engTypeFilter;

                if (matchesSearch && matchesType && matchesEdb && matchesFlap && matchesIcaoWtc && matchesEuWtc && matchesFaaWtc && matchesEngType) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            document.getElementById('visible-count').textContent = visibleCount;
        }
    </script>
</body>
</html>
"""


@lru_cache(maxsize=1)
def load_all_aircraft_data():
    """Load summary data for all aircraft grouped by engine type (cached)."""
    result = {}

    for engine_type in ENGINE_TYPES:
        folder = YAML_BASE_DIR / engine_type
        if not folder.exists():
            result[engine_type] = []
            continue

        aircraft_list = []
        csv_data = {}

        # Load classification data from CSV if it's turbofan
        csv_file = Path(__file__).parent / f"{engine_type}.csv"
        if csv_file.exists():
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_data[row["icao"]] = row

        for yaml_file in sorted(folder.glob("*.yml")):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)

                if not data:
                    continue

                icao = yaml_file.stem.upper()
                classification = csv_data.get(icao.lower(), {})

                pax = data.get("pax") or {}
                fuselage = data.get("fuselage") or {}
                wing = data.get("wing") or {}
                flaps = data.get("flaps") or {}
                engine = data.get("engine") or {}
                drag = data.get("drag") or {}

                aircraft_list.append({
                    # Classification
                    "icao": icao,
                    "name": data.get("aircraft", ""),
                    "iata": data.get("iata"),
                    "civil": classification.get("civil") == "1",
                    "military": classification.get("military") == "1",
                    "business": classification.get("business") == "1",
                    "edb": classification.get("edb") == "1",
                    # Weight
                    "mtow": data.get("mtow"),
                    "mlw": data.get("mlw"),
                    "oew": data.get("oew"),
                    "mfc": data.get("mfc"),
                    # Performance
                    "vmo": data.get("vmo"),
                    "mmo": data.get("mmo"),
                    "ceiling": data.get("ceiling"),
                    # WTC
                    "icao_wtc": data.get("icao_wtc"),
                    "eu_wtc": data.get("eu_wtc"),
                    "faa_wtc": data.get("faa_wtc"),
                    "parking_area_m2": data.get("parking_area_m2"),
                    # Pax
                    "pax_max": pax.get("max"),
                    "pax_high": pax.get("high"),
                    "pax_low": pax.get("low"),
                    # Fuselage
                    "fus_length": fuselage.get("length"),
                    "fus_height": fuselage.get("height"),
                    "fus_width": fuselage.get("width"),
                    # Wing
                    "wing_area": wing.get("area"),
                    "wing_span": wing.get("span"),
                    "wing_mac": wing.get("mac"),
                    "wing_sweep": wing.get("sweep"),
                    "wing_tc": wing.get("t/c"),
                    # Flaps
                    "flaps_type": flaps.get("type"),
                    # Engine
                    "engine_type": engine.get("type"),
                    "engine_mount": engine.get("mount"),
                    "engine_number": engine.get("number"),
                    "engine_default": engine.get("default"),
                    # Drag
                    "drag_cd0": drag.get("cd0"),
                    "drag_k": drag.get("k"),
                    "drag_e": drag.get("e"),
                    "drag_gears": drag.get("gears"),
                })
            except Exception:
                continue

        result[engine_type] = aircraft_list

    return result


@app.route("/")
def overview():
    """Overview page showing all aircraft by engine category."""
    engine_type = request.args.get("engine_type", DEFAULT_ENGINE_TYPE)
    aircraft_data = load_all_aircraft_data()
    aircraft_list = aircraft_data.get(engine_type, [])
    return render_template_string(
        OVERVIEW_TEMPLATE,
        aircraft_list=aircraft_list,
        engine_types=ENGINE_TYPES,
        current_engine_type=engine_type,
    )


@app.route("/editor")
def index():
    """Editor page - show aircraft list."""
    engine_type = request.args.get("engine_type", DEFAULT_ENGINE_TYPE)
    aircraft_by_type = get_aircraft_by_engine_type()
    field_options = get_unique_field_values()
    return render_template_string(
        HTML_TEMPLATE,
        aircraft_by_type=aircraft_by_type,
        engine_types=ENGINE_TYPES,
        current_engine_type=engine_type,
        field_options=field_options,
        data=None,
        current_icao=None,
        message=None,
        classification=None,
    )


@app.route("/<icao>")
def edit(icao):
    """Edit page for specific aircraft."""
    aircraft_by_type = get_aircraft_by_engine_type()
    field_options = get_unique_field_values()
    data, engine_type = load_aircraft_data(icao)

    if not data:
        return render_template_string(
            HTML_TEMPLATE,
            aircraft_by_type=aircraft_by_type,
            engine_types=ENGINE_TYPES,
            current_engine_type=DEFAULT_ENGINE_TYPE,
            field_options=field_options,
            data=None,
            current_icao=None,
            message=f"Aircraft {icao} not found",
            message_type="error",
            classification=None,
        )

    # Check for success/error messages from URL parameters
    message = None
    message_type = None

    if request.args.get("saved") == "true":
        message = "Changes saved successfully!"
        message_type = "success"
    elif request.args.get("error"):
        message = f'Error saving: {request.args.get("error")}'
        message_type = "error"

    # Load classification from turbofan.csv
    classification = load_aircraft_classification(icao)

    return render_template_string(
        HTML_TEMPLATE,
        aircraft_by_type=aircraft_by_type,
        engine_types=ENGINE_TYPES,
        current_engine_type=engine_type,
        field_options=field_options,
        data=data,
        current_icao=icao,
        message=message,
        message_type=message_type,
        classification=classification,
    )


@app.route("/<icao>/save", methods=["POST"])
def save(icao):
    """Save aircraft data."""
    try:
        data = parse_form_data(request.form)
        engine_type = request.form.get("current_engine_type")
        save_aircraft_data(icao, data, engine_type)

        # Invalidate caches after saving
        get_unique_field_values.cache_clear()
        load_all_aircraft_data.cache_clear()

        # Use POST-Redirect-GET pattern to avoid resubmit warning
        return redirect(url_for("edit", icao=icao, saved="true"))
    except Exception as e:
        # On error, redirect with error parameter
        return redirect(url_for("edit", icao=icao, error=str(e)))


@app.route("/<icao>/calculate-drag", methods=["POST"])
def calculate_drag(icao):
    """Calculate drag coefficients for an aircraft."""
    try:
        # Load aircraft data
        data, _ = load_aircraft_data(icao)

        if not data:
            return jsonify({"error": f"Aircraft {icao} not found"}), 404

        # Check if required data exists
        if not data.get("wing") or not data.get("fuselage"):
            return jsonify({"error": "Missing wing or fuselage data"}), 400

        if not data.get("mtow"):
            return jsonify({"error": "Missing mtow"}), 400

        wing = data.get("wing")
        fuselage = data.get("fuselage")
        mtow = data.get("mtow")

        # Check for required fields
        required_wing = ["area", "span", "mac"]
        required_fuselage = ["length", "height", "width"]

        for field in required_wing:
            if not wing.get(field):
                return jsonify({"error": f"Missing wing.{field}"}), 400

        for field in required_fuselage:
            if not fuselage.get(field):
                return jsonify({"error": f"Missing fuselage.{field}"}), 400

        # Calculate drag coefficients using dragcoef module
        # Create a mock aircraft object with the required attributes
        class MockAircraft:
            def __init__(self, wing_data, fuselage_data, mtow_val):
                self.icao = icao
                self.mtow = mtow_val
                self.wing = type(
                    "obj",
                    (object,),
                    {
                        "area": wing_data["area"],
                        "span": wing_data["span"],
                        "mac": wing_data["mac"],
                    },
                )()
                self.fuselage = type(
                    "obj",
                    (object,),
                    {
                        "length": fuselage_data["length"],
                        "height": fuselage_data["height"],
                        "width": fuselage_data["width"],
                    },
                )()

        mock_aircraft = MockAircraft(wing, fuselage, mtow)
        cd0, k, e, gears = dragcoef.calc_drag_coefficients(mock_aircraft)

        return jsonify({"cd0": float(cd0), "k": float(k), "e": float(e), "gears": float(gears)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear-cache")
def clear_cache():
    """Clear all caches and redirect to overview."""
    get_unique_field_values.cache_clear()
    load_all_aircraft_data.cache_clear()
    return redirect(url_for("overview"))


if __name__ == "__main__":
    print("=" * 60)
    print("Aircraft Data Editor")
    print("=" * 60)
    print("\nPre-warming cache...")
    get_unique_field_values()
    load_all_aircraft_data()
    print("Cache ready!")
    print("\nStarting server at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
