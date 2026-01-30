"""
ETo Edge Computing + Smart Pump System v3.0
===========================================
Phase 2 Features:
- Dashboard with all sensor data + sensor status
- VPD checking with growth stage-specific ranges
- Growth stage day tracking (1-365) with auto-update
- Canopy radius input for wetted area calculation
- InfluxDB recovery for missed data

Version: 3.0.0
Date: January 2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
import numpy as np
import uvicorn
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, List
from collections import deque
from contextlib import asynccontextmanager
import requests
import json
import os
import math
import asyncio
import threading

VALVE_IP = os.getenv("VALVE_IP", "10.157.111.93")

# Try to import dill, fallback to pickle
try:
    import dill as pickle
    print("Using dill for enhanced pickling capabilities")
except ImportError:
    import pickle
    print("Using standard pickle (dill not available)")

# Try to import InfluxDB client for recovery
try:
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.query_api import QueryApi
    INFLUXDB_AVAILABLE = True
    print("InfluxDB client available for data recovery")
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("InfluxDB client not available - install with: pip install influxdb-client")

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================
# InfluxDB Configuration for Recovery
# =========================
INFLUXDB_CONFIG = {
    "url": os.getenv("INFLUXDB_URL", "https://us-east-1-1.aws.cloud2.influxdata.com"),
    "token": os.getenv("INFLUXDB_TOKEN", "UUu1b4ouhxQqu7dwZ-Itjz2jH9umHhGe2bxdKLDRUF2SYJzytzzt5WTBz-kh3zTh8xj3N8xhEQpxMVqn0cLTjw=="),
    "org": os.getenv("INFLUXDB_ORG", "d3e89edfa3c8e4c3"),
    "bucket": os.getenv("INFLUXDB_BUCKET", "Weather_V2")
}

# =========================
# Global Variables
# =========================
prediction_history = deque(maxlen=50)
cycle_data = {}
irrigation_config = {}
irrigation_history = deque(maxlen=20)

# Local storage for 15-minute records (for daily aggregation)
# Stores up to 100 records (25+ hours of data)
local_15min_records = deque(maxlen=100)

# Track last processed timestamp for recovery
last_processed_timestamp = None
recovery_status = {
    "last_recovery_attempt": None,
    "last_successful_recovery": None,
    "records_recovered": 0,
    "recovery_errors": []
}

# Sensor status tracking
sensor_status = {
    "temperature": {"online": False, "last_update": None, "value": None},
    "humidity": {"online": False, "last_update": None, "value": None},
    "pressure": {"online": False, "last_update": None, "value": None},
    "wind": {"online": False, "last_update": None, "value": None},
    "solar": {"online": False, "last_update": None, "value": None},
    "rain": {"online": False, "last_update": None, "value": None},
    "vpd": {"online": False, "last_update": None, "value": None}
}

# Growth stage tracking
growth_stage_config = {
    "start_date": None,  # Date when growth cycle started
    "current_day": 1,    # Current day (1-365)
    "last_update": None  # Last time day was updated
}

# Manual pump session tracking
manual_pump_session = {
    "active": False,
    "duration_seconds": 0,
    "start_time": None
}

# Unlimited pump session tracking
unlimited_pump_session = {
    "active": False,
    "start_time": None,
    "total_runtime_seconds": 0
}

# Valve Controller Configuration
VALVE_CONTROLLER_IP = "10.157.111.93"
VALVE_CONTROLLER_PORT = 80
valve_status = {"state": "closed", "last_updated": None, "connection_status": "disconnected"}

MODEL_DIR = Path("models")
MODEL_FILES = {
    "rf_model": MODEL_DIR / "rad_rf_model.pkl",
    "mlp_model": MODEL_DIR / "eto_mlp_model.pkl",
    "scaler_rad": MODEL_DIR / "scaler_rad.pkl",
    "scaler_eto": MODEL_DIR / "scaler_eto.pkl",
    "metadata": MODEL_DIR / "model_metadata.json",
}

models = {
    "rf_model": None,
    "mlp_model": None,
    "scaler_rad": None,
    "scaler_eto": None,
    "metadata": None,
    "loaded_at": None,
}

# =========================
# VPD Ranges by Growth Stage
# =========================
VPD_RANGES = {
    "Flowering & fruit setting (0-45 Days)": {"min": 0.4, "max": 0.8, "optimal": 0.6},
    "Early fruit growth (46-75 days)": {"min": 0.5, "max": 1.0, "optimal": 0.7},
    "Fruit growth (76-110 days)": {"min": 0.6, "max": 1.2, "optimal": 0.9},
    "Fruit Maturity (111-140 days)": {"min": 0.5, "max": 1.0, "optimal": 0.8},
    "Vegetative Growth (141-290 days)": {"min": 0.8, "max": 1.5, "optimal": 1.0},
    "Floral initiation (291-320 days)": {"min": 1.0, "max": 1.8, "optimal": 1.4},
    "Floral development (321-365 days)": {"min": 0.8, "max": 1.5, "optimal": 1.2}
}

# Growth stage day ranges
GROWTH_STAGES = [
    {"name": "Flowering & fruit setting (0-45 Days)", "start": 1, "end": 45},
    {"name": "Early fruit growth (46-75 days)", "start": 46, "end": 75},
    {"name": "Fruit growth (76-110 days)", "start": 76, "end": 110},
    {"name": "Fruit Maturity (111-140 days)", "start": 111, "end": 140},
    {"name": "Vegetative Growth (141-290 days)", "start": 141, "end": 290},
    {"name": "Floral initiation (291-320 days)", "start": 291, "end": 320},
    {"name": "Floral development (321-365 days)", "start": 321, "end": 365}
]

def get_growth_stage_from_day(day: int) -> str:
    """Get growth stage name from day number"""
    for stage in GROWTH_STAGES:
        if stage["start"] <= day <= stage["end"]:
            return stage["name"]
    return "Vegetative Growth (141-290 days)"  # Default

def update_growth_day():
    """Update growth day based on start date"""
    global growth_stage_config, irrigation_config

    if growth_stage_config["start_date"]:
        start = growth_stage_config["start_date"]
        today = date.today()
        days_elapsed = (today - start).days + 1

        # Wrap around if > 365
        growth_stage_config["current_day"] = ((days_elapsed - 1) % 365) + 1
        growth_stage_config["last_update"] = datetime.now().isoformat()

        # Auto-update irrigation config growth stage
        new_stage = get_growth_stage_from_day(growth_stage_config["current_day"])
        if irrigation_config.get("crop_growth_stage") != new_stage:
            irrigation_config["crop_growth_stage"] = new_stage
            logger.info(f"Growth stage auto-updated to: {new_stage} (Day {growth_stage_config['current_day']})")

def check_vpd_status(vpd: float, growth_stage: str) -> dict:
    """Check VPD against growth stage-specific ranges"""
    if growth_stage not in VPD_RANGES:
        return {"status": "unknown", "message": "Unknown growth stage"}

    ranges = VPD_RANGES[growth_stage]

    if vpd < ranges["min"]:
        return {
            "status": "low",
            "message": f"VPD too low ({vpd:.2f} kPa). Range: {ranges['min']}-{ranges['max']} kPa",
            "action": "Increase ventilation or reduce humidity",
            "color": "#2196f3"  # Blue
        }
    elif vpd > ranges["max"]:
        return {
            "status": "high",
            "message": f"VPD too high ({vpd:.2f} kPa). Range: {ranges['min']}-{ranges['max']} kPa",
            "action": "Consider irrigation to reduce plant stress",
            "color": "#f44336"  # Red
        }
    else:
        distance_to_optimal = abs(vpd - ranges["optimal"])
        if distance_to_optimal <= 0.2:
            return {
                "status": "optimal",
                "message": f"VPD optimal ({vpd:.2f} kPa)",
                "action": "No action needed",
                "color": "#4caf50"  # Green
            }
        else:
            return {
                "status": "acceptable",
                "message": f"VPD acceptable ({vpd:.2f} kPa). Optimal: {ranges['optimal']} kPa",
                "action": "Monitor conditions",
                "color": "#ff9800"  # Orange
            }

def update_sensor_status(sensor_name: str, value, online: bool = True):
    """Update sensor status tracking with smart validation"""
    if sensor_name in sensor_status:
        # Smart online detection based on sensor type and value
        is_online = online

        if value is None:
            is_online = False
        elif sensor_name == "solar":
            # Solar: 0 W/m² could be valid at night, but we mark offline if exactly 0
            # In production, ESP32 should send sensor_health status
            is_online = value is not None and value > 0
        elif sensor_name == "temperature":
            # Temperature: valid range -40 to 60°C
            is_online = value is not None and -40 <= value <= 60
        elif sensor_name == "humidity":
            # Humidity: valid range 0-100%
            is_online = value is not None and 0 <= value <= 100
        elif sensor_name == "pressure":
            # Pressure: valid range 800-1100 hPa
            is_online = value is not None and 800 <= value <= 1100
        elif sensor_name == "wind":
            # Wind: valid range 0-50 m/s (0 is valid - calm)
            is_online = value is not None and 0 <= value <= 50
        elif sensor_name == "rain":
            # Rain: valid range 0-500 mm (0 is valid - no rain)
            is_online = value is not None and 0 <= value <= 500
        elif sensor_name == "vpd":
            # VPD: valid range 0-5 kPa
            is_online = value is not None and 0 <= value <= 5

        sensor_status[sensor_name] = {
            "online": is_online,
            "last_update": datetime.now().isoformat(),
            "value": value
        }

# =========================
# Valve Controller Communication
# =========================
class ValveController:
    def __init__(self, ip, port=80):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"

    async def get_valve_status(self):
        try:
            response = requests.get(f"{self.base_url}/valve/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                valve_status["state"] = data.get("valve_state", "unknown")
                valve_status["last_updated"] = datetime.now().isoformat()
                valve_status["connection_status"] = "connected"
                return data
            else:
                valve_status["connection_status"] = "error"
                return None
        except Exception as e:
            logger.error(f"Failed to get valve status: {e}")
            valve_status["connection_status"] = "disconnected"
            return None

    async def open_valve(self, duration_seconds):
        try:
            response = requests.post(f"{self.base_url}/valve/open?duration={duration_seconds}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                valve_status["state"] = "open"
                valve_status["last_updated"] = datetime.now().isoformat()
                valve_status["connection_status"] = "connected"
                logger.info(f"Valve opened for {duration_seconds} seconds")
                return data
            else:
                valve_status["connection_status"] = "error"
                return None
        except Exception as e:
            logger.error(f"Failed to open valve: {e}")
            valve_status["connection_status"] = "disconnected"
            return None

    async def close_valve(self):
        try:
            response = requests.get(f"{self.base_url}/valve/close", timeout=5)
            if response.status_code == 200:
                data = response.json()
                valve_status["state"] = "closed"
                valve_status["last_updated"] = datetime.now().isoformat()
                valve_status["connection_status"] = "connected"
                logger.info("Valve closed manually")
                return data
            else:
                valve_status["connection_status"] = "error"
                return None
        except Exception as e:
            logger.error(f"Failed to close valve: {e}")
            valve_status["connection_status"] = "disconnected"
            return None

valve_controller = ValveController(VALVE_CONTROLLER_IP, VALVE_CONTROLLER_PORT)

# =========================
# InfluxDB Recovery Functions
# =========================
class InfluxDBRecovery:
    def __init__(self):
        self.client = None
        self.query_api = None

    def connect(self):
        if not INFLUXDB_AVAILABLE:
            logger.warning("InfluxDB client not installed")
            return False
        try:
            self.client = InfluxDBClient(
                url=INFLUXDB_CONFIG["url"],
                token=INFLUXDB_CONFIG["token"],
                org=INFLUXDB_CONFIG["org"]
            )
            self.query_api = self.client.query_api()
            logger.info("Connected to InfluxDB for recovery")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False

    def query_missed_data(self, since_timestamp: str = None, hours_back: int = 24):
        if not self.query_api:
            if not self.connect():
                return []
        try:
            if since_timestamp:
                time_filter = f'|> range(start: {since_timestamp})'
            else:
                time_filter = f'|> range(start: -{hours_back}h)'

            query = f'''
            from(bucket: "{INFLUXDB_CONFIG['bucket']}")
                {time_filter}
                |> filter(fn: (r) => r["_measurement"] == "weather_cycle")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''

            logger.info(f"Querying InfluxDB for missed data...")
            tables = self.query_api.query(query, org=INFLUXDB_CONFIG["org"])

            records = []
            for table in tables:
                for record in table.records:
                    records.append({
                        "timestamp": record.get_time().isoformat(),
                        "temp_min": record.values.get("temp_min"),
                        "temp_max": record.values.get("temp_max"),
                        "temp_avg": record.values.get("temp_avg"),
                        "humidity": record.values.get("humidity"),
                        "pressure": record.values.get("pressure"),
                        "wind_speed": record.values.get("wind_speed"),
                        "wind_direction": record.values.get("wind_direction"),
                        "solar_radiation": record.values.get("solar_radiation"),
                        "sunshine_hours": record.values.get("sunshine_hours"),
                        "rainfall_mm": record.values.get("rainfall_mm"),
                        "vpd": record.values.get("vpd"),
                        "device": record.values.get("device", "ESP32")
                    })

            logger.info(f"Retrieved {len(records)} records from InfluxDB")
            return records

        except Exception as e:
            logger.error(f"Failed to query InfluxDB: {e}")

    def query_15min_records_for_date(self, target_date: date):
        """Query all 15-minute aggregate records for a specific date"""
        if not self.query_api:
            if not self.connect():
                return []

        try:
            # Build time range for the target date (midnight to midnight)
            start_time = datetime.combine(target_date, datetime.min.time())
            end_time = start_time + timedelta(days=1)

            query = f'''
            from(bucket: "{INFLUXDB_CONFIG['bucket']}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "weather_15min")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''

            logger.info(f"Querying InfluxDB for 15-min records on {target_date}...")
            tables = self.query_api.query(query, org=INFLUXDB_CONFIG["org"])

            records = []
            for table in tables:
                for record in table.records:
                    records.append({
                        "timestamp": record.get_time().isoformat(),
                        "temp_min_15": record.values.get("temp_min_15"),
                        "temp_max_15": record.values.get("temp_max_15"),
                        "temp_avg_15": record.values.get("temp_avg_15"),
                        "rh_avg_15": record.values.get("rh_avg_15"),
                        "pressure_avg_15": record.values.get("pressure_avg_15"),
                        "wind_avg_15": record.values.get("wind_avg_15"),
                        "wind_max_15": record.values.get("wind_max_15"),
                        "solar_avg_15": record.values.get("solar_avg_15"),
                        "sunshine_min_15": record.values.get("sunshine_min_15"),
                        "rain_mm_15": record.values.get("rain_mm_15"),
                        "vpd_avg_15": record.values.get("vpd_avg_15"),
                        "device": record.values.get("device", "ESP32")
                    })

            logger.info(f"Retrieved {len(records)} 15-min records for {target_date}")
            return records

        except Exception as e:
            logger.error(f"Failed to query 15-min records: {e}")
            recovery_status["recovery_errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            return []

    def close(self):
        if self.client:
            self.client.close()

influx_recovery = InfluxDBRecovery()

# =========================
# Local 15-min Record Storage Functions
# =========================
def round2(value):
    """Round value to 2 decimal places, handling None"""
    if value is None:
        return None
    return round(float(value), 2)

def store_15min_record_locally(data: dict):
    """
    Store a 15-minute aggregate record locally for daily aggregation
    Called when ESP32 sends data to /predict-esp32
    All float values rounded to 2 decimals
    """
    tmin = data.get("tmin")
    tmax = data.get("tmax")
    temp_avg = round2((tmin + tmax) / 2) if tmin and tmax else None

    record = {
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "received_at": datetime.now().isoformat(),
        "temp_min_15": round2(tmin),
        "temp_max_15": round2(tmax),
        "temp_avg_15": temp_avg,
        "rh_avg_15": round2(data.get("humidity")),
        "pressure_avg_15": round2(data.get("pressure")),
        "wind_avg_15": round2(data.get("wind_speed")),
        "wind_max_15": round2(data.get("wind_speed")),  # Same as avg for now
        "solar_avg_15": round2(data.get("solar_radiation")),
        "sunshine_min_15": data.get("sunshine_minutes", 0),  # Integer minutes
        "rain_mm_15": round2(data.get("rainfall_mm", 0)),
        "vpd_avg_15": round2(data.get("vpd")),
        "device": data.get("device_id", "ESP32")
    }

    local_15min_records.append(record)
    logger.debug(f"Stored 15-min record locally (total: {len(local_15min_records)})")

def get_local_15min_records_for_date(target_date: date) -> list:
    """
    Get locally stored 15-minute records for a specific date
    Returns records that fall within the target date (midnight to midnight)
    """
    start_time = datetime.combine(target_date, datetime.min.time())
    end_time = start_time + timedelta(days=1)

    matching_records = []
    for record in local_15min_records:
        try:
            # Parse timestamp
            ts_str = record.get("timestamp") or record.get("received_at")
            if ts_str:
                # Handle different timestamp formats
                if "T" in ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
                else:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

                if start_time <= ts < end_time:
                    matching_records.append(record)
        except Exception as e:
            logger.debug(f"Error parsing timestamp: {e}")
            continue

    logger.info(f"Found {len(matching_records)} local 15-min records for {target_date}")
    return matching_records

# =========================
# Daily Aggregation Storage
# =========================
daily_prediction_result = {
    "last_prediction_date": None,
    "daily_aggregates": None,
    "eto_prediction": None,
    "irrigation_recommendation": None,
    "prediction_timestamp": None
}

# =========================
# Daily Aggregation Function (6:00 AM)
# =========================
def calculate_daily_aggregates_from_15min(records: list) -> dict:
    """
    Calculate daily aggregates from 15-minute records
    Input: List of 15-min records from InfluxDB
    Output: Daily feature vector for ML prediction
    """
    if not records:
        return None

    # Extract values (filter out None)
    temp_mins = [r["temp_min_15"] for r in records if r.get("temp_min_15") is not None]
    temp_maxs = [r["temp_max_15"] for r in records if r.get("temp_max_15") is not None]
    temp_avgs = [r["temp_avg_15"] for r in records if r.get("temp_avg_15") is not None]
    rh_avgs = [r["rh_avg_15"] for r in records if r.get("rh_avg_15") is not None]
    wind_avgs = [r["wind_avg_15"] for r in records if r.get("wind_avg_15") is not None]
    sunshine_mins = [r["sunshine_min_15"] for r in records if r.get("sunshine_min_15") is not None]
    rain_mms = [r["rain_mm_15"] for r in records if r.get("rain_mm_15") is not None]
    pressure_avgs = [r["pressure_avg_15"] for r in records if r.get("pressure_avg_15") is not None]

    if not temp_mins or not temp_maxs:
        logger.warning("Insufficient data for daily aggregation")
        return None

    # Calculate daily aggregates
    daily = {
        "Tmin_day": min(temp_mins),
        "Tmax_day": max(temp_maxs),
        "Tmean_day": sum(temp_avgs) / len(temp_avgs) if temp_avgs else (min(temp_mins) + max(temp_maxs)) / 2,
        "RHmean_day": sum(rh_avgs) / len(rh_avgs) if rh_avgs else 70.0,
        "WindMean_day": sum(wind_avgs) / len(wind_avgs) if wind_avgs else 1.0,
        "SunshineHours_day": sum(sunshine_mins) / 60.0 if sunshine_mins else 0.0,  # Convert minutes to hours
        "Rain_day": sum(rain_mms) if rain_mms else 0.0,
        "PressureMean_day": sum(pressure_avgs) / len(pressure_avgs) if pressure_avgs else 1013.0,
        "records_count": len(records),
        "data_completeness": len(records) / 96.0 * 100  # Percentage of expected 96 records
    }

    logger.info(f"Daily aggregates calculated: Tmin={daily['Tmin_day']:.1f}, Tmax={daily['Tmax_day']:.1f}, "
                f"Sun={daily['SunshineHours_day']:.1f}h, Rain={daily['Rain_day']:.1f}mm, "
                f"Records={daily['records_count']}/96 ({daily['data_completeness']:.0f}%)")

    return daily

async def run_daily_prediction(target_date: date = None):
    """
    Run daily ETo prediction using aggregated data
    HYBRID APPROACH: Uses local storage first, InfluxDB as fallback
    Called at 6:00 AM for yesterday's data
    """
    global daily_prediction_result

    if target_date is None:
        target_date = date.today() - timedelta(days=1)  # Yesterday

    logger.info(f"=== RUNNING DAILY PREDICTION FOR {target_date} ===")

    # STEP 1: Try to get records from local storage first
    records = get_local_15min_records_for_date(target_date)
    data_source = "local"

    # Check if we have enough local data (at least 50% = 48 records)
    if len(records) < 48:
        logger.info(f"Local data incomplete ({len(records)}/96). Querying InfluxDB as fallback...")
        influx_records = influx_recovery.query_15min_records_for_date(target_date)

        if influx_records and len(influx_records) > len(records):
            records = influx_records
            data_source = "influxdb"
            logger.info(f"Using InfluxDB data: {len(records)} records")
        elif records:
            logger.info(f"InfluxDB returned fewer/no records. Using local data: {len(records)} records")
        else:
            logger.warning(f"No 15-min records found for {target_date} (local or InfluxDB)")
            return None
    else:
        logger.info(f"Using local data: {len(records)} records (sufficient coverage)")

    if not records:
        logger.warning(f"No 15-min records found for {target_date}")
        return None

    # Calculate daily aggregates
    daily = calculate_daily_aggregates_from_15min(records)

    if not daily:
        logger.error("Failed to calculate daily aggregates")
        return None

    # Add data source info
    daily["data_source"] = data_source

    # Store aggregates
    daily_prediction_result["daily_aggregates"] = daily
    daily_prediction_result["last_prediction_date"] = target_date.isoformat()
    daily_prediction_result["data_source"] = data_source

    # Check if models are loaded
    if not all(models[key] is not None for key in ["rf_model", "mlp_model", "scaler_rad", "scaler_eto"]):
        logger.error("Models not loaded - cannot run prediction")
        return daily

    try:
        # Run ML prediction
        # Step 1: RAD estimation
        rad_input = enhance_features_for_rad(
            daily["Tmin_day"], daily["Tmax_day"], daily["RHmean_day"],
            daily["WindMean_day"], daily["SunshineHours_day"]
        )

        rad_scaled = models["scaler_rad"].transform(rad_input)
        rad_prediction_raw = models["rf_model"].predict(rad_scaled)[0]
        estimated_rad = apply_nighttime_constraints(rad_prediction_raw, daily["SunshineHours_day"])

        # Step 2: ETo prediction
        eto_input = np.array([[
            daily["Tmin_day"], daily["Tmax_day"], daily["RHmean_day"],
            daily["WindMean_day"], daily["SunshineHours_day"], estimated_rad,
        ]])

        eto_scaled = models["scaler_eto"].transform(eto_input)
        eto_prediction_raw = models["mlp_model"].predict(eto_scaled)[0]
        eto_prediction = max(0.1, min(eto_prediction_raw, 15.0))

        logger.info(f"DAILY ETo PREDICTION: {eto_prediction:.3f} mm/day (RAD: {estimated_rad:.2f} MJ/m2)")

        # Store prediction result
        daily_prediction_result["eto_prediction"] = {
            "eto_mm_day": round(float(eto_prediction), 3),
            "estimated_rad": round(float(estimated_rad), 3),
            "date": target_date.isoformat()
        }
        daily_prediction_result["prediction_timestamp"] = datetime.now().isoformat()

        # Calculate irrigation recommendation
        if irrigation_config:
            config = IrrigationConfig(**irrigation_config)
            irrigation_result = irrigation_calc.calculate_irrigation_recommendation(
                eto_prediction, config, daily["Rain_day"]
            )
            irrigation_result["date"] = target_date.isoformat()
            irrigation_result["prediction_type"] = "daily"
            daily_prediction_result["irrigation_recommendation"] = irrigation_result
            irrigation_history.append(irrigation_result)

            logger.info(f"DAILY IRRIGATION: {irrigation_result['recommendation']}")

        return daily_prediction_result

    except Exception as e:
        logger.error(f"Daily prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return daily

# =========================
# 6:00 AM Scheduler
# =========================
def daily_scheduler():
    """Background thread that runs daily prediction at 6:00 AM"""
    logger.info("Daily scheduler started - will run at 6:00 AM")

    while True:
        now = datetime.now()
        target_time = now.replace(hour=6, minute=0, second=0, microsecond=0)

        # If it's past 6 AM today, schedule for tomorrow
        if now >= target_time:
            target_time += timedelta(days=1)

        # Calculate sleep duration
        sleep_seconds = (target_time - now).total_seconds()
        logger.info(f"Next daily prediction scheduled for {target_time} (in {sleep_seconds/3600:.1f} hours)")

        # Sleep until 6:00 AM
        import time
        time.sleep(sleep_seconds)

        # Run the daily prediction
        logger.info("=== 6:00 AM - TRIGGERING DAILY PREDICTION ===")
        try:
            # Run in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_daily_prediction())
            loop.close()

            if result:
                logger.info("Daily prediction completed successfully")
            else:
                logger.warning("Daily prediction returned no result")
        except Exception as e:
            logger.error(f"Daily prediction error: {e}")

# =========================
# Model Loading Function
# =========================
def load_models():
    try:
        logger.info("Loading models and scalers...")
        models["rf_model"] = pickle.load(open(str(MODEL_FILES["rf_model"]), "rb"))
        models["mlp_model"] = pickle.load(open(str(MODEL_FILES["mlp_model"]), "rb"))
        models["scaler_rad"] = pickle.load(open(str(MODEL_FILES["scaler_rad"]), "rb"))
        models["scaler_eto"] = pickle.load(open(str(MODEL_FILES["scaler_eto"]), "rb"))
        models["metadata"] = json.load(open(str(MODEL_FILES["metadata"]), "r"))
        models["loaded_at"] = datetime.now().isoformat()
        logger.info("Models loaded successfully")

        global irrigation_config
        irrigation_config = {
            "crop_type": "Durian",
            "crop_growth_stage": "Fruit growth (76-110 days)",
            "soil_type": "Loam",
            "irrigation_type": "Sprinkler",
            "emitter_rate": 40.0,
            "plant_spacing": 10.0,
            "num_sprinklers": 2,
            "farm_name": "ESP32 Farm",
            "canopy_radius": 3.0,
            # C1: Plant Maturity Stage
            "plant_maturity": "Mature",  # Young, Middle, Mature
            # C2: Wetted percentage
            "wetted_pct": 50.0,  # Default for 2 sprinklers
            # C3: Trunk buffer
            "trunk_buffer_enabled": False,
            "trunk_buffer_radius": 0.0
        }
        logger.info("Default irrigation config set")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# =========================
# Startup Recovery Check
# =========================
async def check_and_recover_missed_data():
    global last_processed_timestamp, recovery_status

    logger.info("Checking for missed data to recover...")
    recovery_status["last_recovery_attempt"] = datetime.now().isoformat()

    records = influx_recovery.query_missed_data(hours_back=24)

    if records:
        recovered_count = 0
        for record in records:
            try:
                if all(record.get(k) is not None for k in ["temp_min", "temp_max", "humidity", "wind_speed", "sunshine_hours"]):
                    prediction_data = {
                        "timestamp": record["timestamp"],
                        "device_id": record.get("device", "ESP32_RECOVERED"),
                        "input_data": {
                            "tmin": record["temp_min"],
                            "tmax": record["temp_max"],
                            "temp_avg": record.get("temp_avg", (record["temp_min"] + record["temp_max"]) / 2),
                            "humidity": record["humidity"],
                            "wind_speed": record["wind_speed"],
                            "sunshine_hours": record["sunshine_hours"],
                            "rainfall_mm": record.get("rainfall_mm", 0),
                            "pressure": record.get("pressure"),
                            "vpd": record.get("vpd")
                        },
                        "recovered": True
                    }
                    prediction_history.append(prediction_data)
                    recovered_count += 1
            except Exception as e:
                logger.error(f"Error processing recovered record: {e}")

        recovery_status["records_recovered"] = recovered_count
        recovery_status["last_successful_recovery"] = datetime.now().isoformat()
        logger.info(f"Recovered {recovered_count} records from InfluxDB")

        if records:
            last_processed_timestamp = records[-1]["timestamp"]
    else:
        logger.info("No missed data to recover")

# =========================
# Lifespan Event Handler
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    await valve_controller.get_valve_status()
    await check_and_recover_missed_data()
    update_growth_day()  # Initialize growth day

    # Start 6:00 AM daily scheduler in background thread
    scheduler_thread = threading.Thread(target=daily_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("6:00 AM daily prediction scheduler started")

    yield
    influx_recovery.close()

# =========================
# FastAPI App with Lifespan
# =========================
app = FastAPI(
    title="ETo Edge Computing + Smart Pump System v3",
    description="Phase 2: Dashboard, VPD checking, Growth stage tracking, Canopy radius calculation",
    version="3.0.0",
    lifespan=lifespan
)

# =========================
# Pydantic Models
# =========================
class WeatherInput(BaseModel):
    tmin: float
    tmax: float
    humidity: float
    wind_speed: float
    sunshine_hours: float
    rainfall_mm: float = 0.0
    pressure: Optional[float] = None
    vpd: Optional[float] = None
    device_id: str = "ESP32"
    timestamp: Optional[str] = None

    @validator('tmin', 'tmax')
    def validate_temperature(cls, v):
        if not -50 <= v <= 60:
            raise ValueError(f'Temperature {v}C is outside realistic range (-50 to 60C)')
        return v

    @validator('tmax')
    def validate_temp_range(cls, v, values):
        if 'tmin' in values and v < values['tmin']:
            raise ValueError('Maximum temperature cannot be less than minimum temperature')
        return v

    @validator('humidity')
    def validate_humidity(cls, v):
        if not 0 <= v <= 100:
            raise ValueError(f'Humidity {v}% must be between 0-100%')
        return v

    @validator('wind_speed')
    def validate_wind_speed(cls, v):
        if not 0 <= v <= 45.0:
            raise ValueError(f'Wind speed {v} m/s is outside sensor range (0-45 m/s)')
        return v

    @validator('sunshine_hours')
    def validate_sunshine(cls, v):
        if not 0 <= v <= 24:
            raise ValueError(f'Sunshine hours {v} must be between 0-24 hours')
        return v

    @validator('rainfall_mm')
    def validate_rainfall(cls, v):
        if not 0 <= v <= 500:
            raise ValueError(f'Rainfall {v}mm is outside realistic range (0-500mm)')
        return v

class IrrigationConfig(BaseModel):
    crop_type: str = "Durian"
    crop_growth_stage: str = "Fruit growth (76-110 days)"
    soil_type: str = "Loam"
    irrigation_type: str = "Sprinkler"
    emitter_rate: float = 40.0
    plant_spacing: float = 10.0
    num_sprinklers: int = 2
    farm_name: str = "Default Farm"
    canopy_radius: float = 3.0  # Canopy radius in meters
    # C1: Plant Maturity Stage - affects effective root zone (Zr) and MAD logic
    plant_maturity: str = "Mature"  # Young, Middle, Mature
    # C2: Wetted percentage - based on sprinkler coverage
    wetted_pct: float = 50.0  # Default for 2 sprinklers (1=25%, 2=50%, 3=75%, 4=85%)
    # C3: Trunk buffer area - option to exclude area near trunk
    trunk_buffer_enabled: bool = False
    trunk_buffer_radius: float = 0.0  # 0 to 0.7m

class GrowthStageConfig(BaseModel):
    start_day: int  # Day 1-365 to set as current
    start_date: Optional[str] = None  # Optional: set start date for auto-progression

class EtoPrediction(BaseModel):
    eto_mm_day: float
    estimated_rad: float
    input_data: dict
    timestamp: str
    device_id: str
    processing_time_ms: float
    model_info: dict
    validation_warnings: List[str] = []
    vpd_status: Optional[dict] = None  # NEW: VPD status info

class RecoveryRequest(BaseModel):
    hours_back: int = 24
    since_timestamp: Optional[str] = None

class PumpCommand(BaseModel):
    action: str
    duration: Optional[int] = None

# =========================
# Enhanced Irrigation Calculator with Canopy Radius
# =========================
class IrrigationCalculator:
    def __init__(self):
        self.previous_deficit = 0
        self.yesterday_deficit = 0
        self.daily_irrigation_applied = 0
        self.last_irrigation_time = None
        self.soil_depth = 0.5

        # C1: Effective Root Zone (Zr) by Plant Maturity Stage
        self.ZR_BY_MATURITY = {
            "Young": 0.30,   # 30cm effective root zone
            "Middle": 0.45,  # 45cm effective root zone
            "Mature": 0.60   # 60cm effective root zone
        }

        # C1: MAD Irrigation Ratios by Growth Stage (for Middle/Mature plants)
        self.MAD_IR_BY_STAGE = {
            "Flowering & fruit setting (0-45 Days)": 0.40,
            "Early fruit growth (46-75 days)": 0.45,
            "Fruit growth (76-110 days)": 0.55,
            "Fruit Maturity (111-140 days)": 0.60,
            "Vegetative Growth (141-290 days)": 0.70,
            "Floral initiation (291-320 days)": 0.50,
            "Floral development (321-365 days)": 0.45
        }

        # C2: Default wetted percentage by number of sprinklers
        self.WETTED_PCT_BY_SPRINKLERS = {
            1: 25.0,
            2: 50.0,
            3: 75.0,
            4: 85.0
        }

    def get_swhc(self, soil_type):
        swhc_values = {
            'Sandy Loam': 120, 'Loamy Sand': 80, 'Loam': 160,
            'Sandy Clay Loam': 110, 'Clay Loam': 160
        }
        return swhc_values.get(soil_type, 160)

    def get_effective_root_zone(self, plant_maturity):
        """C1: Get Zr (effective root zone depth in meters) based on plant maturity"""
        return self.ZR_BY_MATURITY.get(plant_maturity, 0.60)  # Default to Mature

    def get_mad_ir(self, crop_growth_stage, plant_maturity):
        """C1: Get MAD Irrigation Ratio based on growth stage and plant maturity
        Young plants: Fixed p=0.5 (RAW = TAW × 0.5)
        Middle/Mature plants: MAD varies by growth stage
        """
        if plant_maturity == "Young":
            return 0.5  # Fixed for young plants
        return self.MAD_IR_BY_STAGE.get(crop_growth_stage, 0.5)

    def get_default_wetted_pct(self, num_sprinklers):
        """C2: Get default wetted percentage based on number of sprinklers"""
        return self.WETTED_PCT_BY_SPRINKLERS.get(num_sprinklers, 50.0)

    def calculate_root_and_wetted_area(self, canopy_radius, trunk_buffer_enabled, trunk_buffer_radius, wetted_pct):
        """C3: Calculate root area and wetted area with trunk buffer consideration
        A_canopy = π × canopy_radius²
        A_buffer = π × trunk_buffer_radius² (capped at 20% of canopy area)
        A_root = A_canopy - A_buffer
        A_wetted = A_root × (wetted_pct / 100)
        """
        canopy_area = math.pi * canopy_radius ** 2

        # Calculate trunk buffer area if enabled
        buffer_area = 0
        if trunk_buffer_enabled and trunk_buffer_radius > 0:
            buffer_area = math.pi * trunk_buffer_radius ** 2
            # Cap buffer at 20% of canopy area
            max_buffer = canopy_area * 0.20
            if buffer_area > max_buffer:
                buffer_area = max_buffer
                logger.info(f"Trunk buffer capped at 20% of canopy: {max_buffer:.2f}m²")

        # Calculate root area (canopy minus buffer)
        root_area = canopy_area - buffer_area

        # Calculate wetted area
        wetted_area = root_area * (wetted_pct / 100)

        logger.info(f"Area calculation: Canopy={canopy_area:.2f}m², Buffer={buffer_area:.2f}m², Root={root_area:.2f}m², Wetted={wetted_area:.2f}m² ({wetted_pct}%)")

        return {
            "canopy_area": round(canopy_area, 2),
            "buffer_area": round(buffer_area, 2),
            "root_area": round(root_area, 2),
            "wetted_area": round(wetted_area, 2)
        }

    def get_kc(self, crop_type, crop_growth_stage):
        kc_values = {
            'Durian': {
                "Flowering & fruit setting (0-45 Days)": 0.5,
                "Early fruit growth (46-75 days)": 0.6,
                "Fruit growth (76-110 days)": 0.85,
                "Fruit Maturity (111-140 days)": 0.75,
                "Vegetative Growth (141-290 days)": 0.6,
                "Floral initiation (291-320 days)": 0.4,
                "Floral development (321-365 days)": 0.4
            },
            'Mangosteen': {
                "Flowering & fruit setting (0-45 Days)": 0.5,
                "Early fruit growth (46-75 days)": 0.6,
                "Fruit growth (76-110 days)": 0.8,
                "Fruit Maturity (111-140 days)": 0.7,
                "Vegetative Growth (141-290 days)": 0.6,
                "Floral initiation (291-320 days)": 0.4,
                "Floral development (321-365 days)": 0.4
            }
        }
        return kc_values.get(crop_type, {}).get(crop_growth_stage, 0.6)

    def get_wetted_diameter(self, emitter_rate):
        wetted_diameter_values = {
            20: 1.5, 30: 1.8, 35: 2.0, 40: 2.2, 50: 2.5, 58: 2.8, 70: 3.0
        }
        return wetted_diameter_values.get(int(emitter_rate), 2.0)

    def calculate_wetted_area(self, plant_spacing, emitter_rate, num_sprinklers, canopy_radius=None, overlap_factor=0.8):
        """Calculate wetted area - with optional canopy radius override"""
        wetted_diameter = self.get_wetted_diameter(emitter_rate)
        area_per_emitter = math.pi * (wetted_diameter / 2) ** 2
        total_wetted_area = num_sprinklers * area_per_emitter * overlap_factor

        # If canopy radius is provided, use it to calculate target wetted area
        if canopy_radius and canopy_radius > 0:
            canopy_area = math.pi * canopy_radius ** 2
            # Use the smaller of wetted area or canopy area
            effective_area = min(total_wetted_area, canopy_area)
            logger.info(f"Using canopy radius {canopy_radius}m -> canopy area {canopy_area:.2f}m², wetted area {total_wetted_area:.2f}m², effective {effective_area:.2f}m²")
            return effective_area

        tree_area_allocation = plant_spacing * plant_spacing
        return min(total_wetted_area, tree_area_allocation)

    def calculate_irrigation_volume(self, deficit_mm, wetted_area_m2, irrigation_efficiency):
        gross_volume = deficit_mm * wetted_area_m2
        required_volume = gross_volume / irrigation_efficiency
        return required_volume

    def calculate_irrigation_time(self, volume, emitter_rate, num_sprinklers):
        total_flow_rate = emitter_rate * num_sprinklers
        logger.info(f"Irrigation time: Volume={volume:.1f}L, Rate={emitter_rate}L/h, Sprinklers={num_sprinklers}")

        if total_flow_rate > 0:
            irrigation_time_hours = volume / total_flow_rate
            irrigation_time_seconds = int(irrigation_time_hours * 3600)
        else:
            irrigation_time_seconds = 0

        irrigation_time_seconds = min(irrigation_time_seconds, 7200)
        logger.info(f"Final irrigation time: {irrigation_time_seconds}s = {irrigation_time_seconds/60:.1f}min")
        return irrigation_time_seconds

    def calculate_effective_rainfall(self, rainfall_mm, irrigation_type="Drip"):
        if rainfall_mm <= 0:
            return 0
        if rainfall_mm < 5:
            effectiveness = 0.6
        elif rainfall_mm < 15:
            effectiveness = 0.8
        elif rainfall_mm < 30:
            effectiveness = 0.9
        else:
            effectiveness = 0.7
        if irrigation_type == "Drip":
            effectiveness *= 1.05
        return rainfall_mm * effectiveness

    def calculate_soil_water_balance(self, etc_today, rainfall_mm, irrigation_applied, irrigation_type):
        effective_rainfall = self.calculate_effective_rainfall(rainfall_mm, irrigation_type)
        water_gains = irrigation_applied + effective_rainfall
        water_losses = etc_today
        current_deficit = self.previous_deficit + water_losses - water_gains

        if current_deficit <= 0:
            logger.info(f"Heavy rain effect: Deficit was {current_deficit:.1f}mm, setting to 0mm")
            current_deficit = 0

        return {
            "current_deficit": current_deficit,
            "effective_rainfall": effective_rainfall,
            "water_gains": water_gains,
            "water_losses": water_losses
        }

    def mark_irrigation_completed(self):
        self.previous_deficit = 0
        self.last_irrigation_time = datetime.now()
        logger.info("Irrigation completed - deficit reset to 0")

    def calculate_irrigation_recommendation(self, eto_mm_day, config, rainfall_mm=0.0, vpd=None):
        """Enhanced irrigation calculation with C1/C2/C3 updates:
        - C1: Plant Maturity Stage (affects Zr and MAD)
        - C2: Wetted % field
        - C3: Trunk Buffer Area option
        """
        swhc = self.get_swhc(config.soil_type)

        # Get config values with defaults
        canopy_radius = getattr(config, 'canopy_radius', 3.0) or 3.0
        plant_maturity = getattr(config, 'plant_maturity', 'Mature') or 'Mature'
        wetted_pct = getattr(config, 'wetted_pct', 50.0) or 50.0
        trunk_buffer_enabled = getattr(config, 'trunk_buffer_enabled', False) or False
        trunk_buffer_radius = getattr(config, 'trunk_buffer_radius', 0.0) or 0.0

        # C3: Calculate areas with trunk buffer consideration
        area_calc = self.calculate_root_and_wetted_area(
            canopy_radius, trunk_buffer_enabled, trunk_buffer_radius, wetted_pct
        )
        wetted_area = area_calc["wetted_area"]

        # C1: Get effective root zone based on plant maturity
        zr = self.get_effective_root_zone(plant_maturity)

        # C1: Calculate TAW using Zr instead of fixed soil_depth
        # TAW = SWHC (mm/m) × Zr (m)
        taw = swhc * zr

        # C1: Calculate RAW with MAD based on plant maturity and growth stage
        # Young: p = 0.5 (fixed)
        # Middle/Mature: p varies by growth stage
        p = 0.5  # Depletion fraction
        RAW = taw * p

        # C1: Calculate MAD threshold for irrigation trigger
        mad_ir = self.get_mad_ir(config.crop_growth_stage, plant_maturity)
        MAD_TI = RAW * mad_ir  # Management Allowable Depletion for Trigger Irrigation

        kc = self.get_kc(config.crop_type, config.crop_growth_stage)
        etc = eto_mm_day * kc

        water_balance = self.calculate_soil_water_balance(
            etc, rainfall_mm, self.daily_irrigation_applied, config.irrigation_type
        )

        current_deficit = water_balance["current_deficit"]
        effective_rainfall = water_balance["effective_rainfall"]
        irrigation_efficiency = 0.9 if config.irrigation_type == "Drip" else 0.8

        # VPD check
        vpd_status = None
        vpd_warning = None
        if vpd is not None:
            vpd_status = check_vpd_status(vpd, config.crop_growth_stage)
            if vpd_status["status"] == "high":
                vpd_warning = f"High VPD detected: {vpd_status['message']}"

        # Irrigation decision based on MAD_TI threshold
        if current_deficit >= MAD_TI:
            # C3 Formula: Net irrigation (L/tree) = Net_depth (mm) × A_wetted (m²)
            # Net_depth = current_deficit (mm converted to liters per m²)
            # Volume (L) = deficit (mm) × area (m²) = deficit × area (since 1mm = 1L/m²)
            net_irrigation = current_deficit * wetted_area  # Liters per tree
            irrigation_volume = net_irrigation / irrigation_efficiency  # Gross volume
            irrigation_time_seconds = self.calculate_irrigation_time(
                irrigation_volume, config.emitter_rate, config.num_sprinklers
            )
            self.previous_deficit = current_deficit

            recommendation = f"Pump activation recommended: {irrigation_volume:.1f} L per tree, {irrigation_time_seconds//60} min runtime"
            irrigation_needed = True
        else:
            irrigation_volume = 0
            irrigation_time_seconds = 0
            self.daily_irrigation_applied = 0
            self.previous_deficit = current_deficit
            recommendation = f"No pump activation needed. Deficit: {current_deficit:.1f}mm (Threshold: {MAD_TI:.1f}mm)"
            irrigation_needed = False

        # Add VPD override suggestion
        if vpd_status and vpd_status["status"] == "high" and not irrigation_needed:
            recommendation += f" | VPD Warning: {vpd_status['action']}"

        logger.info(f"Irrigation calc: Zr={zr}m, TAW={taw:.1f}mm, RAW={RAW:.1f}mm, MAD_TI={MAD_TI:.1f}mm, Deficit={current_deficit:.1f}mm")

        return {
            "irrigation_needed": irrigation_needed,
            "irrigation_volume_l_per_tree": round(irrigation_volume, 2),
            "irrigation_time_seconds": irrigation_time_seconds,
            "eto_mm_day": round(eto_mm_day, 3),
            "etc_mm_day": round(etc, 3),
            "deficit_mm": round(current_deficit, 2),
            "RAW_threshold_mm": round(RAW, 2),
            "MAD_TI_threshold_mm": round(MAD_TI, 2),
            "recommendation": recommendation,
            "rainfall_mm": round(rainfall_mm, 2),
            "effective_rainfall_mm": round(effective_rainfall, 2),
            "vpd_status": vpd_status,
            "vpd_warning": vpd_warning,
            "farm_config": {
                "crop_type": config.crop_type,
                "growth_stage": config.crop_growth_stage,
                "growth_day": growth_stage_config["current_day"],
                "soil_type": config.soil_type,
                "kc": round(kc, 2),
                "plant_maturity": plant_maturity,
                "effective_root_zone_m": zr,
                "taw_mm": round(taw, 2),
                "mad_ir": mad_ir,
                "wetted_pct": wetted_pct,
                "canopy_radius_m": canopy_radius,
                "canopy_area_m2": area_calc["canopy_area"],
                "buffer_area_m2": area_calc["buffer_area"],
                "root_area_m2": area_calc["root_area"],
                "wetted_area_m2": area_calc["wetted_area"],
                "trunk_buffer_enabled": trunk_buffer_enabled,
                "trunk_buffer_radius_m": trunk_buffer_radius
            }
        }

irrigation_calc = IrrigationCalculator()

# =========================
# Feature Engineering Functions
# =========================
def enhance_features_for_rad(tmin, tmax, humidity, wind_speed, sunshine_hours):
    data = {
        'MIN TEMP': tmin, 'MAX TEMP': tmax, 'HUMIDITY': humidity,
        'WIND': wind_speed, 'SUN': sunshine_hours,
    }
    data['TEMP_RANGE'] = tmax - tmin
    data['SOLAR_PROXY'] = sunshine_hours * (tmax / 35.0) * (1 - humidity / 100)
    data['SUN_TEMP_INTERACTION'] = sunshine_hours * tmax
    data['HUMIDITY_TEMP_RATIO'] = humidity / max(tmax, 1.0)

    if sunshine_hours <= 2: data['SUN_CATEGORY'] = 0
    elif sunshine_hours <= 5: data['SUN_CATEGORY'] = 1
    elif sunshine_hours <= 8: data['SUN_CATEGORY'] = 2
    elif sunshine_hours <= 15: data['SUN_CATEGORY'] = 3
    else: data['SUN_CATEGORY'] = 4

    data['WIND_CLEARNESS'] = 0 if wind_speed > 5.0 else 1

    rad_features = [
        data['MIN TEMP'], data['MAX TEMP'], data['HUMIDITY'],
        data['WIND'], data['SUN'], data['TEMP_RANGE'],
        data['SOLAR_PROXY'], data['SUN_TEMP_INTERACTION'],
        data['HUMIDITY_TEMP_RATIO'], data['SUN_CATEGORY'],
        data['WIND_CLEARNESS']
    ]
    return np.array([rad_features])

def apply_nighttime_constraints(rad_prediction, sunshine_hours, threshold=2.5):
    if sunshine_hours <= threshold:
        scale = max(0.3, sunshine_hours / threshold)
        constrained_rad = rad_prediction * scale
    else:
        constrained_rad = rad_prediction
    return max(constrained_rad, 1.0)

def fmt(value, unit=""):
    if isinstance(value, (int, float)):
        return f"{value:.2f}{unit}"
    return f"N/A{unit}"

def get_current_rainfall():
    """Get current rainfall - from sensor_status (15-min data) or prediction_history"""
    # First try sensor_status (updated by /receive-15min)
    rain_from_sensor = sensor_status.get("rain", {}).get("value")
    if rain_from_sensor is not None and rain_from_sensor > 0:
        return rain_from_sensor

    # Fallback to prediction_history
    if prediction_history:
        latest = prediction_history[-1]
        return latest.get("input_data", {}).get("rainfall_mm", 0)
    return 0

def get_current_solar_radiation():
    """Get current solar radiation from sensor_status (W/m²)"""
    solar_val = sensor_status.get("solar", {}).get("value")
    return solar_val if solar_val is not None else 0

# =========================
# Recovery Endpoints
# =========================
@app.post("/recover-missed-data")
async def recover_missed_data(request: RecoveryRequest):
    global last_processed_timestamp, recovery_status

    if not INFLUXDB_AVAILABLE:
        raise HTTPException(status_code=503, detail="InfluxDB client not installed")

    recovery_status["last_recovery_attempt"] = datetime.now().isoformat()

    try:
        records = influx_recovery.query_missed_data(
            since_timestamp=request.since_timestamp,
            hours_back=request.hours_back
        )

        recovered_count = 0
        for record in records:
            if all(record.get(k) is not None for k in ["temp_min", "temp_max", "humidity", "wind_speed", "sunshine_hours"]):
                prediction_data = {
                    "timestamp": record["timestamp"],
                    "device_id": record.get("device", "ESP32_RECOVERED"),
                    "input_data": {
                        "tmin": record["temp_min"],
                        "tmax": record["temp_max"],
                        "humidity": record["humidity"],
                        "wind_speed": record["wind_speed"],
                        "sunshine_hours": record["sunshine_hours"],
                        "rainfall_mm": record.get("rainfall_mm", 0),
                        "pressure": record.get("pressure"),
                        "vpd": record.get("vpd")
                    },
                    "recovered": True
                }
                prediction_history.append(prediction_data)
                recovered_count += 1

        recovery_status["records_recovered"] = recovered_count
        recovery_status["last_successful_recovery"] = datetime.now().isoformat()

        if records:
            last_processed_timestamp = records[-1]["timestamp"]

        return {
            "status": "success",
            "records_recovered": recovered_count,
            "last_timestamp": last_processed_timestamp,
            "recovery_status": recovery_status
        }

    except Exception as e:
        recovery_status["recovery_errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Recovery failed: {e}")

@app.get("/recovery/status")
async def get_recovery_status():
    return {
        "influxdb_available": INFLUXDB_AVAILABLE,
        "influxdb_config": {
            "url": INFLUXDB_CONFIG["url"],
            "bucket": INFLUXDB_CONFIG["bucket"],
            "org": INFLUXDB_CONFIG["org"]
        },
        "last_processed_timestamp": last_processed_timestamp,
        "recovery_status": recovery_status,
        "prediction_history_count": len(prediction_history)
    }

# =========================
# Daily Prediction Endpoints
# =========================
@app.post("/daily-prediction/run")
async def trigger_daily_prediction(target_date: str = None):
    """
    Manually trigger daily prediction.
    If target_date not provided, uses yesterday.
    Format: YYYY-MM-DD
    """
    if target_date:
        try:
            prediction_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        prediction_date = date.today() - timedelta(days=1)

    result = await run_daily_prediction(prediction_date)

    if result:
        return {
            "status": "success",
            "date": prediction_date.isoformat(),
            "result": result
        }
    else:
        raise HTTPException(status_code=500, detail="Daily prediction failed - check logs")


@app.post("/daily-prediction/simulate")
async def simulate_daily_prediction(data: dict = None):
    """
    SIMULATION ENDPOINT: Test daily ETo prediction with custom 24-hour data

    You can provide either:
    1. Pre-calculated daily values directly
    2. Array of 15-min records to aggregate

    Example 1 - Direct daily values:
    {
        "mode": "direct",
        "Tmin": 22.5,
        "Tmax": 34.2,
        "RHmean": 75.0,
        "WindMean": 1.5,
        "SunshineHours": 8.5,
        "Rain": 0.0
    }

    Example 2 - Simulate with 15-min records:
    {
        "mode": "aggregate",
        "records": [
            {"temp_min_15": 23, "temp_max_15": 25, "rh_avg_15": 80, "wind_avg_15": 1.0, "sunshine_min_15": 0, "rain_mm_15": 0},
            ...
        ]
    }

    Example 3 - Generate random test data:
    {
        "mode": "random",
        "num_records": 96
    }
    """
    global daily_prediction_result

    if data is None:
        data = {}

    mode = data.get("mode", "direct")

    logger.info(f"=== SIMULATING DAILY PREDICTION (mode: {mode}) ===")

    try:
        if mode == "direct":
            # Direct daily values provided
            daily = {
                "Tmin_day": data.get("Tmin", 24.0),
                "Tmax_day": data.get("Tmax", 32.0),
                "Tmean_day": data.get("Tmean", (data.get("Tmin", 24.0) + data.get("Tmax", 32.0)) / 2),
                "RHmean_day": data.get("RHmean", 75.0),
                "WindMean_day": data.get("WindMean", 1.5),
                "SunshineHours_day": data.get("SunshineHours", 8.0),
                "Rain_day": data.get("Rain", 0.0),
                "PressureMean_day": data.get("Pressure", 1013.0),
                "records_count": 96,
                "data_completeness": 100.0,
                "data_source": "simulation_direct"
            }

        elif mode == "aggregate":
            # Aggregate from provided 15-min records
            records = data.get("records", [])
            if not records:
                raise HTTPException(status_code=400, detail="No records provided for aggregation")

            daily = calculate_daily_aggregates_from_15min(records)
            if not daily:
                raise HTTPException(status_code=400, detail="Failed to calculate aggregates from records")
            daily["data_source"] = "simulation_aggregate"

        elif mode == "random":
            # Generate random realistic test data
            import random
            num_records = data.get("num_records", 96)

            # Base values for realistic simulation
            base_temp = random.uniform(24, 28)
            temp_range = random.uniform(6, 12)
            base_humidity = random.uniform(65, 85)

            records = []
            for i in range(num_records):
                # Simulate diurnal variation
                hour_of_day = (i * 15 / 60) % 24
                temp_offset = temp_range * 0.5 * math.sin((hour_of_day - 6) * math.pi / 12)

                temp_min = base_temp + temp_offset - random.uniform(0, 2)
                temp_max = base_temp + temp_offset + random.uniform(0, 2)

                # Sunshine only during day (6 AM - 6 PM)
                is_daytime = 6 <= hour_of_day <= 18
                sunshine_min = random.uniform(10, 15) if is_daytime and random.random() > 0.3 else 0

                records.append({
                    "temp_min_15": round(temp_min, 1),
                    "temp_max_15": round(temp_max, 1),
                    "temp_avg_15": round((temp_min + temp_max) / 2, 1),
                    "rh_avg_15": round(base_humidity + random.uniform(-10, 10), 1),
                    "wind_avg_15": round(random.uniform(0.5, 3.0), 1),
                    "sunshine_min_15": round(sunshine_min, 1),
                    "rain_mm_15": round(random.uniform(0, 0.5) if random.random() > 0.9 else 0, 2),
                    "pressure_avg_15": round(1010 + random.uniform(-5, 5), 1)
                })

            daily = calculate_daily_aggregates_from_15min(records)
            if not daily:
                raise HTTPException(status_code=400, detail="Failed to generate random data")
            daily["data_source"] = "simulation_random"
            daily["generated_records"] = num_records

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}. Use 'direct', 'aggregate', or 'random'")

        # Log the daily values
        logger.info(f"Daily values: Tmin={daily['Tmin_day']:.1f}, Tmax={daily['Tmax_day']:.1f}, "
                   f"RH={daily['RHmean_day']:.1f}%, Wind={daily['WindMean_day']:.1f}m/s, "
                   f"Sun={daily['SunshineHours_day']:.1f}h, Rain={daily['Rain_day']:.1f}mm")

        # Check if models are loaded
        if not all(models[key] is not None for key in ["rf_model", "mlp_model", "scaler_rad", "scaler_eto"]):
            raise HTTPException(status_code=500, detail="Models not loaded - cannot run prediction")

        # Run ML prediction
        # Step 1: RAD estimation
        rad_input = enhance_features_for_rad(
            daily["Tmin_day"], daily["Tmax_day"], daily["RHmean_day"],
            daily["WindMean_day"], daily["SunshineHours_day"]
        )

        rad_scaled = models["scaler_rad"].transform(rad_input)
        rad_prediction_raw = models["rf_model"].predict(rad_scaled)[0]
        estimated_rad = apply_nighttime_constraints(rad_prediction_raw, daily["SunshineHours_day"])

        # Step 2: ETo prediction
        eto_input = np.array([[
            daily["Tmin_day"], daily["Tmax_day"], daily["RHmean_day"],
            daily["WindMean_day"], daily["SunshineHours_day"], estimated_rad,
        ]])

        eto_scaled = models["scaler_eto"].transform(eto_input)
        eto_prediction_raw = models["mlp_model"].predict(eto_scaled)[0]
        eto_prediction = max(0.1, min(eto_prediction_raw, 15.0))

        logger.info(f"SIMULATED ETo: {eto_prediction:.3f} mm/day (RAD: {estimated_rad:.2f} MJ/m²)")

        # Calculate irrigation recommendation if config exists
        irrigation_recommendation = None
        if irrigation_config:
            config = IrrigationConfig(**irrigation_config)
            irrigation_recommendation = irrigation_calc.calculate_irrigation_recommendation(
                eto_prediction, config, daily["Rain_day"]
            )
            irrigation_recommendation["simulation"] = True

        # Build response
        result = {
            "status": "success",
            "simulation_mode": mode,
            "input_data": {
                "Tmin": daily["Tmin_day"],
                "Tmax": daily["Tmax_day"],
                "Tmean": daily["Tmean_day"],
                "RHmean": daily["RHmean_day"],
                "WindMean": daily["WindMean_day"],
                "SunshineHours": daily["SunshineHours_day"],
                "Rain": daily["Rain_day"],
                "data_source": daily.get("data_source", "unknown"),
                "records_count": daily.get("records_count", 0),
                "data_completeness_pct": daily.get("data_completeness", 0)
            },
            "ml_prediction": {
                "estimated_rad_MJ_m2": round(float(estimated_rad), 3),
                "eto_mm_day": round(float(eto_prediction), 3),
                "eto_raw": round(float(eto_prediction_raw), 4)
            },
            "irrigation_recommendation": irrigation_recommendation,
            "model_info": {
                "rad_model": "Random Forest",
                "eto_model": "MLP Neural Network",
                "trained_on": models["metadata"].get("training_date", "unknown") if models["metadata"] else "unknown"
            },
            "timestamp": datetime.now().isoformat()
        }

        # Optionally store as latest prediction
        if data.get("store_result", False):
            daily_prediction_result["daily_aggregates"] = daily
            daily_prediction_result["eto_prediction"] = result["ml_prediction"]
            daily_prediction_result["irrigation_recommendation"] = irrigation_recommendation
            daily_prediction_result["prediction_timestamp"] = datetime.now().isoformat()
            daily_prediction_result["last_prediction_date"] = "SIMULATION"
            daily_prediction_result["data_source"] = daily.get("data_source")
            result["stored"] = True

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")

@app.get("/daily-prediction/latest")
async def get_latest_daily_prediction():
    """Get the most recent daily prediction result"""
    if daily_prediction_result["eto_prediction"]:
        return daily_prediction_result
    else:
        return {"status": "no_prediction", "message": "No daily prediction has been run yet"}

@app.get("/daily-prediction/status")
async def get_daily_prediction_status():
    """Get status of the daily prediction system"""
    now = datetime.now()
    next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)

    # Get local storage stats
    today = date.today()
    yesterday = today - timedelta(days=1)
    today_records = len(get_local_15min_records_for_date(today))
    yesterday_records = len(get_local_15min_records_for_date(yesterday))

    return {
        "last_prediction_date": daily_prediction_result.get("last_prediction_date"),
        "last_prediction_time": daily_prediction_result.get("prediction_timestamp"),
        "data_source_used": daily_prediction_result.get("data_source", "unknown"),
        "next_scheduled_run": next_run.isoformat(),
        "hours_until_next_run": (next_run - now).total_seconds() / 3600,
        "local_storage": {
            "total_records": len(local_15min_records),
            "today_records": today_records,
            "today_coverage_pct": round(today_records / 96 * 100, 1),
            "yesterday_records": yesterday_records,
            "yesterday_coverage_pct": round(yesterday_records / 96 * 100, 1),
            "max_capacity": 100
        },
        "influxdb_available": INFLUXDB_AVAILABLE
    }

# =========================
# Growth Stage Endpoints
# =========================
@app.post("/growth-stage/set")
async def set_growth_stage(config: GrowthStageConfig):
    """Set the current growth day and optionally start date for auto-progression"""
    global growth_stage_config, irrigation_config

    if not 1 <= config.start_day <= 365:
        raise HTTPException(status_code=400, detail="Day must be between 1 and 365")

    growth_stage_config["current_day"] = config.start_day

    if config.start_date:
        try:
            start_date = datetime.strptime(config.start_date, "%Y-%m-%d").date()
            # Calculate what the start date should be to have current_day today
            days_back = config.start_day - 1
            growth_stage_config["start_date"] = date.today() - timedelta(days=days_back)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        # Set start_date based on current day
        days_back = config.start_day - 1
        growth_stage_config["start_date"] = date.today() - timedelta(days=days_back)

    growth_stage_config["last_update"] = datetime.now().isoformat()

    # Update irrigation config growth stage
    new_stage = get_growth_stage_from_day(config.start_day)
    irrigation_config["crop_growth_stage"] = new_stage

    logger.info(f"Growth stage set to Day {config.start_day} ({new_stage})")

    return {
        "status": "success",
        "current_day": growth_stage_config["current_day"],
        "growth_stage": new_stage,
        "start_date": growth_stage_config["start_date"].isoformat() if growth_stage_config["start_date"] else None,
        "vpd_range": VPD_RANGES.get(new_stage, {})
    }

@app.get("/growth-stage/status")
async def get_growth_stage_status():
    """Get current growth stage status"""
    update_growth_day()  # Update day if needed

    current_stage = get_growth_stage_from_day(growth_stage_config["current_day"])
    vpd_range = VPD_RANGES.get(current_stage, {})

    return {
        "current_day": growth_stage_config["current_day"],
        "growth_stage": current_stage,
        "start_date": growth_stage_config["start_date"].isoformat() if growth_stage_config["start_date"] else None,
        "last_update": growth_stage_config["last_update"],
        "vpd_range": vpd_range,
        "all_stages": GROWTH_STAGES
    }

# =========================
# Sensor Status Endpoint
# =========================
@app.get("/sensors/status")
async def get_sensor_status():
    """Get status of all sensors"""
    return {
        "sensors": sensor_status,
        "last_data_received": last_processed_timestamp,
        "summary": {
            "total": len(sensor_status),
            "online": sum(1 for s in sensor_status.values() if s["online"]),
            "offline": sum(1 for s in sensor_status.values() if not s["online"])
        }
    }

# =========================
# VPD Endpoint
# =========================
@app.get("/vpd/check")
async def check_current_vpd():
    """Check current VPD against growth stage ranges"""
    current_vpd = sensor_status["vpd"]["value"]
    if current_vpd is None and prediction_history:
        current_vpd = prediction_history[-1].get("input_data", {}).get("vpd")

    if current_vpd is None:
        return {"status": "no_data", "message": "No VPD data available"}

    current_stage = irrigation_config.get("crop_growth_stage", "Fruit growth (76-110 days)")
    vpd_status = check_vpd_status(current_vpd, current_stage)

    return {
        "current_vpd": current_vpd,
        "growth_stage": current_stage,
        "growth_day": growth_stage_config["current_day"],
        "vpd_status": vpd_status,
        "vpd_range": VPD_RANGES.get(current_stage, {})
    }

# =========================
# Enhanced Prediction Endpoint
# =========================
@app.post("/predict", response_model=EtoPrediction)
async def predict_eto_endpoint(weather_data: WeatherInput):
    global last_processed_timestamp

    if not all(models[key] is not None for key in ["rf_model", "mlp_model", "scaler_rad", "scaler_eto"]):
        raise HTTPException(status_code=500, detail="Models not loaded")

    start_time = datetime.now()
    warnings = []

    # Update growth day
    update_growth_day()

    try:
        # Update sensor status
        update_sensor_status("temperature", (weather_data.tmin + weather_data.tmax) / 2)
        update_sensor_status("humidity", weather_data.humidity)
        update_sensor_status("wind", weather_data.wind_speed)
        update_sensor_status("solar", weather_data.sunshine_hours)
        if weather_data.pressure:
            update_sensor_status("pressure", weather_data.pressure)
        if weather_data.vpd:
            update_sensor_status("vpd", weather_data.vpd)

        # Data quality checks
        if weather_data.tmin == weather_data.tmax:
            warnings.append("Temperature min/max are identical - sensor may need calibration")

        if weather_data.wind_speed > 20.0:
            warnings.append(f"High wind speed ({weather_data.wind_speed} m/s) - unusual but within sensor range")

        if weather_data.humidity > 95:
            warnings.append("Very high humidity - may affect prediction accuracy")

        if weather_data.rainfall_mm > 50:
            warnings.append(f"Heavy rainfall ({weather_data.rainfall_mm}mm) detected")

        logger.info(f"Weather data: T={weather_data.tmin}-{weather_data.tmax}C, RH={weather_data.humidity}%, Wind={weather_data.wind_speed}m/s, Sun={weather_data.sunshine_hours}h, Rain={weather_data.rainfall_mm}mm")

        # Step 1: RAD estimation
        rad_input_enhanced = enhance_features_for_rad(
            weather_data.tmin, weather_data.tmax, weather_data.humidity,
            weather_data.wind_speed, weather_data.sunshine_hours
        )

        rad_scaled = models["scaler_rad"].transform(rad_input_enhanced)
        rad_prediction_raw = models["rf_model"].predict(rad_scaled)[0]
        estimated_rad = apply_nighttime_constraints(rad_prediction_raw, weather_data.sunshine_hours)

        # Step 2: ETo prediction
        eto_input = np.array([[
            weather_data.tmin, weather_data.tmax, weather_data.humidity,
            weather_data.wind_speed, weather_data.sunshine_hours, estimated_rad,
        ]])

        eto_scaled = models["scaler_eto"].transform(eto_input)
        eto_prediction_raw = models["mlp_model"].predict(eto_scaled)[0]
        eto_prediction = max(0.1, min(eto_prediction_raw, 15.0))

        # Step 3: VPD check
        vpd_status = None
        if weather_data.vpd:
            current_stage = irrigation_config.get("crop_growth_stage", "Fruit growth (76-110 days)")
            vpd_status = check_vpd_status(weather_data.vpd, current_stage)
            if vpd_status["status"] in ["high", "low"]:
                warnings.append(vpd_status["message"])

        # Step 4: Enhanced irrigation calculation
        irrigation_result = None
        if irrigation_config:
            config = IrrigationConfig(**irrigation_config)
            irrigation_result = irrigation_calc.calculate_irrigation_recommendation(
                eto_prediction, config, weather_data.rainfall_mm, weather_data.vpd
            )
            irrigation_result["timestamp"] = weather_data.timestamp or datetime.now().isoformat()
            irrigation_history.append(irrigation_result)

            # Auto pump control
            if irrigation_result["irrigation_needed"] and irrigation_result["irrigation_time_seconds"] > 0:
                irrigation_time_seconds = irrigation_result["irrigation_time_seconds"]

                pump_result = await valve_controller.open_valve(irrigation_time_seconds)
                if pump_result:
                    irrigation_calc.mark_irrigation_completed()
                    logger.info(f"Pump activated automatically for {irrigation_time_seconds} seconds")
                    warnings.append(f"Pump activated for {irrigation_time_seconds // 60} minutes")
                else:
                    warnings.append("Failed to activate pump - check valve controller connection")

            logger.info(f"Irrigation (with rainfall): {irrigation_result['recommendation']}")

        await valve_controller.get_valve_status()

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        timestamp = weather_data.timestamp or datetime.now().isoformat()

        prediction_data = {
            "timestamp": timestamp,
            "device_id": weather_data.device_id,
            "input_data": {
                "tmin": weather_data.tmin,
                "tmax": weather_data.tmax,
                "temp_avg": (weather_data.tmin + weather_data.tmax) / 2,
                "humidity": weather_data.humidity,
                "wind_speed": weather_data.wind_speed,
                "sunshine_hours": weather_data.sunshine_hours,
                "rainfall_mm": weather_data.rainfall_mm,
                "pressure": weather_data.pressure,
                "vpd": weather_data.vpd,
            },
            "estimated_rad": round(float(estimated_rad), 3),
            "eto_mm_day": round(float(eto_prediction), 3),
            "processing_time_ms": round(processing_time, 2),
            "validation_warnings": warnings,
            "irrigation_recommendation": irrigation_result,
            "vpd_status": vpd_status
        }

        prediction_history.append(prediction_data)
        cycle_data.update(prediction_data)
        last_processed_timestamp = timestamp

        response = EtoPrediction(
            eto_mm_day=prediction_data["eto_mm_day"],
            estimated_rad=prediction_data["estimated_rad"],
            input_data=prediction_data["input_data"],
            timestamp=timestamp,
            device_id=weather_data.device_id,
            processing_time_ms=prediction_data["processing_time_ms"],
            validation_warnings=warnings,
            vpd_status=vpd_status,
            model_info={
                "trained_on": models["metadata"].get("training_date", "unknown") if models["metadata"] else "unknown",
                "eto_model_r2": models["metadata"]["performance_metrics"]["eto_model"]["test_r2"]
                if models["metadata"] and "performance_metrics" in models["metadata"] else 0,
                "rad_model_r2": models["metadata"]["performance_metrics"]["rad_model"]["test_r2"]
                if models["metadata"] and "performance_metrics" in models["metadata"] else 0,
            },
        )

        return response

    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Data validation failed: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# =========================
# ESP32-Compatible Endpoint
# =========================
# =========================
# NEW: 15-Minute Data Receive Endpoint (No ML Prediction)
# =========================
@app.post("/receive-15min")
async def receive_15min_data(data: dict):
    """
    NEW V5 ENDPOINT: Receive 15-minute aggregate from ESP32
    - Stores data locally for daily aggregation
    - Updates dashboard/sensor status
    - Does NOT run ML prediction (that happens at 6:00 AM)

    ESP32 can optionally send sensor_health dict with online status:
    {
        "sensor_health": {
            "temperature": true,
            "humidity": true,
            "pressure": true,
            "wind": true,
            "solar": false,  // sensor error
            "rain": true
        }
    }
    """
    try:
        # Extract data from ESP32 payload
        record = {
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "received_at": datetime.now().isoformat(),
            "temp_min_15": data.get("tmin"),
            "temp_max_15": data.get("tmax"),
            "temp_avg_15": (data.get("tmin", 0) + data.get("tmax", 0)) / 2 if data.get("tmin") and data.get("tmax") else None,
            "rh_avg_15": data.get("humidity"),
            "pressure_avg_15": data.get("pressure"),
            "wind_avg_15": data.get("wind_speed"),
            "wind_max_15": data.get("wind_speed"),
            "solar_avg_15": data.get("solar_radiation"),
            "sunshine_min_15": data.get("sunshine_minutes", 0),  # Direct minutes from ESP32
            "rain_mm_15": data.get("rainfall", data.get("rainfall_mm", 0)),
            "vpd_avg_15": data.get("vpd"),
            "device": data.get("device_id", "ESP32"),
            "aggregate_number": data.get("aggregate_number", 0)
        }

        # Store locally
        local_15min_records.append(record)

        # Get optional sensor health status from ESP32
        sensor_health = data.get("sensor_health", {})

        # Update sensor status for dashboard with smart validation
        # Temperature
        temp_online = sensor_health.get("temperature", True) if sensor_health else True
        if record["temp_avg_15"] is not None:
            update_sensor_status("temperature", record["temp_avg_15"], temp_online)

        # Humidity
        humidity_online = sensor_health.get("humidity", True) if sensor_health else True
        if record["rh_avg_15"] is not None:
            update_sensor_status("humidity", record["rh_avg_15"], humidity_online)

        # Pressure
        pressure_online = sensor_health.get("pressure", True) if sensor_health else True
        if record["pressure_avg_15"] is not None:
            update_sensor_status("pressure", record["pressure_avg_15"], pressure_online)

        # Wind
        wind_online = sensor_health.get("wind", True) if sensor_health else True
        if record["wind_avg_15"] is not None:
            update_sensor_status("wind", record["wind_avg_15"], wind_online)

        # Solar - special handling: 0 W/m² = offline unless ESP32 explicitly says online
        solar_value = record["solar_avg_15"]
        solar_online = sensor_health.get("solar", None)  # None means use smart detection
        if solar_online is None:
            # Smart detection: solar sensor is online only if value > 0
            # At night, ESP32 should send sensor_health.solar = true to indicate sensor is working
            solar_online = solar_value is not None and solar_value > 0
        if solar_value is not None:
            update_sensor_status("solar", solar_value, solar_online)

        # Rain
        rain_online = sensor_health.get("rain", True) if sensor_health else True
        if record["rain_mm_15"] is not None:
            update_sensor_status("rain", record["rain_mm_15"], rain_online)

        # VPD (calculated, so online if temp and humidity are online)
        vpd_online = temp_online and humidity_online
        if record["vpd_avg_15"] is not None:
            update_sensor_status("vpd", record["vpd_avg_15"], vpd_online)

        # Get local storage stats
        today = date.today()
        today_records = len(get_local_15min_records_for_date(today))

        logger.info(f"[RECEIVE-15MIN] Stored record #{record['aggregate_number']} | "
                   f"T:{record['temp_min_15']}-{record['temp_max_15']}°C | "
                   f"RH:{record['rh_avg_15']}% | Rain:{record['rain_mm_15']}mm | "
                   f"Today: {today_records}/96 records")

        # Build sensor status summary for response
        sensors_online = sum(1 for s in sensor_status.values() if s.get("online", False))
        sensors_total = len(sensor_status)

        return {
            "status": "success",
            "message": "15-min data received and stored",
            "record_stored": {
                "timestamp": record["timestamp"],
                "aggregate_number": record["aggregate_number"],
                "temp_range": f"{record['temp_min_15']}-{record['temp_max_15']}°C",
                "humidity": record["rh_avg_15"],
                "rain_mm": record["rain_mm_15"],
                "sunshine_min": record["sunshine_min_15"]
            },
            "local_storage": {
                "total_records": len(local_15min_records),
                "today_records": today_records,
                "today_coverage_pct": round(today_records / 96 * 100, 1)
            },
            "sensor_status": {
                "online_count": sensors_online,
                "total_count": sensors_total,
                "details": {name: s.get("online", False) for name, s in sensor_status.items()}
            },
            "next_daily_prediction": "6:00 AM"
        }

    except Exception as e:
        logger.error(f"Failed to receive 15-min data: {e}")
        raise HTTPException(status_code=422, detail=f"Data processing failed: {e}")


# =========================
# LEGACY: ESP32-Compatible Endpoint (Still runs ML - for backward compatibility)
# =========================
@app.post("/predict-esp32")
async def predict_eto_esp32_compatible(data: dict):
    """
    LEGACY ENDPOINT: ESP32-compatible endpoint that runs ML prediction
    NOTE: For V5 architecture, use /receive-15min instead (no ML prediction)
    This endpoint is kept for backward compatibility with V4
    """
    try:
        converted_data = {
            "tmin": data.get("tmin"),
            "tmax": data.get("tmax"),
            "humidity": data.get("humidity"),
            "wind_speed": data.get("wind_speed"),
            "sunshine_hours": data.get("sunshine_hours"),
            "rainfall_mm": data.get("rainfall", data.get("rainfall_mm", 0.0)),
            "pressure": data.get("pressure"),
            "vpd": data.get("vpd"),
            "device_id": data.get("device_id", "ESP32"),
            "timestamp": data.get("timestamp")
        }

        # Update rain sensor status
        update_sensor_status("rain", converted_data["rainfall_mm"])

        # Store 15-min record locally for daily aggregation
        store_15min_record_locally(converted_data)
        logger.info(f"[LEGACY] Stored 15-min record locally (total: {len(local_15min_records)} records)")

        weather_input = WeatherInput(**converted_data)
        return await predict_eto_endpoint(weather_input)

    except Exception as e:
        logger.error(f"ESP32 prediction failed: {e}")
        raise HTTPException(status_code=422, detail=f"ESP32 data conversion failed: {e}")

# =========================
# Valve Control Endpoints
# =========================
@app.post("/pump/control")
async def control_pump(command: PumpCommand):
    global manual_pump_session, unlimited_pump_session

    try:
        if command.action == "open":
            if command.duration is None:
                unlimited_pump_session = {
                    "active": True,
                    "start_time": datetime.now(),
                    "total_runtime_seconds": 0
                }
                manual_pump_session = {"active": False, "duration_seconds": 0, "start_time": None}

                result = await valve_controller.open_valve(86400)

                if result:
                    return {
                        "status": "success",
                        "message": "Pump activated in unlimited mode",
                        "mode": "unlimited",
                        "valve_status": valve_status,
                        "unlimited_session": unlimited_pump_session
                    }
                else:
                    unlimited_pump_session["active"] = False
                    raise HTTPException(status_code=500, detail="Failed to activate pump")

            elif command.duration <= 0:
                raise HTTPException(status_code=400, detail="Duration must be positive")

            else:
                duration = min(command.duration, 10800)

                manual_pump_session = {
                    "active": True,
                    "duration_seconds": duration,
                    "start_time": datetime.now()
                }
                unlimited_pump_session = {"active": False, "start_time": None, "total_runtime_seconds": 0}

                result = await valve_controller.open_valve(duration)

                if result:
                    if irrigation_history and irrigation_history[-1].get("irrigation_needed", False):
                        irrigation_calc.mark_irrigation_completed()
                        logger.info("Manual irrigation completed - deficit reset")

                    return {
                        "status": "success",
                        "message": f"Pump activated for {duration} seconds",
                        "mode": "timed",
                        "valve_status": valve_status,
                        "manual_session": manual_pump_session
                    }
                else:
                    manual_pump_session["active"] = False
                    raise HTTPException(status_code=500, detail="Failed to activate pump")

        elif command.action == "close":
            water_discharged = 0
            runtime_seconds = 0

            if unlimited_pump_session.get("active") and unlimited_pump_session.get("start_time"):
                runtime_seconds = (datetime.now() - unlimited_pump_session["start_time"]).total_seconds()
                emitter_rate = irrigation_config.get('emitter_rate', 50)
                num_sprinklers = irrigation_config.get('num_sprinklers', 2)
                total_flow_rate = emitter_rate * num_sprinklers
                water_discharged = (total_flow_rate * runtime_seconds) / 3600

            manual_pump_session = {"active": False, "duration_seconds": 0, "start_time": None}
            unlimited_pump_session["active"] = False

            result = await valve_controller.close_valve()

            if result:
                response = {
                    "status": "success",
                    "message": "Pump deactivated",
                    "valve_status": valve_status
                }

                if water_discharged > 0:
                    response["summary"] = {
                        "runtime_seconds": int(runtime_seconds),
                        "runtime_minutes": round(runtime_seconds / 60, 1),
                        "water_discharged_liters": round(water_discharged, 2),
                        "emitter_rate": irrigation_config.get('emitter_rate', 0),
                        "num_sprinklers": irrigation_config.get('num_sprinklers', 0)
                    }

                return response
            else:
                raise HTTPException(status_code=500, detail="Failed to deactivate pump")

        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'open' or 'close'")

    except Exception as e:
        manual_pump_session["active"] = False
        unlimited_pump_session["active"] = False
        logger.error(f"Pump control failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pump control error: {e}")

@app.get("/pump/status")
async def get_pump_status():
    await valve_controller.get_valve_status()
    return {
        "pump_status": valve_status,
        "controller_ip": VALVE_CONTROLLER_IP,
        "last_checked": datetime.now().isoformat()
    }

@app.get("/pump/unlimited-session")
async def get_unlimited_pump_session():
    if unlimited_pump_session.get("active") and unlimited_pump_session.get("start_time"):
        elapsed = (datetime.now() - unlimited_pump_session["start_time"]).total_seconds()
        emitter_rate = irrigation_config.get('emitter_rate', 50)
        num_sprinklers = irrigation_config.get('num_sprinklers', 2)
        total_flow_rate = emitter_rate * num_sprinklers
        water_discharged = (total_flow_rate * elapsed) / 3600

        return {
            "active": True,
            "elapsed_seconds": int(elapsed),
            "elapsed_minutes": round(elapsed / 60, 1),
            "start_time": unlimited_pump_session["start_time"].isoformat(),
            "water_discharged_liters": round(water_discharged, 2),
            "flow_rate_lph": total_flow_rate
        }
    else:
        return {"active": False}

@app.get("/pump/manual-session")
async def get_manual_pump_session():
    if manual_pump_session["active"] and manual_pump_session["start_time"]:
        elapsed = (datetime.now() - manual_pump_session["start_time"]).total_seconds()
        remaining = max(0, manual_pump_session["duration_seconds"] - elapsed)

        return {
            "active": manual_pump_session["active"],
            "duration_seconds": manual_pump_session["duration_seconds"],
            "elapsed_seconds": int(elapsed),
            "remaining_seconds": int(remaining),
            "start_time": manual_pump_session["start_time"].isoformat(),
            "auto_close_expected": (manual_pump_session["start_time"] + timedelta(seconds=manual_pump_session["duration_seconds"])).isoformat()
        }
    else:
        return {"active": False}

@app.post("/valve/config")
async def update_valve_config(ip: str, port: int = 80):
    global valve_controller, VALVE_CONTROLLER_IP, VALVE_CONTROLLER_PORT

    VALVE_CONTROLLER_IP = ip
    VALVE_CONTROLLER_PORT = port
    valve_controller = ValveController(ip, port)

    await valve_controller.get_valve_status()

    return {
        "status": "success",
        "message": f"Valve controller configured to {ip}:{port}",
        "connection_status": valve_status["connection_status"]
    }

# =========================
# Irrigation Endpoints
# =========================
@app.post("/irrigation/config")
async def set_irrigation_config(config: IrrigationConfig):
    global irrigation_config
    irrigation_config = config.dict()
    logger.info(f"Irrigation config updated: {config.farm_name} - {config.crop_type}")
    return {"status": "success", "message": "Irrigation configuration updated", "config": irrigation_config}

@app.get("/irrigation/config")
async def get_irrigation_config():
    return irrigation_config

@app.get("/irrigation/latest")
async def get_latest_irrigation():
    if irrigation_history:
        return irrigation_history[-1]
    return {"message": "No irrigation recommendations available"}

@app.get("/irrigation/history")
async def get_irrigation_history():
    return list(irrigation_history)

# =========================
# Raw Sensor Data Page
# =========================
@app.get("/raw-data", response_class=HTMLResponse)
async def raw_sensor_data_page():
    """Display stored 15-minute aggregate records"""

    # Get local 15-min records
    records = list(local_15min_records)
    records.reverse()  # Most recent first

    # Build table rows
    table_rows = ""
    for i, record in enumerate(records[:50]):  # Show last 50 records
        timestamp = record.get("timestamp", record.get("received_at", "N/A"))
        if timestamp and len(timestamp) > 19:
            timestamp = timestamp[:19]  # Trim to readable format

        table_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td>{timestamp}</td>
            <td>{record.get('temp_min_15', 'N/A')}</td>
            <td>{record.get('temp_max_15', 'N/A')}</td>
            <td>{record.get('rh_avg_15', 'N/A')}</td>
            <td>{record.get('wind_avg_15', 'N/A')}</td>
            <td>{record.get('pressure_avg_15', 'N/A')}</td>
            <td>{record.get('sunshine_min_15', 'N/A')}</td>
            <td>{record.get('rain_mm_15', 'N/A')}</td>
            <td>{record.get('vpd_avg_15', 'N/A')}</td>
        </tr>
        """

    if not table_rows:
        table_rows = "<tr><td colspan='10' style='text-align:center; padding:20px;'>No data received yet. Waiting for ESP32...</td></tr>"

    # Get storage stats
    today = date.today()
    yesterday = today - timedelta(days=1)
    today_records = len(get_local_15min_records_for_date(today))
    yesterday_records = len(get_local_15min_records_for_date(yesterday))

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raw Data | AquaSense</title>
        <meta http-equiv="refresh" content="60">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --primary: #0077B6;
                --primary-dark: #005f8a;
                --primary-light: #E1F5FE;
                --accent: #00B4D8;
                --accent-light: #90E0EF;
                --success: #10B981;
                --success-light: #D1FAE5;
                --warning: #F59E0B;
                --warning-light: #FEF3C7;
                --danger: #EF4444;
                --danger-light: #FEE2E2;
                --gray-50: #F9FAFB;
                --gray-100: #F3F4F6;
                --gray-200: #E5E7EB;
                --gray-300: #D1D5DB;
                --gray-400: #9CA3AF;
                --gray-500: #6B7280;
                --gray-600: #4B5563;
                --gray-700: #374151;
                --gray-800: #1F2937;
                --gray-900: #111827;
                --white: #FFFFFF;
                --radius: 12px;
                --radius-lg: 16px;
                --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
                --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--gray-50);
                min-height: 100vh;
                color: var(--gray-800);
                line-height: 1.5;
            }}
            .page-wrapper {{ display: flex; min-height: 100vh; }}

            /* Sidebar */
            .sidebar {{
                width: 220px;
                background: var(--white);
                border-right: 1px solid var(--gray-200);
                padding: 20px 0;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
                z-index: 100;
            }}
            .sidebar-logo {{
                padding: 0 20px 20px;
                border-bottom: 1px solid var(--gray-100);
                margin-bottom: 16px;
            }}
            .logo-container {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .logo-icon {{
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.3rem;
            }}
            .logo-text {{
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--primary);
            }}
            .logo-subtitle {{
                font-size: 0.65rem;
                color: var(--gray-500);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .sidebar-nav {{ padding: 0 12px; }}
            .nav-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 14px;
                color: var(--gray-600);
                text-decoration: none;
                border-radius: var(--radius);
                margin-bottom: 4px;
                font-size: 0.875rem;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .nav-item:hover {{ background: var(--gray-100); color: var(--gray-800); }}
            .nav-item.active {{
                background: var(--primary-light);
                color: var(--primary);
                font-weight: 600;
            }}
            .nav-icon {{ font-size: 1rem; width: 20px; text-align: center; }}

            /* Main Content */
            .main-content {{
                flex: 1;
                margin-left: 220px;
                padding: 20px 24px;
            }}

            /* Top Bar */
            .top-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                flex-wrap: wrap;
                gap: 12px;
            }}
            .page-title {{
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--gray-900);
            }}
            .datetime-display {{
                text-align: right;
            }}
            .datetime-date {{ font-size: 0.85rem; font-weight: 600; color: var(--gray-800); }}
            .datetime-time {{ font-size: 0.75rem; color: var(--gray-500); }}

            /* Stats Grid */
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }}
            .stat-card {{
                background: var(--white);
                border-radius: var(--radius);
                border: 1px solid var(--gray-200);
                padding: 20px;
                text-align: center;
                transition: all 0.2s;
            }}
            .stat-card:hover {{
                box-shadow: var(--shadow-md);
                transform: translateY(-2px);
            }}
            .stat-icon {{
                width: 40px;
                height: 40px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 10px;
                font-size: 1.2rem;
            }}
            .stat-value {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
            .stat-label {{ font-size: 0.8rem; color: var(--gray-500); margin-top: 4px; }}

            /* Section Header */
            .section-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 16px;
            }}
            .section-icon {{
                width: 32px;
                height: 32px;
                background: var(--primary-light);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1rem;
            }}
            .section-title {{ font-size: 1rem; font-weight: 600; color: var(--gray-800); }}

            /* Data Table */
            .data-table-wrapper {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                overflow: hidden;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .data-table th {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
                padding: 14px 10px;
                text-align: left;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .data-table td {{
                padding: 12px 10px;
                border-bottom: 1px solid var(--gray-100);
                font-size: 0.85rem;
                color: var(--gray-700);
            }}
            .data-table tr:hover {{ background: var(--gray-50); }}
            .data-table tr:nth-child(even) {{ background: var(--gray-50); }}
            .data-table tr:nth-child(even):hover {{ background: var(--gray-100); }}

            /* Refresh Info */
            .refresh-info {{
                text-align: center;
                color: var(--gray-500);
                font-size: 0.75rem;
                margin-top: 16px;
                padding: 12px;
                background: var(--white);
                border-radius: var(--radius);
                border: 1px solid var(--gray-200);
            }}

            /* Responsive */
            @media (max-width: 768px) {{
                .sidebar {{ display: none; }}
                .main-content {{ margin-left: 0; padding: 16px; }}
                .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
                .data-table {{ font-size: 0.75rem; }}
                .data-table th, .data-table td {{ padding: 8px 6px; }}
            }}
        </style>
    </head>
    <body>
        <div class="page-wrapper">
            <!-- Sidebar -->
            <aside class="sidebar">
                <div class="sidebar-logo">
                    <div class="logo-container">
                        <div class="logo-icon">💧</div>
                        <div>
                            <div class="logo-text">AquaSense</div>
                            <div class="logo-subtitle">Smart Irrigation</div>
                        </div>
                    </div>
                </div>
                <nav class="sidebar-nav">
                    <a href="/dashboard" class="nav-item">
                        <span class="nav-icon">📊</span> Dashboard
                    </a>
                    <a href="/config" class="nav-item">
                        <span class="nav-icon">⚙️</span> Settings
                    </a>
                    <a href="/raw-data" class="nav-item active">
                        <span class="nav-icon">📈</span> Raw Data
                    </a>
                    <a href="/docs" class="nav-item">
                        <span class="nav-icon">📄</span> API Docs
                    </a>
                </nav>
            </aside>

            <!-- Main Content -->
            <main class="main-content">
                <!-- Top Bar -->
                <div class="top-bar">
                    <h1 class="page-title">Raw Sensor Data</h1>
                    <div class="datetime-display">
                        <div class="datetime-date">{datetime.now().strftime("%A, %d %B %Y")}</div>
                        <div class="datetime-time">{datetime.now().strftime("%H:%M")} • Auto-refresh 60s</div>
                    </div>
                </div>

                <!-- Stats Grid -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon" style="background: var(--primary-light);">📦</div>
                        <div class="stat-value">{len(local_15min_records)}</div>
                        <div class="stat-label">Total Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon" style="background: var(--success-light);">📅</div>
                        <div class="stat-value">{today_records}</div>
                        <div class="stat-label">Today ({today_records/96*100:.0f}%)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon" style="background: var(--warning-light);">📆</div>
                        <div class="stat-value">{yesterday_records}</div>
                        <div class="stat-label">Yesterday ({yesterday_records/96*100:.0f}%)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon" style="background: var(--gray-100);">💾</div>
                        <div class="stat-value">100</div>
                        <div class="stat-label">Max Capacity</div>
                    </div>
                </div>

                <!-- Data Table Section -->
                <div class="section-header">
                    <div class="section-icon">📊</div>
                    <span class="section-title">15-Minute Aggregates from ESP32</span>
                    <span style="margin-left: auto; font-size: 0.75rem; color: var(--gray-500);">Most recent first</span>
                </div>
                <div class="data-table-wrapper">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Timestamp</th>
                                <th>T_min (°C)</th>
                                <th>T_max (°C)</th>
                                <th>RH (%)</th>
                                <th>Wind (m/s)</th>
                                <th>Pressure (hPa)</th>
                                <th>Sunshine (min)</th>
                                <th>Rain (mm)</th>
                                <th>VPD (kPa)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>

                <div class="refresh-info">
                    🔄 Auto-refreshes every 60 seconds | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </main>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# =========================
# Debug Endpoints
# =========================
@app.get("/debug/latest-data")
async def debug_latest_data():
    return {
        "latest_prediction": prediction_history[-1] if prediction_history else "No predictions",
        "cycle_data": cycle_data,
        "irrigation_history_latest": irrigation_history[-1] if irrigation_history else "No irrigation data",
        "prediction_count": len(prediction_history),
        "valve_status": valve_status,
        "current_deficit": irrigation_calc.previous_deficit,
        "last_processed_timestamp": last_processed_timestamp,
        "recovery_status": recovery_status,
        "growth_stage": {
            "current_day": growth_stage_config["current_day"],
            "stage": get_growth_stage_from_day(growth_stage_config["current_day"])
        },
        "sensor_status": sensor_status
    }

# =========================
# Health & Info Endpoints
# =========================
@app.get("/health")
async def health():
    await valve_controller.get_valve_status()
    return {
        "status": "ok",
        "version": "3.0.0",
        "models_loaded": all(models.values()),
        "irrigation_enabled": bool(irrigation_config),
        "valve_controller": valve_status,
        "influxdb_recovery": INFLUXDB_AVAILABLE,
        "last_processed_timestamp": last_processed_timestamp,
        "growth_day": growth_stage_config["current_day"],
        "sensors_online": sum(1 for s in sensor_status.values() if s["online"])
    }

@app.get("/model-info")
async def model_info():
    return {
        "metadata": models["metadata"],
        "loaded_at": models["loaded_at"],
        "irrigation_config": irrigation_config,
        "valve_status": valve_status,
        "feature_info": {
            "rad_features": 11,
            "eto_features": 6,
            "enhancement": "Phase 2: VPD checking, Growth stage tracking, Canopy radius"
        },
        "validation_ranges": {
            "temperature": "-50 to 60 C",
            "humidity": "0 to 100 %",
            "wind_speed": "0 to 45 m/s (Rika RK120-01)",
            "sunshine_hours": "0 to 24 h",
            "rainfall": "0 to 500 mm"
        },
        "growth_stage_config": growth_stage_config,
        "vpd_ranges": VPD_RANGES
    }

# =========================
# Dashboard Page
# =========================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    global manual_pump_session
    await valve_controller.get_valve_status()
    update_growth_day()

    latest_data = cycle_data if cycle_data else {}
    latest_irrigation = irrigation_history[-1] if irrigation_history else {}

    current_rainfall = get_current_rainfall()

    # Prepare data for charts - use local_15min_records (from /receive-15min)
    history_data = {
        'labels': [], 'humidity': [], 'wind_speed': [], 'sunshine_min': [],
        'solar_wm2': [], 'temp_min': [], 'temp_max': [], 'pressure': [],
        'rainfall': [], 'vpd': []
    }

    # Get last 12 records from local_15min_records (most recent 15-min data)
    recent_records = list(local_15min_records)[-12:]
    for record in recent_records:
        timestamp = record.get('timestamp', record.get('received_at', ''))
        if timestamp:
            try:
                # Handle different timestamp formats
                if 'T' in str(timestamp):
                    dt = datetime.fromisoformat(str(timestamp).replace('Z', '').split('+')[0])
                else:
                    dt = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
                formatted_time = dt.strftime('%H:%M')
            except:
                formatted_time = str(timestamp)[-8:-3] if len(str(timestamp)) > 8 else str(timestamp)

            history_data['labels'].append(formatted_time)
            history_data['humidity'].append(round(record.get('rh_avg_15', 0) or 0, 2))
            history_data['wind_speed'].append(round(record.get('wind_avg_15', 0) or 0, 2))
            history_data['sunshine_min'].append(record.get('sunshine_min_15', 0) or 0)
            history_data['solar_wm2'].append(round(record.get('solar_avg_15', 0) or 0, 2))
            history_data['temp_min'].append(round(record.get('temp_min_15', 0) or 0, 2))
            history_data['temp_max'].append(round(record.get('temp_max_15', 0) or 0, 2))
            history_data['pressure'].append(round(record.get('pressure_avg_15', 0) or 0, 2))
            history_data['rainfall'].append(round(record.get('rain_mm_15', 0) or 0, 2))
            history_data['vpd'].append(round(record.get('vpd_avg_15', 0) or 0, 2))

    # Current values - try cycle_data first, then fall back to sensor_status (from /receive-15min)
    current_eto = latest_data.get("eto_mm_day", 0)
    current_etc = latest_irrigation.get("etc_mm_day", 0) if latest_irrigation else 0
    current_rad = latest_data.get("estimated_rad", 0)

    # Temperature: use cycle_data if available, otherwise use sensor_status
    current_temp_min = latest_data.get("input_data", {}).get("tmin", 0)
    current_temp_max = latest_data.get("input_data", {}).get("tmax", 0)
    if current_temp_min == 0 and current_temp_max == 0:
        temp_val = sensor_status.get("temperature", {}).get("value")
        if temp_val:
            current_temp_min = temp_val
            current_temp_max = temp_val

    # Other sensors: use cycle_data if available, otherwise use sensor_status
    current_humidity = latest_data.get("input_data", {}).get("humidity", 0)
    if current_humidity == 0:
        current_humidity = sensor_status.get("humidity", {}).get("value") or 0

    current_wind = latest_data.get("input_data", {}).get("wind_speed", 0)
    if current_wind == 0:
        current_wind = sensor_status.get("wind", {}).get("value") or 0

    current_sunshine = latest_data.get("input_data", {}).get("sunshine_hours", 0)
    if current_sunshine == 0:
        current_sunshine = sensor_status.get("solar", {}).get("value") or 0

    current_pressure = latest_data.get("input_data", {}).get("pressure", 0) or 0
    if current_pressure == 0:
        current_pressure = sensor_status.get("pressure", {}).get("value") or 0

    current_vpd = latest_data.get("input_data", {}).get("vpd", 0) or 0
    if current_vpd == 0:
        current_vpd = sensor_status.get("vpd", {}).get("value") or 0

    # Solar radiation from sensor (W/m²) - real-time from ESP32
    current_solar_wm2 = get_current_solar_radiation()

    # Rn_est from daily prediction (MJ/m²/day) - updated at 6:00 AM
    current_rn_est = daily_prediction_result.get("eto_prediction", {}).get("estimated_rad", 0) if daily_prediction_result.get("eto_prediction") else current_rad

    irrigation_volume = latest_irrigation.get("irrigation_volume_l_per_tree", 0)
    irrigation_needed = latest_irrigation.get("irrigation_needed", False)

    # Growth stage info
    current_day = growth_stage_config["current_day"]
    current_stage = get_growth_stage_from_day(current_day)
    vpd_range = VPD_RANGES.get(current_stage, {"min": 0, "max": 2, "optimal": 1})

    # Get current Kc value from irrigation calculator
    current_kc = irrigation_calc.get_kc(
        irrigation_config.get("crop_type", "Durian"),
        current_stage
    )

    # VPD status
    vpd_status = check_vpd_status(current_vpd, current_stage) if current_vpd else {"status": "unknown", "color": "#999"}

    # Irrigation time calculation
    def get_irrigation_time():
        if unlimited_pump_session.get("active") and unlimited_pump_session.get("start_time"):
            return -1
        if manual_pump_session.get("active", False) and manual_pump_session.get("start_time"):
            elapsed = (datetime.now() - manual_pump_session["start_time"]).total_seconds()
            remaining = max(0, manual_pump_session["duration_seconds"] - elapsed)
            if remaining > 0:
                return remaining / 60
            else:
                manual_pump_session["active"] = False
        emitter_rate = irrigation_config.get('emitter_rate', 50)
        num_sprinklers = irrigation_config.get('num_sprinklers', 2)
        total_flow_rate = emitter_rate * num_sprinklers
        automatic_time = (irrigation_volume / total_flow_rate * 60) if total_flow_rate > 0 and irrigation_volume > 0 else 0
        return automatic_time

    irrigation_time_minutes = get_irrigation_time()

    # Pump status
    pump_is_on = valve_status.get("state", "closed") == "open"
    manual_is_active = manual_pump_session.get("active", False)
    pump_is_effectively_on = pump_is_on or manual_is_active

    if pump_is_effectively_on:
        pump_status_color = 'linear-gradient(135deg, #4caf50 0%, #388e3c 100%)'
        pump_status_text = 'PUMP STATUS: ON (Manual)' if manual_is_active else 'PUMP STATUS: ON'
    else:
        if irrigation_needed:
            pump_status_color = 'linear-gradient(135deg, #ef5350 0%, #e53935 100%)'
            pump_status_text = 'PUMP ACTIVATION REQUIRED'
        else:
            pump_status_color = 'linear-gradient(135deg, #f44336 0%, #d32f2f 100%)'
            pump_status_text = 'PUMP STATUS: OFF'

    pump_animation = 'pumpPulse 2s ease-in-out infinite' if pump_is_on else 'irrigationPulse 3s ease-in-out infinite' if irrigation_needed else 'none'

    # Water level
    deficit_mm = latest_irrigation.get("deficit_mm", 0)
    threshold_mm = latest_irrigation.get("RAW_threshold_mm", 100)

    if threshold_mm > 0:
        water_level_percent = max(0, min(100, (1 - (deficit_mm / threshold_mm)) * 100))
    else:
        water_level_percent = 100

    if water_level_percent <= 30:
        water_fill_color = '#ef5350'
    elif water_level_percent <= 70:
        water_fill_color = '#ffa726'
    else:
        water_fill_color = '#4caf50'

    # Sensor status indicators
    def sensor_indicator(name):
        s = sensor_status.get(name, {"online": False})
        color = "#4caf50" if s["online"] else "#999"
        return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};margin-right:5px;"></span>'

    # Pre-compute sensor status classes for HTML (avoids f-string issues with empty dict)
    temp_status_class = 'online' if sensor_status.get('temperature', {}).get('online') else 'offline'
    humidity_status_class = 'online' if sensor_status.get('humidity', {}).get('online') else 'offline'
    wind_status_class = 'online' if sensor_status.get('wind', {}).get('online') else 'offline'
    solar_status_class = 'online' if sensor_status.get('solar', {}).get('online') else 'offline'
    rain_status_class = 'online' if sensor_status.get('rain', {}).get('online') else 'offline'
    pressure_status_class = 'online' if sensor_status.get('pressure', {}).get('online') else 'offline'

    # Countdown section
    countdown_section = ""
    countdown_script = ""
    irrigation_time_value = get_irrigation_time()

    if pump_is_effectively_on:
        if irrigation_time_value == -1:
            elapsed = (datetime.now() - unlimited_pump_session["start_time"]).total_seconds()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            emitter_rate = irrigation_config.get('emitter_rate', 50)
            num_sprinklers = irrigation_config.get('num_sprinklers', 2)
            total_flow_rate = emitter_rate * num_sprinklers
            water_discharged = (total_flow_rate * elapsed) / 3600

            countdown_section = f'''
            <div class="countdown-timer unlimited-mode">
                <div class="countdown-label">UNLIMITED MODE - Running Time</div>
                <div class="countdown-time" id="countdownTime">{minutes}:{seconds:02d}</div>
                <div class="countdown-label">Water Discharged</div>
                <div class="countdown-time" id="waterDischarged">{water_discharged:.1f} L</div>
            </div>
            '''
            countdown_script = f'''
            let elapsedSeconds = {int(elapsed)};
            let flowRate = {total_flow_rate};
            function updateUnlimitedTimer() {{
                const minutes = Math.floor(elapsedSeconds / 60);
                const seconds = elapsedSeconds % 60;
                const waterLiters = (flowRate * elapsedSeconds) / 3600;
                const countdownElement = document.getElementById('countdownTime');
                const waterElement = document.getElementById('waterDischarged');
                if (countdownElement) countdownElement.textContent = `${{minutes}}:${{seconds.toString().padStart(2, '0')}}`;
                if (waterElement) waterElement.textContent = waterLiters.toFixed(1) + ' L';
                elapsedSeconds++;
            }}
            if (document.getElementById('countdownTime')) {{
                updateUnlimitedTimer();
                setInterval(updateUnlimitedTimer, 1000);
            }}
            '''
        elif irrigation_time_value > 0:
            remaining_seconds = irrigation_time_value * 60
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            countdown_section = f'''
            <div class="countdown-timer">
                <div class="countdown-time" id="countdownTime">{minutes}:{seconds:02d}</div>
                <div class="countdown-label">Time Remaining</div>
            </div>
            '''
            countdown_script = f'''
            let countdownSeconds = {int(remaining_seconds)};
            function updateCountdown() {{
                const minutes = Math.floor(countdownSeconds / 60);
                const seconds = countdownSeconds % 60;
                const countdownElement = document.getElementById('countdownTime');
                if (countdownElement) countdownElement.textContent = `${{minutes}}:${{seconds.toString().padStart(2, '0')}}`;
                if (countdownSeconds > 0) countdownSeconds--;
                else {{ if (countdownElement) countdownElement.textContent = "COMPLETE"; setTimeout(() => window.location.reload(), 2000); }}
            }}
            if (document.getElementById('countdownTime')) {{ updateCountdown(); setInterval(updateCountdown, 1000); }}
            '''

    # Prepare irrigation history for chart
    irrigation_events = []
    for event in list(irrigation_history)[-10:]:
        event_time = event.get('timestamp', '')
        if event_time:
            try:
                dt = datetime.fromisoformat(str(event_time).replace('Z', '').split('+')[0])
                irrigation_events.append({
                    'time': dt.strftime('%m/%d %H:%M'),
                    'volume': event.get('irrigation_volume_l_per_tree', 0),
                    'needed': event.get('irrigation_needed', False)
                })
            except:
                pass

    # Get daily ETo/ETc history from daily_prediction_result history (if available)
    daily_history = {
        'labels': [],
        'eto': [],
        'etc': [],
        'rn': []
    }
    # For now, use current values as placeholder - in production, this would come from stored daily data
    if daily_prediction_result.get("eto_prediction"):
        daily_history['labels'].append(datetime.now().strftime('%m/%d'))
        daily_history['eto'].append(round(daily_prediction_result["eto_prediction"].get("eto_mm_day", 0), 2))
        daily_history['etc'].append(round(latest_irrigation.get("etc_mm_day", 0) if latest_irrigation else 0, 2))
        daily_history['rn'].append(round(daily_prediction_result["eto_prediction"].get("estimated_rad", 0), 2))

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AquaSense | Smart Irrigation Dashboard</title>
        <meta http-equiv="refresh" content="30">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            :root {{
                --primary: #0077B6;
                --primary-dark: #005f8a;
                --primary-light: #E1F5FE;
                --accent: #00B4D8;
                --accent-light: #90E0EF;
                --success: #10B981;
                --success-light: #D1FAE5;
                --warning: #F59E0B;
                --warning-light: #FEF3C7;
                --danger: #EF4444;
                --danger-light: #FEE2E2;
                --gray-50: #F9FAFB;
                --gray-100: #F3F4F6;
                --gray-200: #E5E7EB;
                --gray-300: #D1D5DB;
                --gray-400: #9CA3AF;
                --gray-500: #6B7280;
                --gray-600: #4B5563;
                --gray-700: #374151;
                --gray-800: #1F2937;
                --gray-900: #111827;
                --white: #FFFFFF;
                --radius: 12px;
                --radius-lg: 16px;
                --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
                --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--gray-50);
                min-height: 100vh;
                color: var(--gray-800);
                line-height: 1.5;
            }}
            .page-wrapper {{ display: flex; min-height: 100vh; }}

            /* Sidebar */
            .sidebar {{
                width: 220px;
                background: var(--white);
                border-right: 1px solid var(--gray-200);
                padding: 20px 0;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
                z-index: 100;
            }}
            .sidebar-logo {{
                padding: 0 20px 20px;
                border-bottom: 1px solid var(--gray-100);
                margin-bottom: 16px;
            }}
            .logo-container {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .logo-icon {{
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.3rem;
            }}
            .logo-text {{
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--primary);
            }}
            .logo-subtitle {{
                font-size: 0.65rem;
                color: var(--gray-500);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .sidebar-nav {{ padding: 0 12px; }}
            .nav-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 14px;
                color: var(--gray-600);
                text-decoration: none;
                border-radius: var(--radius);
                margin-bottom: 4px;
                font-size: 0.875rem;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .nav-item:hover {{ background: var(--gray-100); color: var(--gray-800); }}
            .nav-item.active {{
                background: var(--primary-light);
                color: var(--primary);
                font-weight: 600;
            }}
            .nav-icon {{ font-size: 1rem; width: 20px; text-align: center; }}

            /* Farm Info Card in Sidebar */
            .farm-info {{
                margin: 20px 12px;
                padding: 14px;
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                border-radius: var(--radius);
                color: var(--white);
            }}
            .farm-name {{ font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; }}
            .farm-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
            .farm-stat {{ text-align: center; }}
            .farm-stat-value {{ font-size: 1rem; font-weight: 700; }}
            .farm-stat-label {{ font-size: 0.65rem; opacity: 0.85; }}

            /* Main Content */
            .main-content {{
                flex: 1;
                margin-left: 220px;
                padding: 20px 24px;
            }}

            /* Top Bar */
            .top-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                flex-wrap: wrap;
                gap: 12px;
            }}
            .page-title {{
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--gray-900);
            }}
            .top-bar-right {{
                display: flex;
                align-items: center;
                gap: 16px;
            }}
            .datetime-display {{
                text-align: right;
            }}
            .datetime-date {{ font-size: 0.85rem; font-weight: 600; color: var(--gray-800); }}
            .datetime-time {{ font-size: 0.75rem; color: var(--gray-500); }}

            /* Growth Stage Pill */
            .growth-pill {{
                display: flex;
                align-items: center;
                gap: 8px;
                background: var(--success-light);
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.75rem;
            }}
            .growth-day {{ font-weight: 700; color: var(--success); }}
            .growth-kc {{ background: var(--success); color: white; padding: 2px 8px; border-radius: 10px; font-weight: 600; }}

            /* Section Headers */
            .section-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 16px;
            }}
            .section-icon {{
                width: 32px;
                height: 32px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1rem;
            }}
            .section-icon.blue {{ background: var(--primary-light); }}
            .section-icon.green {{ background: var(--success-light); }}
            .section-icon.orange {{ background: var(--warning-light); }}
            .section-title {{ font-size: 1rem; font-weight: 600; color: var(--gray-800); }}

            /* VPD + Pump Control Section */
            .control-section {{
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 20px;
                margin-bottom: 24px;
            }}

            /* VPD Card */
            .vpd-card {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                overflow: hidden;
            }}
            .vpd-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .vpd-main-value {{ font-size: 2rem; font-weight: 700; }}
            .vpd-unit {{ font-size: 0.9rem; opacity: 0.9; }}
            .vpd-status-badge {{
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }}
            .vpd-body {{ padding: 16px 20px; }}
            .vpd-range {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }}
            .vpd-range-label {{ font-size: 0.8rem; color: var(--gray-500); }}
            .vpd-range-value {{ font-size: 0.85rem; font-weight: 600; color: var(--gray-700); }}
            .vpd-gauge {{
                height: 8px;
                background: var(--gray-200);
                border-radius: 4px;
                overflow: hidden;
                position: relative;
            }}
            .vpd-gauge-fill {{
                height: 100%;
                border-radius: 4px;
                transition: width 0.5s ease;
            }}
            .vpd-gauge-marker {{
                position: absolute;
                top: -3px;
                width: 3px;
                height: 14px;
                background: var(--gray-800);
                border-radius: 2px;
            }}

            /* Pump Control Card */
            .pump-card {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                overflow: hidden;
            }}
            .pump-header {{
                padding: 16px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--gray-100);
            }}
            .pump-status {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .pump-status-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }}
            .pump-status-dot.on {{ background: var(--success); }}
            .pump-status-dot.off {{ background: var(--danger); }}
            .pump-status-dot.needed {{ background: var(--warning); animation: blink 1s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
            @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
            .pump-status-text {{ font-weight: 600; font-size: 0.95rem; }}
            .pump-status-text.on {{ color: var(--success); }}
            .pump-status-text.off {{ color: var(--gray-600); }}
            .pump-status-text.needed {{ color: var(--warning); }}

            .pump-body {{ padding: 16px 20px; }}
            .pump-metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-bottom: 16px;
            }}
            .pump-metric {{
                text-align: center;
                padding: 12px;
                background: var(--gray-50);
                border-radius: var(--radius);
            }}
            .pump-metric-value {{
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--primary);
            }}
            .pump-metric-label {{
                font-size: 0.7rem;
                color: var(--gray-500);
                text-transform: uppercase;
                margin-top: 2px;
            }}

            .pump-controls {{
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }}
            .pump-btn {{
                flex: 1;
                min-width: 80px;
                padding: 10px 16px;
                border: none;
                border-radius: var(--radius);
                font-size: 0.8rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                font-family: inherit;
            }}
            .pump-btn-stop {{
                background: var(--danger);
                color: white;
            }}
            .pump-btn-stop:hover {{ background: #DC2626; }}
            .pump-btn-start {{
                background: var(--success);
                color: white;
            }}
            .pump-btn-start:hover {{ background: #059669; }}
            .pump-btn-timed {{
                background: var(--primary);
                color: white;
            }}
            .pump-btn-timed:hover {{ background: var(--primary-dark); }}

            /* Water Level Gauge */
            .water-gauge {{
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px;
                background: var(--gray-50);
                border-radius: var(--radius);
            }}
            .water-tank {{
                width: 40px;
                height: 60px;
                border: 2px solid var(--gray-400);
                border-radius: 4px;
                position: relative;
                overflow: hidden;
                background: var(--gray-100);
            }}
            .water-fill {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                transition: height 0.8s ease;
            }}
            .water-info {{ flex: 1; }}
            .water-percent {{ font-size: 1.2rem; font-weight: 700; }}
            .water-detail {{ font-size: 0.75rem; color: var(--gray-500); }}

            /* Countdown Timer */
            .countdown {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
                color: white;
                padding: 16px;
                border-radius: var(--radius);
                text-align: center;
                margin-top: 12px;
            }}
            .countdown-time {{ font-size: 2rem; font-weight: 700; }}
            .countdown-label {{ font-size: 0.8rem; opacity: 0.9; }}

            /* Sensor Status Section */
            .sensors-section {{ margin-bottom: 24px; }}
            .sensor-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
            }}
            .sensor-card {{
                background: var(--white);
                border-radius: var(--radius);
                border: 1px solid var(--gray-200);
                padding: 16px;
                transition: all 0.2s;
            }}
            .sensor-card:hover {{
                box-shadow: var(--shadow-md);
                transform: translateY(-2px);
            }}
            .sensor-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 10px;
            }}
            .sensor-icon {{
                width: 36px;
                height: 36px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1rem;
            }}
            .sensor-status {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
            }}
            .sensor-status.online {{ background: var(--success); }}
            .sensor-status.offline {{ background: var(--gray-400); }}
            .sensor-name {{ font-size: 0.75rem; color: var(--gray-500); margin-bottom: 4px; }}
            .sensor-value {{ font-size: 1.5rem; font-weight: 700; color: var(--gray-800); }}
            .sensor-unit {{ font-size: 0.8rem; color: var(--gray-500); font-weight: 400; }}
            .sensor-bar {{
                height: 4px;
                background: var(--gray-200);
                border-radius: 2px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .sensor-bar-fill {{
                height: 100%;
                border-radius: 2px;
                transition: width 0.5s ease;
            }}

            /* Charts Section */
            .charts-section {{ margin-bottom: 24px; }}
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
            }}
            .chart-card {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                padding: 16px;
            }}
            .chart-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 12px;
            }}
            .chart-title {{ font-size: 0.9rem; font-weight: 600; color: var(--gray-700); }}
            .chart-badge {{
                font-size: 0.65rem;
                padding: 2px 8px;
                border-radius: 10px;
                background: var(--gray-100);
                color: var(--gray-600);
            }}
            .chart-container {{ height: 160px; }}

            /* Daily Trends Section */
            .daily-section {{ margin-bottom: 24px; }}
            .daily-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
            }}
            .daily-card {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                padding: 20px;
                text-align: center;
            }}
            .daily-icon {{
                width: 48px;
                height: 48px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 12px;
                font-size: 1.5rem;
            }}
            .daily-value {{ font-size: 2rem; font-weight: 700; margin-bottom: 4px; }}
            .daily-label {{ font-size: 0.8rem; color: var(--gray-500); }}
            .daily-chart {{ height: 80px; margin-top: 12px; }}

            /* Irrigation History Section */
            .irrigation-section {{ margin-bottom: 24px; }}
            .irrigation-chart-card {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                padding: 20px;
            }}

            /* Responsive */
            @media (max-width: 1024px) {{
                .control-section {{ grid-template-columns: 1fr; }}
                .daily-grid {{ grid-template-columns: 1fr; }}
            }}
            @media (max-width: 768px) {{
                .sidebar {{ display: none; }}
                .main-content {{ margin-left: 0; padding: 16px; }}
                .pump-metrics {{ grid-template-columns: 1fr; }}
                .sensor-grid {{ grid-template-columns: repeat(2, 1fr); }}
            }}
            @media (max-width: 480px) {{
                .sensor-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="page-wrapper">
            <!-- Sidebar -->
            <aside class="sidebar">
                <div class="sidebar-logo">
                    <div class="logo-container">
                        <div class="logo-icon">💧</div>
                        <div>
                            <div class="logo-text">AquaSense</div>
                            <div class="logo-subtitle">Smart Irrigation</div>
                        </div>
                    </div>
                </div>
                <nav class="sidebar-nav">
                    <a href="/dashboard" class="nav-item active">
                        <span class="nav-icon">📊</span> Dashboard
                    </a>
                    <a href="/config" class="nav-item">
                        <span class="nav-icon">⚙️</span> Settings
                    </a>
                    <a href="/raw-data" class="nav-item">
                        <span class="nav-icon">📈</span> Raw Data
                    </a>
                    <a href="/docs" class="nav-item">
                        <span class="nav-icon">📄</span> API Docs
                    </a>
                </nav>
                <div class="farm-info">
                    <div class="farm-name">{irrigation_config.get('farm_name', 'My Farm')}</div>
                    <div class="farm-stats">
                        <div class="farm-stat">
                            <div class="farm-stat-value">Day {current_day}</div>
                            <div class="farm-stat-label">Growth</div>
                        </div>
                        <div class="farm-stat">
                            <div class="farm-stat-value">{current_kc:.2f}</div>
                            <div class="farm-stat-label">Kc Value</div>
                        </div>
                    </div>
                </div>
            </aside>

            <!-- Main Content -->
            <main class="main-content">
                <!-- Top Bar -->
                <div class="top-bar">
                    <h1 class="page-title">Dashboard</h1>
                    <div class="top-bar-right">
                        <div class="growth-pill">
                            <span class="growth-day">Day {current_day}</span>
                            <span style="color: var(--gray-600);">{current_stage[:20]}...</span>
                            <span class="growth-kc">Kc {current_kc:.2f}</span>
                        </div>
                        <div class="datetime-display">
                            <div class="datetime-date">{datetime.now().strftime("%A, %d %B %Y")}</div>
                            <div class="datetime-time">{datetime.now().strftime("%H:%M")} • Auto-refresh 30s</div>
                        </div>
                    </div>
                </div>

                <!-- VPD + Pump Control Section -->
                <div class="control-section">
                    <!-- VPD Card -->
                    <div class="vpd-card">
                        <div class="vpd-header">
                            <div>
                                <div class="vpd-main-value">{current_vpd:.2f} <span class="vpd-unit">kPa</span></div>
                                <div style="font-size: 0.8rem; opacity: 0.9;">Vapor Pressure Deficit</div>
                            </div>
                            <div class="vpd-status-badge" style="background: {vpd_status.get('color', '#999')};">{vpd_status.get('status', 'N/A').upper()}</div>
                        </div>
                        <div class="vpd-body">
                            <div class="vpd-range">
                                <span class="vpd-range-label">Optimal Range</span>
                                <span class="vpd-range-value">{vpd_range.get('min', 0)} - {vpd_range.get('max', 2)} kPa</span>
                            </div>
                            <div class="vpd-gauge">
                                <div class="vpd-gauge-fill" style="width: {min(current_vpd / 3 * 100, 100)}%; background: {vpd_status.get('color', '#999')};"></div>
                            </div>
                            <div style="margin-top: 12px; font-size: 0.8rem; color: var(--gray-600);">
                                {'⚠️ Consider irrigation' if vpd_status.get('status') == 'high' else '✅ VPD within acceptable range' if vpd_status.get('status') in ['optimal', 'normal'] else '❄️ Low VPD detected'}
                            </div>
                        </div>
                    </div>

                    <!-- Pump Control Card -->
                    <div class="pump-card">
                        <div class="pump-header">
                            <div class="pump-status">
                                <div class="pump-status-dot {'on' if pump_is_effectively_on else 'needed' if irrigation_needed else 'off'}"></div>
                                <span class="pump-status-text {'on' if pump_is_effectively_on else 'needed' if irrigation_needed else 'off'}">
                                    {'PUMP RUNNING' if pump_is_effectively_on else 'IRRIGATION NEEDED' if irrigation_needed else 'PUMP OFF'}
                                </span>
                            </div>
                            <div style="font-size: 0.75rem; color: var(--gray-500);">
                                {irrigation_config.get('irrigation_type', 'Sprinkler')} System
                            </div>
                        </div>
                        <div class="pump-body">
                            <div class="pump-metrics">
                                <div class="pump-metric">
                                    <div class="pump-metric-value">{irrigation_volume:.1f}</div>
                                    <div class="pump-metric-label">L/Tree</div>
                                </div>
                                <div class="pump-metric">
                                    <div class="pump-metric-value">{irrigation_time_minutes:.0f}</div>
                                    <div class="pump-metric-label">Minutes</div>
                                </div>
                                <div class="water-gauge">
                                    <div class="water-tank">
                                        <div class="water-fill" style="height: {water_level_percent}%; background: {water_fill_color};"></div>
                                    </div>
                                    <div class="water-info">
                                        <div class="water-percent" style="color: {water_fill_color};">{water_level_percent:.0f}%</div>
                                        <div class="water-detail">Soil Water</div>
                                        <div class="water-detail">Deficit: {deficit_mm:.1f}mm</div>
                                    </div>
                                </div>
                            </div>
                            <div class="pump-controls">
                                <button class="pump-btn pump-btn-stop" onclick="controlPump('close')">⏹ Stop</button>
                                <button class="pump-btn pump-btn-start" onclick="controlPump('open', null)">▶ Start</button>
                                <button class="pump-btn pump-btn-timed" onclick="controlPump('open', 3600)">60min</button>
                                <button class="pump-btn pump-btn-timed" onclick="controlPump('open', 7200)">120min</button>
                            </div>
                            {f'<div class="countdown"><div class="countdown-time" id="countdownTime">--:--</div><div class="countdown-label">Time Remaining</div></div>' if pump_is_effectively_on else ''}
                        </div>
                    </div>
                </div>

                <!-- Sensor Readings Section -->
                <div class="sensors-section">
                    <div class="section-header">
                        <div class="section-icon blue">📡</div>
                        <span class="section-title">Live Sensor Readings</span>
                        <span style="margin-left: auto; font-size: 0.75rem; color: var(--gray-500);">15-min aggregates from ESP32</span>
                    </div>
                    <div class="sensor-grid">
                        <div class="sensor-card">
                            <div class="sensor-header">
                                <div class="sensor-icon" style="background: var(--primary-light);">🌡️</div>
                                <div class="sensor-status {temp_status_class}"></div>
                            </div>
                            <div class="sensor-name">Temperature</div>
                            <div class="sensor-value">{current_temp_max:.1f}<span class="sensor-unit">°C</span></div>
                            <div style="font-size: 0.7rem; color: var(--gray-500);">Min: {current_temp_min:.1f}°C</div>
                            <div class="sensor-bar"><div class="sensor-bar-fill" style="width: {min(current_temp_max/50*100, 100)}%; background: var(--primary);"></div></div>
                        </div>
                        <div class="sensor-card">
                            <div class="sensor-header">
                                <div class="sensor-icon" style="background: #E3F2FD;">💧</div>
                                <div class="sensor-status {humidity_status_class}"></div>
                            </div>
                            <div class="sensor-name">Humidity</div>
                            <div class="sensor-value">{current_humidity:.0f}<span class="sensor-unit">%</span></div>
                            <div class="sensor-bar"><div class="sensor-bar-fill" style="width: {current_humidity}%; background: #42a5f5;"></div></div>
                        </div>
                        <div class="sensor-card">
                            <div class="sensor-header">
                                <div class="sensor-icon" style="background: #E8F5E9;">🌬️</div>
                                <div class="sensor-status {wind_status_class}"></div>
                            </div>
                            <div class="sensor-name">Wind Speed</div>
                            <div class="sensor-value">{current_wind:.2f}<span class="sensor-unit">m/s</span></div>
                            <div class="sensor-bar"><div class="sensor-bar-fill" style="width: {min(current_wind*10, 100)}%; background: #66bb6a;"></div></div>
                        </div>
                        <div class="sensor-card">
                            <div class="sensor-header">
                                <div class="sensor-icon" style="background: #FFF8E1;">☀️</div>
                                <div class="sensor-status {solar_status_class}"></div>
                            </div>
                            <div class="sensor-name">Solar Radiation</div>
                            <div class="sensor-value">{current_solar_wm2:.0f}<span class="sensor-unit">W/m²</span></div>
                            <div class="sensor-bar"><div class="sensor-bar-fill" style="width: {min(current_solar_wm2/10, 100)}%; background: #ffa726;"></div></div>
                        </div>
                        <div class="sensor-card">
                            <div class="sensor-header">
                                <div class="sensor-icon" style="background: #E1F5FE;">🌧️</div>
                                <div class="sensor-status {rain_status_class}"></div>
                            </div>
                            <div class="sensor-name">Rainfall</div>
                            <div class="sensor-value">{current_rainfall:.1f}<span class="sensor-unit">mm</span></div>
                            <div class="sensor-bar"><div class="sensor-bar-fill" style="width: {min(current_rainfall*5, 100)}%; background: #2196f3;"></div></div>
                        </div>
                        <div class="sensor-card">
                            <div class="sensor-header">
                                <div class="sensor-icon" style="background: #F3E5F5;">📊</div>
                                <div class="sensor-status {pressure_status_class}"></div>
                            </div>
                            <div class="sensor-name">Pressure</div>
                            <div class="sensor-value">{current_pressure:.0f}<span class="sensor-unit">hPa</span></div>
                            <div class="sensor-bar"><div class="sensor-bar-fill" style="width: {min((current_pressure-950)/100*100, 100) if current_pressure else 0}%; background: #9c27b0;"></div></div>
                        </div>
                    </div>
                </div>

                <!-- Daily ET Values Section -->
                <div class="daily-section">
                    <div class="section-header">
                        <div class="section-icon green">🌿</div>
                        <span class="section-title">Daily Evapotranspiration</span>
                        <span style="margin-left: auto; font-size: 0.75rem; color: var(--gray-500);">Updated at 6:00 AM</span>
                    </div>
                    <div class="daily-grid">
                        <div class="daily-card">
                            <div class="daily-icon" style="background: var(--primary-light);">💨</div>
                            <div class="daily-value" style="color: var(--primary);">{current_eto:.2f}</div>
                            <div class="daily-label">ETo (mm/day)</div>
                            <div style="font-size: 0.7rem; color: var(--gray-400); margin-top: 4px;">Reference ET</div>
                        </div>
                        <div class="daily-card">
                            <div class="daily-icon" style="background: var(--success-light);">🌱</div>
                            <div class="daily-value" style="color: var(--success);">{current_etc:.2f}</div>
                            <div class="daily-label">ETc (mm/day)</div>
                            <div style="font-size: 0.7rem; color: var(--gray-400); margin-top: 4px;">Crop ET (Kc × ETo)</div>
                        </div>
                        <div class="daily-card">
                            <div class="daily-icon" style="background: var(--warning-light);">☀️</div>
                            <div class="daily-value" style="color: var(--warning);">{current_rn_est:.2f}</div>
                            <div class="daily-label">Rn_est (MJ/m²/day)</div>
                            <div style="font-size: 0.7rem; color: var(--gray-400); margin-top: 4px;">Net Radiation</div>
                        </div>
                    </div>
                </div>

                <!-- Sensor Trends Section -->
                <div class="charts-section">
                    <div class="section-header">
                        <div class="section-icon blue">📈</div>
                        <span class="section-title">15-Minute Trends</span>
                    </div>
                    <div class="charts-grid">
                        <div class="chart-card">
                            <div class="chart-header">
                                <span class="chart-title">🌡️ Temperature</span>
                                <span class="chart-badge">°C</span>
                            </div>
                            <div class="chart-container"><canvas id="temperatureChart"></canvas></div>
                        </div>
                        <div class="chart-card">
                            <div class="chart-header">
                                <span class="chart-title">💧 Humidity</span>
                                <span class="chart-badge">%</span>
                            </div>
                            <div class="chart-container"><canvas id="humidityChart"></canvas></div>
                        </div>
                        <div class="chart-card">
                            <div class="chart-header">
                                <span class="chart-title">🌬️ Wind Speed</span>
                                <span class="chart-badge">m/s</span>
                            </div>
                            <div class="chart-container"><canvas id="windChart"></canvas></div>
                        </div>
                        <div class="chart-card">
                            <div class="chart-header">
                                <span class="chart-title">☀️ Solar Radiation</span>
                                <span class="chart-badge">W/m²</span>
                            </div>
                            <div class="chart-container"><canvas id="radiationChart"></canvas></div>
                        </div>
                        <div class="chart-card">
                            <div class="chart-header">
                                <span class="chart-title">🌧️ Rainfall</span>
                                <span class="chart-badge">mm</span>
                            </div>
                            <div class="chart-container"><canvas id="rainfallChart"></canvas></div>
                        </div>
                        <div class="chart-card">
                            <div class="chart-header">
                                <span class="chart-title">📊 Pressure</span>
                                <span class="chart-badge">hPa</span>
                            </div>
                            <div class="chart-container"><canvas id="pressureChart"></canvas></div>
                        </div>
                    </div>
                </div>

                <!-- Irrigation History Section -->
                <div class="irrigation-section">
                    <div class="section-header">
                        <div class="section-icon orange">💦</div>
                        <span class="section-title">Irrigation History</span>
                        <span style="margin-left: auto; font-size: 0.75rem; color: var(--gray-500);">Last 10 events</span>
                    </div>
                    <div class="irrigation-chart-card">
                        <div class="chart-container" style="height: 200px;"><canvas id="irrigationHistoryChart"></canvas></div>
                    </div>
                </div>
            </main>
        </div>

        <script>
            {countdown_script}

            async function controlPump(action, duration = null) {{
                try {{
                    const payload = {{ action: action }};
                    if (duration !== null) payload.duration = duration;
                    const response = await fetch('/pump/control', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
                    const result = await response.json();
                    if (response.ok) {{
                        if (result.summary) {{ alert(`Pump Summary:\\nRuntime: ${{result.summary.runtime_minutes}} min\\nWater: ${{result.summary.water_discharged_liters}} L`); }}
                        else {{ alert(result.message); }}
                        setTimeout(() => window.location.reload(), 1000);
                    }} else {{ alert('Error: ' + result.detail); }}
                }} catch (error) {{ alert('Connection error: ' + error.message); }}
            }}

            const chartConfig = {{
                type: 'line',
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ display: true, ticks: {{ color: '#9CA3AF', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }},
                        y: {{ display: true, ticks: {{ color: '#9CA3AF', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}
                    }},
                    elements: {{ point: {{ radius: 2, hoverRadius: 4 }}, line: {{ borderWidth: 2, tension: 0.4 }} }}
                }}
            }};

            new Chart(document.getElementById('humidityChart').getContext('2d'), {{
                ...chartConfig,
                data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['humidity']}, borderColor: '#0077B6', backgroundColor: 'rgba(0, 119, 182, 0.1)', fill: true }}] }}
            }});
            new Chart(document.getElementById('windChart').getContext('2d'), {{
                ...chartConfig,
                data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['wind_speed']}, borderColor: '#10B981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true }}] }}
            }});
            new Chart(document.getElementById('radiationChart').getContext('2d'), {{
                ...chartConfig,
                data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['solar_wm2']}, borderColor: '#F59E0B', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: true }}] }}
            }});
            new Chart(document.getElementById('rainfallChart').getContext('2d'), {{
                ...chartConfig,
                data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['rainfall']}, borderColor: '#3B82F6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true }}] }}
            }});
            new Chart(document.getElementById('temperatureChart').getContext('2d'), {{
                ...chartConfig,
                data: {{
                    labels: {history_data['labels']},
                    datasets: [
                        {{ label: 'Max', data: {history_data['temp_max']}, borderColor: '#EF4444', backgroundColor: 'transparent', fill: false }},
                        {{ label: 'Min', data: {history_data['temp_min']}, borderColor: '#3B82F6', backgroundColor: 'transparent', fill: false }}
                    ]
                }},
                options: {{ ...chartConfig.options, plugins: {{ legend: {{ display: true, labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }} }} }}
            }});
            new Chart(document.getElementById('pressureChart').getContext('2d'), {{
                ...chartConfig,
                data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['pressure']}, borderColor: '#8B5CF6', backgroundColor: 'rgba(139, 92, 246, 0.1)', fill: true }}] }}
            }});

            // Irrigation History Chart
            const irrigationEvents = {json.dumps([e['time'] for e in irrigation_events])};
            const irrigationVolumes = {json.dumps([e['volume'] for e in irrigation_events])};
            const irrigationNeeded = {json.dumps([e['needed'] for e in irrigation_events])};

            new Chart(document.getElementById('irrigationHistoryChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: irrigationEvents,
                    datasets: [{{
                        label: 'Irrigation Volume (L/Tree)',
                        data: irrigationVolumes,
                        backgroundColor: irrigationNeeded.map(n => n ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)'),
                        borderColor: irrigationNeeded.map(n => n ? '#EF4444' : '#10B981'),
                        borderWidth: 1,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const needed = irrigationNeeded[context.dataIndex];
                                    return `${{context.parsed.y.toFixed(1)}} L/Tree (${{needed ? 'Needed' : 'Adequate'}})`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{ display: true, ticks: {{ color: '#9CA3AF', font: {{ size: 9 }}, maxRotation: 45 }}, grid: {{ display: false }} }},
                        y: {{ display: true, ticks: {{ color: '#9CA3AF', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,0,0,0.05)' }}, title: {{ display: true, text: 'L/Tree', font: {{ size: 10 }}, color: '#9CA3AF' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

# =========================
# Configuration Page
# =========================
@app.get("/config", response_class=HTMLResponse)
async def configuration_page():
    current_config = irrigation_config
    current_day = growth_stage_config["current_day"]
    current_stage = current_config.get('crop_growth_stage', 'Not Set')
    # Get short stage name
    stage_short = current_stage.split('(')[0].strip() if '(' in current_stage else current_stage[:25]

    def is_selected(field, value):
        return "selected" if str(current_config.get(field)) == str(value) else ""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Farm Settings | Smart Irrigation</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --netafim-green: #00A651;
                --netafim-green-dark: #008C45;
                --netafim-green-light: #E8F5E9;
                --netafim-blue: #0077B6;
                --netafim-blue-light: #E3F2FD;
                --earth-brown: #8B5A2B;
                --water-blue: #03A9F4;
                --water-blue-light: #E1F5FE;
                --sun-yellow: #FFC107;
                --sun-yellow-light: #FFF8E1;
                --gray-50: #FAFAFA;
                --gray-100: #F5F5F5;
                --gray-200: #EEEEEE;
                --gray-300: #E0E0E0;
                --gray-400: #BDBDBD;
                --gray-500: #9E9E9E;
                --gray-600: #757575;
                --gray-700: #616161;
                --gray-800: #424242;
                --gray-900: #212121;
                --white: #FFFFFF;
                --radius: 8px;
                --radius-lg: 12px;
                --radius-xl: 16px;
                --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
                --shadow: 0 2px 4px rgba(0,0,0,0.08);
                --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
                --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--gray-50);
                min-height: 100vh;
                color: var(--gray-800);
                line-height: 1.6;
            }}
            .page-wrapper {{ display: flex; min-height: 100vh; }}

            /* Sidebar */
            .sidebar {{
                width: 240px;
                background: var(--white);
                border-right: 1px solid var(--gray-200);
                padding: 24px 0;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
            }}
            .sidebar-logo {{
                padding: 0 24px 24px;
                border-bottom: 1px solid var(--gray-100);
                margin-bottom: 16px;
            }}
            .logo-container {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .logo-icon {{
                width: 36px;
                height: 36px;
                background: linear-gradient(135deg, var(--netafim-green) 0%, var(--netafim-green-dark) 100%);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
            }}
            .logo-text {{
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--netafim-green);
            }}
            .logo-subtitle {{
                font-size: 0.6rem;
                color: var(--gray-500);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .sidebar-nav {{ padding: 0 12px; }}
            .nav-item {{
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                color: var(--gray-600);
                text-decoration: none;
                border-radius: var(--radius);
                margin-bottom: 4px;
                font-size: 0.9rem;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .nav-item:hover {{ background: var(--gray-100); color: var(--gray-800); }}
            .nav-item.active {{
                background: var(--netafim-green-light);
                color: var(--netafim-green);
            }}
            .nav-icon {{ font-size: 1.1rem; width: 24px; text-align: center; }}

            /* Main Content */
            .main-content {{
                flex: 1;
                margin-left: 240px;
                padding: 24px 32px;
                max-width: 1000px;
            }}

            /* Page Header */
            .page-header {{
                margin-bottom: 24px;
            }}
            .page-header h1 {{
                font-size: 1.75rem;
                font-weight: 700;
                color: var(--gray-900);
                margin-bottom: 4px;
            }}
            .page-header p {{
                color: var(--gray-500);
                font-size: 0.95rem;
            }}

            /* Status Banner */
            .status-banner {{
                background: linear-gradient(135deg, var(--netafim-green) 0%, var(--netafim-green-dark) 100%);
                border-radius: var(--radius-xl);
                padding: 20px 24px;
                margin-bottom: 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: var(--white);
                box-shadow: var(--shadow-md);
            }}
            .status-banner-left h3 {{
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 2px;
            }}
            .status-banner-left p {{
                font-size: 0.85rem;
                opacity: 0.9;
            }}
            .status-pills {{
                display: flex;
                gap: 12px;
            }}
            .status-pill {{
                background: rgba(255,255,255,0.2);
                backdrop-filter: blur(8px);
                padding: 8px 16px;
                border-radius: 20px;
                text-align: center;
                min-width: 90px;
            }}
            .status-pill .label {{
                font-size: 0.65rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                opacity: 0.9;
                margin-bottom: 2px;
            }}
            .status-pill .value {{
                font-size: 0.95rem;
                font-weight: 600;
            }}

            /* Card */
            .card {{
                background: var(--white);
                border-radius: var(--radius-lg);
                border: 1px solid var(--gray-200);
                margin-bottom: 20px;
                overflow: hidden;
            }}
            .card-header {{
                padding: 16px 20px;
                border-bottom: 1px solid var(--gray-100);
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .card-icon {{
                width: 36px;
                height: 36px;
                border-radius: var(--radius);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
            }}
            .card-icon.green {{ background: var(--netafim-green-light); }}
            .card-icon.blue {{ background: var(--water-blue-light); }}
            .card-icon.yellow {{ background: var(--sun-yellow-light); }}
            .card-title {{
                font-size: 0.95rem;
                font-weight: 600;
                color: var(--gray-800);
            }}
            .card-body {{
                padding: 20px;
            }}

            /* Form Grid */
            .form-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
            }}
            .form-grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
            .form-grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
            .span-2 {{ grid-column: span 2; }}

            /* Form Group */
            .form-group {{ margin-bottom: 0; }}
            .form-group label {{
                display: block;
                font-size: 0.8rem;
                font-weight: 500;
                color: var(--gray-600);
                margin-bottom: 6px;
            }}
            .form-group select, .form-group input[type="text"], .form-group input[type="number"] {{
                width: 100%;
                padding: 10px 14px;
                font-size: 0.9rem;
                border: 1px solid var(--gray-300);
                border-radius: var(--radius);
                background: var(--white);
                color: var(--gray-800);
                transition: all 0.2s ease;
                font-family: inherit;
            }}
            .form-group select:hover, .form-group input:hover {{
                border-color: var(--gray-400);
            }}
            .form-group select:focus, .form-group input:focus {{
                outline: none;
                border-color: var(--netafim-green);
                box-shadow: 0 0 0 3px rgba(0, 166, 81, 0.1);
            }}
            .form-group small {{
                display: block;
                font-size: 0.75rem;
                color: var(--gray-400);
                margin-top: 4px;
            }}

            /* Metric Display */
            .metric-display {{
                background: var(--gray-50);
                border: 1px solid var(--gray-200);
                border-radius: var(--radius);
                padding: 14px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 1.75rem;
                font-weight: 700;
                color: var(--netafim-green);
                line-height: 1.2;
            }}
            .metric-value.blue {{ color: var(--water-blue); }}
            .metric-value.yellow {{ color: #D97706; }}  /* Darker amber for better visibility */
            .metric-label {{
                font-size: 0.7rem;
                color: var(--gray-500);
                text-transform: uppercase;
                letter-spacing: 0.3px;
                margin-top: 2px;
            }}

            /* Info Alert */
            .info-alert {{
                background: var(--netafim-green-light);
                border: 1px solid rgba(0, 166, 81, 0.2);
                border-radius: var(--radius);
                padding: 12px 16px;
                font-size: 0.85rem;
                color: var(--netafim-green-dark);
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .info-alert.blue {{
                background: var(--water-blue-light);
                border-color: rgba(3, 169, 244, 0.2);
                color: var(--netafim-blue);
            }}

            /* Action Bar */
            .action-bar {{
                display: flex;
                justify-content: flex-end;
                gap: 12px;
                padding: 16px 20px;
                background: var(--gray-50);
                border-top: 1px solid var(--gray-100);
            }}
            .btn {{
                padding: 10px 20px;
                font-size: 0.875rem;
                border: none;
                border-radius: var(--radius);
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s ease;
                font-family: inherit;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }}
            .btn-primary {{
                background: var(--netafim-green);
                color: var(--white);
            }}
            .btn-primary:hover {{
                background: var(--netafim-green-dark);
                transform: translateY(-1px);
                box-shadow: var(--shadow-md);
            }}
            .btn-secondary {{
                background: var(--white);
                color: var(--gray-700);
                border: 1px solid var(--gray-300);
            }}
            .btn-secondary:hover {{ background: var(--gray-50); }}

            /* Message */
            .message {{
                margin: 0 0 16px 0;
                padding: 12px 16px;
                border-radius: var(--radius);
                font-size: 0.875rem;
                font-weight: 500;
            }}
            .message.success {{ background: var(--netafim-green-light); color: var(--netafim-green-dark); }}
            .message.error {{ background: #fee2e2; color: #991b1b; }}

            /* Responsive */
            @media (max-width: 900px) {{
                .sidebar {{ display: none; }}
                .main-content {{ margin-left: 0; padding: 16px; }}
                .mobile-nav {{
                    display: flex;
                    justify-content: center;
                    gap: 8px;
                    margin-bottom: 20px;
                    background: var(--white);
                    padding: 8px;
                    border-radius: var(--radius-lg);
                    box-shadow: var(--shadow);
                }}
                .status-pills {{ flex-wrap: wrap; }}
            }}
            @media (min-width: 901px) {{
                .mobile-nav {{ display: none; }}
            }}
            @media (max-width: 600px) {{
                .form-grid {{ grid-template-columns: 1fr; }}
                .form-grid-3, .form-grid-4 {{ grid-template-columns: 1fr 1fr; }}
                .status-banner {{ flex-direction: column; gap: 16px; align-items: flex-start; }}
                .status-pills {{ width: 100%; justify-content: space-between; }}
            }}
        </style>
    </head>
    <body>
        <div class="page-wrapper">
            <!-- Sidebar Navigation -->
            <aside class="sidebar">
                <div class="sidebar-logo">
                    <div class="logo-container">
                        <div class="logo-icon">🌿</div>
                        <div>
                            <div class="logo-text">AquaSense</div>
                            <div class="logo-subtitle">Farm Settings</div>
                        </div>
                    </div>
                </div>
                <nav class="sidebar-nav">
                    <a href="/dashboard" class="nav-item">
                        <span class="nav-icon">📊</span> Dashboard
                    </a>
                    <a href="/config" class="nav-item active">
                        <span class="nav-icon">⚙️</span> Settings
                    </a>
                    <a href="/raw-data" class="nav-item">
                        <span class="nav-icon">📈</span> Raw Data
                    </a>
                    <a href="/docs" class="nav-item">
                        <span class="nav-icon">📄</span> API Docs
                    </a>
                </nav>
            </aside>

            <!-- Main Content -->
            <main class="main-content">
                <!-- Mobile Navigation -->
                <nav class="mobile-nav">
                    <a href="/dashboard" class="nav-item">Dashboard</a>
                    <a href="/config" class="nav-item active">Settings</a>
                    <a href="/raw-data" class="nav-item">Data</a>
                    <a href="/docs" class="nav-item">API</a>
                </nav>

                <div class="page-header">
                    <h1>Farm Settings</h1>
                    <p>Configure your irrigation system parameters</p>
                </div>

                <!-- Status Banner -->
                <div class="status-banner">
                    <div class="status-banner-left">
                        <h3>{current_config.get('farm_name', 'My Farm')}</h3>
                        <p>{current_config.get('crop_type', 'Crop')} • {stage_short}</p>
                    </div>
                    <div class="status-pills">
                        <div class="status-pill">
                            <div class="label">Day</div>
                            <div class="value">{current_day}</div>
                        </div>
                        <div class="status-pill">
                            <div class="label">Maturity</div>
                            <div class="value">{current_config.get('plant_maturity', 'Mature')}</div>
                        </div>
                        <div class="status-pill">
                            <div class="label">Wetted</div>
                            <div class="value">{current_config.get('wetted_pct', 50.0)}%</div>
                        </div>
                    </div>
                </div>

                <div id="message"></div>

                <form id="configForm">
                    <!-- Farm & Crop Card -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon green">🏡</div>
                            <span class="card-title">Farm & Crop</span>
                        </div>
                        <div class="card-body">
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="farm_name">Farm Name</label>
                                    <input type="text" id="farm_name" value="{current_config.get('farm_name', '')}" placeholder="Enter farm name" required>
                                </div>
                                <div class="form-group">
                                    <label for="crop_type">Crop Type</label>
                                    <select id="crop_type" required>
                                        <option value="">Select crop</option>
                                        <option value="Durian" {is_selected('crop_type', 'Durian')}>Durian</option>
                                        <option value="Mangosteen" {is_selected('crop_type', 'Mangosteen')}>Mangosteen</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Growth Stage Card -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon green">🌱</div>
                            <span class="card-title">Growth Stage</span>
                        </div>
                        <div class="card-body">
                            <div class="form-grid" style="margin-bottom: 16px;">
                                <div class="form-group span-2">
                                    <label for="growth_stage">Growth Stage</label>
                                    <select id="growth_stage" onchange="updateDayRange()">
                                        <option value="1-45" data-min="1" data-max="45" data-kc="0.5" data-mad="0.40" {is_selected('crop_growth_stage', 'Flowering & fruit setting (0-45 Days)')}>Flowering & Fruit Setting (Days 1-45)</option>
                                        <option value="46-75" data-min="46" data-max="75" data-kc="0.6" data-mad="0.45" {is_selected('crop_growth_stage', 'Early fruit growth (46-75 days)')}>Early Fruit Growth (Days 46-75)</option>
                                        <option value="76-110" data-min="76" data-max="110" data-kc="0.85" data-mad="0.55" {is_selected('crop_growth_stage', 'Fruit growth (76-110 days)')}>Fruit Growth (Days 76-110)</option>
                                        <option value="111-140" data-min="111" data-max="140" data-kc="0.75" data-mad="0.60" {is_selected('crop_growth_stage', 'Fruit Maturity (111-140 days)')}>Fruit Maturity (Days 111-140)</option>
                                        <option value="141-290" data-min="141" data-max="290" data-kc="0.6" data-mad="0.70" {is_selected('crop_growth_stage', 'Vegetative Growth (141-290 days)')}>Vegetative Growth (Days 141-290)</option>
                                        <option value="291-320" data-min="291" data-max="320" data-kc="0.4" data-mad="0.50" {is_selected('crop_growth_stage', 'Floral initiation (291-320 days)')}>Floral Initiation (Days 291-320)</option>
                                        <option value="321-365" data-min="321" data-max="365" data-kc="0.4" data-mad="0.45" {is_selected('crop_growth_stage', 'Floral development (321-365 days)')}>Floral Development (Days 321-365)</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-grid form-grid-4">
                                <div class="form-group">
                                    <label for="growth_day">Current Day</label>
                                    <input type="number" id="growth_day" value="{current_day}" min="1" max="365" required>
                                    <small id="day_range_hint">Range: 1-365</small>
                                </div>
                                <div class="form-group">
                                    <label>Kc Value</label>
                                    <div class="metric-display">
                                        <div class="metric-value" id="kc_display">--</div>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="plant_maturity">Plant Maturity</label>
                                    <select id="plant_maturity" onchange="updateMaturityInfo()" required>
                                        <option value="Young" data-zr="0.30" {'selected' if current_config.get('plant_maturity', 'Mature') == 'Young' else ''}>Young</option>
                                        <option value="Middle" data-zr="0.45" {'selected' if current_config.get('plant_maturity', 'Mature') == 'Middle' else ''}>Middle</option>
                                        <option value="Mature" data-zr="0.60" {'selected' if current_config.get('plant_maturity', 'Mature') == 'Mature' else ''}>Mature</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Root Zone (Zr)</label>
                                    <div class="metric-display">
                                        <div class="metric-value yellow" id="zr_display">0.60m</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Soil & Irrigation Card -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon blue">💧</div>
                            <span class="card-title">Soil & Irrigation</span>
                        </div>
                        <div class="card-body">
                            <div class="form-grid form-grid-4">
                                <div class="form-group">
                                    <label for="soil_type">Soil Type</label>
                                    <select id="soil_type" required>
                                        <option value="">Select</option>
                                        <option value="Sandy Loam" {is_selected('soil_type', 'Sandy Loam')}>Sandy Loam</option>
                                        <option value="Loamy Sand" {is_selected('soil_type', 'Loamy Sand')}>Loamy Sand</option>
                                        <option value="Loam" {is_selected('soil_type', 'Loam')}>Loam</option>
                                        <option value="Sandy Clay Loam" {is_selected('soil_type', 'Sandy Clay Loam')}>Sandy Clay Loam</option>
                                        <option value="Clay Loam" {is_selected('soil_type', 'Clay Loam')}>Clay Loam</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="irrigation_type">Irrigation Type</label>
                                    <select id="irrigation_type" required>
                                        <option value="">Select</option>
                                        <option value="Drip" {is_selected('irrigation_type', 'Drip')}>Drip (90%)</option>
                                        <option value="Sprinkler" {is_selected('irrigation_type', 'Sprinkler')}>Sprinkler (80%)</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="emitter_rate">Emitter Rate (L/h)</label>
                                    <select id="emitter_rate" required>
                                        <option value="">Select</option>
                                        <option value="20" {is_selected('emitter_rate', 20)}>20</option>
                                        <option value="30" {is_selected('emitter_rate', 30)}>30</option>
                                        <option value="35" {is_selected('emitter_rate', 35)}>35</option>
                                        <option value="40" {is_selected('emitter_rate', 40)}>40</option>
                                        <option value="50" {is_selected('emitter_rate', 50)}>50</option>
                                        <option value="58" {is_selected('emitter_rate', 58)}>58</option>
                                        <option value="70" {is_selected('emitter_rate', 70)}>70</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="num_sprinklers">Sprinklers/Tree</label>
                                    <select id="num_sprinklers" onchange="updateWettedPct()" required>
                                        <option value="">Select</option>
                                        <option value="1" data-wetted="25" {is_selected('num_sprinklers', 1)}>1</option>
                                        <option value="2" data-wetted="50" {is_selected('num_sprinklers', 2)}>2</option>
                                        <option value="3" data-wetted="75" {is_selected('num_sprinklers', 3)}>3</option>
                                        <option value="4" data-wetted="85" {is_selected('num_sprinklers', 4)}>4</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Canopy & Coverage Card -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon yellow">🌳</div>
                            <span class="card-title">Canopy & Coverage</span>
                        </div>
                        <div class="card-body">
                            <div class="form-grid form-grid-4">
                                <div class="form-group">
                                    <label for="plant_spacing">Tree Spacing (m)</label>
                                    <select id="plant_spacing" required>
                                        <option value="">Select</option>
                                        <option value="10" {is_selected('plant_spacing', 10)}>10m</option>
                                        <option value="11" {is_selected('plant_spacing', 11)}>11m</option>
                                        <option value="12" {is_selected('plant_spacing', 12)}>12m</option>
                                        <option value="15" {is_selected('plant_spacing', 15)}>15m</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="canopy_radius">Canopy Radius (m)</label>
                                    <input type="number" id="canopy_radius" value="{current_config.get('canopy_radius', 3.0)}" min="0.5" max="10" step="0.5" required>
                                </div>
                                <div class="form-group">
                                    <label for="wetted_pct">Wetted Area %</label>
                                    <input type="number" id="wetted_pct" value="{current_config.get('wetted_pct', 50.0)}" min="10" max="100" step="5" required>
                                </div>
                                <div class="form-group">
                                    <label for="trunk_buffer_enabled">Trunk Buffer</label>
                                    <select id="trunk_buffer_enabled" onchange="toggleTrunkBuffer()" required>
                                        <option value="false" {'selected' if not current_config.get('trunk_buffer_enabled', False) else ''}>Disabled</option>
                                        <option value="true" {'selected' if current_config.get('trunk_buffer_enabled', False) else ''}>Enabled</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-grid" id="trunk_radius_row" style="display: {'grid' if current_config.get('trunk_buffer_enabled', False) else 'none'}; margin-top: 16px;">
                                <div class="form-group">
                                    <label for="trunk_buffer_radius">Buffer Radius (m)</label>
                                    <input type="number" id="trunk_buffer_radius" value="{current_config.get('trunk_buffer_radius', 0.0)}" min="0" max="0.7" step="0.1">
                                    <small>Max 0.7m, capped at 20% of canopy</small>
                                </div>
                            </div>
                            <div class="info-alert blue" id="area_preview" style="margin-top: 16px;">
                                <span>📐</span>
                                <span>Canopy: --m² → Root: --m² → Wetted: --m²</span>
                            </div>
                        </div>
                        <div class="action-bar">
                            <button type="button" class="btn btn-secondary" onclick="window.location.reload()">Reset</button>
                            <button type="submit" class="btn btn-primary">💾 Save Configuration</button>
                        </div>
                    </div>
                </form>
            </main>
        </div>

        <script>
            // Growth stage names mapping
            const stageNames = {{
                '1-45': 'Flowering & fruit setting (0-45 Days)',
                '46-75': 'Early fruit growth (46-75 days)',
                '76-110': 'Fruit growth (76-110 days)',
                '111-140': 'Fruit Maturity (111-140 days)',
                '141-290': 'Vegetative Growth (141-290 days)',
                '291-320': 'Floral initiation (291-320 days)',
                '321-365': 'Floral development (321-365 days)'
            }};

            // Update day range and Kc display when stage changes
            function updateDayRange() {{
                const stageSelect = document.getElementById('growth_stage');
                const dayInput = document.getElementById('growth_day');
                const hintElement = document.getElementById('day_range_hint');
                const kcDisplay = document.getElementById('kc_display');

                const selectedOption = stageSelect.options[stageSelect.selectedIndex];
                const minDay = parseInt(selectedOption.dataset.min);
                const maxDay = parseInt(selectedOption.dataset.max);
                const kc = selectedOption.dataset.kc;

                // Update day input constraints
                dayInput.min = minDay;
                dayInput.max = maxDay;

                // Clamp current value if outside range
                const currentDay = parseInt(dayInput.value);
                if (currentDay < minDay) dayInput.value = minDay;
                if (currentDay > maxDay) dayInput.value = maxDay;

                // Update hint and Kc display
                hintElement.textContent = `Day ${{minDay}}-${{maxDay}}`;
                kcDisplay.textContent = kc;
            }}

            // C1: Update Plant Maturity info
            function updateMaturityInfo() {{
                const maturitySelect = document.getElementById('plant_maturity');
                const zrDisplay = document.getElementById('zr_display');

                const selectedOption = maturitySelect.options[maturitySelect.selectedIndex];
                const zr = selectedOption.dataset.zr;

                zrDisplay.textContent = `${{zr}}m`;
            }}

            // C2: Update wetted percentage based on sprinkler count
            function updateWettedPct() {{
                const sprSelect = document.getElementById('num_sprinklers');
                const wettedInput = document.getElementById('wetted_pct');

                const selectedOption = sprSelect.options[sprSelect.selectedIndex];
                if (selectedOption && selectedOption.dataset.wetted) {{
                    wettedInput.value = selectedOption.dataset.wetted;
                }}
                updateAreaPreview();
            }}

            // C3: Toggle trunk buffer radius visibility
            function toggleTrunkBuffer() {{
                const enabled = document.getElementById('trunk_buffer_enabled').value === 'true';
                const radiusRow = document.getElementById('trunk_radius_row');
                radiusRow.style.display = enabled ? 'grid' : 'none';

                if (!enabled) {{
                    document.getElementById('trunk_buffer_radius').value = 0;
                }}
                updateAreaPreview();
            }}

            // Update area preview
            function updateAreaPreview() {{
                const canopyRadius = parseFloat(document.getElementById('canopy_radius').value) || 3.0;
                const wettedPct = parseFloat(document.getElementById('wetted_pct').value) || 50;
                const bufferEnabled = document.getElementById('trunk_buffer_enabled').value === 'true';
                const bufferRadius = bufferEnabled ? (parseFloat(document.getElementById('trunk_buffer_radius').value) || 0) : 0;

                const canopyArea = Math.PI * canopyRadius * canopyRadius;
                let bufferArea = Math.PI * bufferRadius * bufferRadius;

                // Cap buffer at 20% of canopy
                const maxBuffer = canopyArea * 0.20;
                if (bufferArea > maxBuffer) bufferArea = maxBuffer;

                const rootArea = canopyArea - bufferArea;
                const wettedArea = rootArea * (wettedPct / 100);

                document.getElementById('area_preview').innerHTML =
                    `<span>📐</span><span>Canopy: ${{canopyArea.toFixed(1)}}m² → Root: ${{rootArea.toFixed(1)}}m² → Wetted: ${{wettedArea.toFixed(1)}}m²</span>`;
            }}

            // Initialize on page load
            window.onload = function() {{
                // Find the stage that matches current day
                const currentDay = parseInt(document.getElementById('growth_day').value);
                const stageSelect = document.getElementById('growth_stage');

                for (let i = 0; i < stageSelect.options.length; i++) {{
                    const opt = stageSelect.options[i];
                    const minDay = parseInt(opt.dataset.min);
                    const maxDay = parseInt(opt.dataset.max);
                    if (currentDay >= minDay && currentDay <= maxDay) {{
                        stageSelect.selectedIndex = i;
                        break;
                    }}
                }}
                updateDayRange();
                updateMaturityInfo();
                updateAreaPreview();
            }};

            // Add event listeners for area preview updates
            document.getElementById('canopy_radius').addEventListener('change', updateAreaPreview);
            document.getElementById('wetted_pct').addEventListener('change', updateAreaPreview);
            document.getElementById('trunk_buffer_radius').addEventListener('change', updateAreaPreview);

            // Validate day when changed
            document.getElementById('growth_day').addEventListener('change', function() {{
                const stageSelect = document.getElementById('growth_stage');
                const selectedOption = stageSelect.options[stageSelect.selectedIndex];
                const minDay = parseInt(selectedOption.dataset.min);
                const maxDay = parseInt(selectedOption.dataset.max);
                let value = parseInt(this.value);

                if (value < minDay) this.value = minDay;
                if (value > maxDay) this.value = maxDay;
            }});

            document.getElementById('configForm').addEventListener('submit', async function(e) {{
                e.preventDefault();

                const growthDay = parseInt(document.getElementById('growth_day').value);
                const stageSelect = document.getElementById('growth_stage');
                const stageValue = stageSelect.value;
                const stageName = stageNames[stageValue];

                // First update growth stage
                try {{
                    await fetch('/growth-stage/set', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ start_day: growthDay }})
                    }});
                }} catch (err) {{ console.log('Growth stage update:', err); }}

                // Then update irrigation config
                const formData = {{
                    farm_name: document.getElementById('farm_name').value,
                    crop_type: document.getElementById('crop_type').value,
                    crop_growth_stage: stageName,
                    soil_type: document.getElementById('soil_type').value,
                    irrigation_type: document.getElementById('irrigation_type').value,
                    emitter_rate: parseFloat(document.getElementById('emitter_rate').value),
                    plant_spacing: parseFloat(document.getElementById('plant_spacing').value),
                    num_sprinklers: parseInt(document.getElementById('num_sprinklers').value),
                    canopy_radius: parseFloat(document.getElementById('canopy_radius').value),
                    // C1: Plant Maturity Stage
                    plant_maturity: document.getElementById('plant_maturity').value,
                    // C2: Wetted percentage
                    wetted_pct: parseFloat(document.getElementById('wetted_pct').value),
                    // C3: Trunk buffer
                    trunk_buffer_enabled: document.getElementById('trunk_buffer_enabled').value === 'true',
                    trunk_buffer_radius: parseFloat(document.getElementById('trunk_buffer_radius').value) || 0
                }};

                try {{
                    const response = await fetch('/irrigation/config', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(formData)
                    }});

                    const result = await response.json();

                    if (response.ok) {{
                        document.getElementById('message').innerHTML =
                            '<div class="message success">Configuration saved! Redirecting to dashboard...</div>';
                        setTimeout(() => window.location.href = '/dashboard', 2000);
                    }} else {{
                        document.getElementById('message').innerHTML =
                            '<div class="message error">Error: ' + (result.detail || 'Unknown error') + '</div>';
                    }}
                }} catch (error) {{
                    document.getElementById('message').innerHTML =
                        '<div class="message error">Connection error: ' + error.message + '</div>';
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    uvicorn.run("jetson_edge_v3:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
