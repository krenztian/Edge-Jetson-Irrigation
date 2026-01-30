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
    """Update sensor status tracking"""
    if sensor_name in sensor_status:
        sensor_status[sensor_name] = {
            "online": online,
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
            "canopy_radius": 3.0  # NEW: Canopy radius in meters
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
    canopy_radius: float = 3.0  # NEW: Canopy radius in meters

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

    def get_swhc(self, soil_type):
        swhc_values = {
            'Sandy Loam': 120, 'Loamy Sand': 80, 'Loam': 160,
            'Sandy Clay Loam': 110, 'Clay Loam': 160
        }
        return swhc_values.get(soil_type, 160)

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
        """Enhanced irrigation calculation with VPD checking"""
        swhc = self.get_swhc(config.soil_type)

        # Use canopy_radius if available
        canopy_radius = getattr(config, 'canopy_radius', None) or config.__dict__.get('canopy_radius')
        wetted_area = self.calculate_wetted_area(
            config.plant_spacing, config.emitter_rate, config.num_sprinklers, canopy_radius
        )

        taw = swhc * self.soil_depth
        mad = 0.3
        RAW = taw * mad

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

        if current_deficit >= RAW:
            irrigation_volume = self.calculate_irrigation_volume(
                current_deficit, wetted_area, irrigation_efficiency
            )
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
            recommendation = f"No pump activation needed. Deficit: {current_deficit:.1f}mm (Threshold: {RAW:.1f}mm)"
            irrigation_needed = False

        # Add VPD override suggestion
        if vpd_status and vpd_status["status"] == "high" and not irrigation_needed:
            recommendation += f" | VPD Warning: {vpd_status['action']}"

        return {
            "irrigation_needed": irrigation_needed,
            "irrigation_volume_l_per_tree": round(irrigation_volume, 2),
            "irrigation_time_seconds": irrigation_time_seconds,
            "eto_mm_day": round(eto_mm_day, 3),
            "etc_mm_day": round(etc, 3),
            "deficit_mm": round(current_deficit, 2),
            "RAW_threshold_mm": round(RAW, 2),
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
                "wetted_area_m2": round(wetted_area, 2),
                "canopy_radius_m": canopy_radius
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
    """Get current rainfall from latest data"""
    if prediction_history:
        latest = prediction_history[-1]
        return latest.get("input_data", {}).get("rainfall_mm", 0)
    return 0

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
@app.post("/predict-esp32")
async def predict_eto_esp32_compatible(data: dict):
    """ESP32-compatible endpoint that handles parameter name differences"""
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

    # Prepare data for charts
    history_data = {
        'labels': [], 'humidity': [], 'wind_speed': [], 'sunshine_hours': [],
        'estimated_rad': [], 'temp_min': [], 'temp_max': [], 'eto_values': [],
        'etc_values': [], 'rainfall': [], 'vpd': []
    }

    for pred in list(prediction_history)[-12:]:
        timestamp = pred.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%H:%M')
            except:
                formatted_time = timestamp[-8:-3] if len(timestamp) > 8 else timestamp

            history_data['labels'].append(formatted_time)
            input_data = pred.get('input_data', {})
            history_data['humidity'].append(input_data.get('humidity', 0))
            history_data['wind_speed'].append(input_data.get('wind_speed', 0))
            history_data['sunshine_hours'].append(input_data.get('sunshine_hours', 0))
            history_data['estimated_rad'].append(pred.get('estimated_rad', 0))
            history_data['temp_min'].append(input_data.get('tmin', 0))
            history_data['temp_max'].append(input_data.get('tmax', 0))
            history_data['eto_values'].append(pred.get('eto_mm_day', 0))
            history_data['rainfall'].append(input_data.get('rainfall_mm', 0))
            history_data['vpd'].append(input_data.get('vpd', 0) or 0)

            if latest_irrigation:
                history_data['etc_values'].append(latest_irrigation.get('etc_mm_day', 0))
            else:
                history_data['etc_values'].append(0)

    # Current values
    current_eto = latest_data.get("eto_mm_day", 0)
    current_etc = latest_irrigation.get("etc_mm_day", 0) if latest_irrigation else 0
    current_rad = latest_data.get("estimated_rad", 0)
    current_temp_min = latest_data.get("input_data", {}).get("tmin", 0)
    current_temp_max = latest_data.get("input_data", {}).get("tmax", 0)
    current_humidity = latest_data.get("input_data", {}).get("humidity", 0)
    current_wind = latest_data.get("input_data", {}).get("wind_speed", 0)
    current_sunshine = latest_data.get("input_data", {}).get("sunshine_hours", 0)
    current_pressure = latest_data.get("input_data", {}).get("pressure", 0) or 0
    current_vpd = latest_data.get("input_data", {}).get("vpd", 0) or 0

    irrigation_volume = latest_irrigation.get("irrigation_volume_l_per_tree", 0)
    irrigation_needed = latest_irrigation.get("irrigation_needed", False)

    # Growth stage info
    current_day = growth_stage_config["current_day"]
    current_stage = get_growth_stage_from_day(current_day)
    vpd_range = VPD_RANGES.get(current_stage, {"min": 0, "max": 2, "optimal": 1})

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

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Pump Dashboard v3</title>
        <meta http-equiv="refresh" content="30">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #e3f2fd 0%, #f8f9ff 100%); min-height: 100vh; color: #2c3e50; line-height: 1.6; }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; margin-bottom: 25px; padding: 20px; background: rgba(255, 255, 255, 0.9); border-radius: 20px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); }}
            .header h1 {{ font-size: 2rem; font-weight: 600; color: #1976D2; margin-bottom: 8px; }}
            .header .subtitle {{ color: #666; font-size: 1rem; }}
            .nav-bar {{ display: flex; justify-content: center; gap: 12px; margin-bottom: 25px; flex-wrap: wrap; }}
            .nav-btn {{ padding: 8px 16px; background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(25, 118, 210, 0.2); border-radius: 15px; color: #1976D2; text-decoration: none; font-weight: 500; font-size: 0.85rem; transition: all 0.3s ease; }}
            .nav-btn:hover, .nav-btn.active {{ background: #1976D2; color: white; transform: translateY(-1px); }}
            .growth-stage-banner {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 25px; border-radius: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px; }}
            .growth-info {{ display: flex; align-items: center; gap: 20px; }}
            .growth-day {{ font-size: 2.5rem; font-weight: 700; }}
            .growth-stage-name {{ font-size: 1.1rem; opacity: 0.9; }}
            .vpd-indicator {{ background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 10px; text-align: center; }}
            .vpd-value {{ font-size: 1.5rem; font-weight: 600; }}
            .vpd-status {{ font-size: 0.85rem; opacity: 0.9; }}
            .sensor-status-bar {{ background: rgba(255, 255, 255, 0.9); padding: 10px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px; }}
            .sensor-item {{ display: flex; align-items: center; font-size: 0.85rem; }}
            .dashboard-main {{ display: grid; grid-template-columns: 350px 1fr; gap: 25px; margin-bottom: 30px; }}
            .temp-main {{ background: linear-gradient(135deg, #42a5f5 0%, #1976d2 100%); color: white; border-radius: 25px; padding: 35px; position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between; min-height: 400px; }}
            .temp-main::before {{ content: ''; position: absolute; top: -50%; right: -50%; width: 200px; height: 200px; background: rgba(255, 255, 255, 0.1); border-radius: 50%; }}
            .location-info {{ position: relative; z-index: 2; }}
            .location {{ font-size: 1.1rem; font-weight: 500; margin-bottom: 5px; opacity: 0.9; }}
            .date-time {{ font-size: 0.9rem; opacity: 0.7; }}
            .temp-display {{ position: relative; z-index: 2; text-align: center; margin: 20px 0; }}
            .temp-current {{ font-size: 4.5rem; font-weight: 300; margin-bottom: 10px; }}
            .temp-status {{ font-size: 1.2rem; opacity: 0.9; }}
            .temp-range {{ display: flex; justify-content: space-between; position: relative; z-index: 2; margin-top: 20px; }}
            .temp-range-item {{ text-align: center; }}
            .temp-range-label {{ font-size: 0.8rem; opacity: 0.7; margin-bottom: 5px; }}
            .temp-range-value {{ font-size: 1.1rem; font-weight: 600; }}
            .weather-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin-bottom: 25px; }}
            .metric-card {{ background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 20px; backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease; position: relative; }}
            .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12); }}
            .metric-header {{ display: flex; align-items: center; margin-bottom: 12px; }}
            .metric-icon {{ width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 14px; font-weight: bold; color: white; }}
            .humidity-icon {{ background: #42a5f5; }}
            .wind-icon {{ background: #66bb6a; }}
            .sun-icon {{ background: #ffa726; }}
            .radiation-icon {{ background: #ef5350; }}
            .rain-icon {{ background: #2196f3; }}
            .pressure-icon {{ background: #9c27b0; }}
            .vpd-icon {{ background: #ff5722; }}
            .metric-title {{ font-size: 0.85rem; color: #666; font-weight: 500; }}
            .metric-value {{ font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-bottom: 5px; }}
            .metric-unit {{ font-size: 0.8rem; color: #999; margin-bottom: 10px; }}
            .metric-bar {{ width: 100%; height: 5px; background: #f0f0f0; border-radius: 3px; overflow: hidden; }}
            .metric-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.5s ease; }}
            .humidity-bar {{ background: #42a5f5; }}
            .wind-bar {{ background: #66bb6a; }}
            .sun-bar {{ background: #ffa726; }}
            .radiation-bar {{ background: #ef5350; }}
            .rain-bar {{ background: #2196f3; }}
            .pressure-bar {{ background: #9c27b0; }}
            .vpd-bar {{ background: #ff5722; }}
            .pump-section {{ background: {pump_status_color}; color: white; border-radius: 25px; padding: 30px; margin-bottom: 30px; text-align: center; position: relative; animation: {pump_animation}; }}
            @keyframes pumpPulse {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }} 50% {{ box-shadow: 0 0 0 15px rgba(76, 175, 80, 0); }} }}
            @keyframes irrigationPulse {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(239, 83, 80, 0.4); }} 50% {{ box-shadow: 0 0 0 15px rgba(239, 83, 80, 0); }} }}
            .unlimited-mode {{ border: 2px solid rgba(255, 255, 255, 0.5); animation: unlimitedPulse 2s ease-in-out infinite; }}
            @keyframes unlimitedPulse {{ 0%, 100% {{ border-color: rgba(255, 255, 255, 0.5); }} 50% {{ border-color: rgba(255, 255, 255, 1); }} }}
            .pump-status {{ font-size: 2rem; font-weight: 600; margin-bottom: 15px; }}
            .pump-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px; margin-top: 20px; }}
            .pump-metric {{ text-align: center; }}
            .pump-metric-value {{ font-size: 2.5rem; font-weight: 600; margin-bottom: 5px; }}
            .pump-metric-label {{ font-size: 1rem; opacity: 0.8; font-weight: 500; }}
            .water-level-container {{ display: flex; flex-direction: column; align-items: center; background: rgba(255, 255, 255, 0.15); border-radius: 15px; padding: 15px; }}
            .water-level-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 10px; opacity: 0.8; }}
            .battery-gauge {{ display: flex; align-items: flex-end; margin-bottom: 10px; }}
            .battery-body {{ width: 50px; height: 80px; border: 3px solid rgba(255, 255, 255, 0.8); border-radius: 6px; position: relative; background: rgba(255, 255, 255, 0.1); overflow: hidden; margin-right: 5px; }}
            .battery-fill {{ position: absolute; bottom: 0; left: 0; right: 0; background: {water_fill_color}; transition: height 0.8s ease; border-radius: 3px 3px 0 0; height: {water_level_percent}%; }}
            .battery-tip {{ width: 15px; height: 6px; background: rgba(255, 255, 255, 0.8); border-radius: 0 2px 2px 0; margin-bottom: 35px; }}
            .water-level-labels {{ text-align: center; font-size: 0.85rem; opacity: 0.8; }}
            .water-level-current {{ font-weight: 600; margin-bottom: 2px; }}
            .water-level-threshold {{ font-size: 0.75rem; opacity: 0.7; }}
            .countdown-timer {{ background: rgba(255, 255, 255, 0.15); border-radius: 15px; padding: 20px; margin-top: 20px; }}
            .countdown-time {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 5px; }}
            .countdown-label {{ font-size: 0.9rem; opacity: 0.9; }}
            .pump-controls {{ display: flex; justify-content: center; gap: 15px; margin-top: 20px; flex-wrap: wrap; }}
            .pump-btn {{ padding: 12px 25px; font-size: 16px; border: none; border-radius: 12px; cursor: pointer; font-weight: 600; transition: all 0.3s ease; color: white; background: rgba(255, 255, 255, 0.2); border: 2px solid rgba(255, 255, 255, 0.3); }}
            .pump-btn:hover {{ background: rgba(255, 255, 255, 0.3); transform: translateY(-2px); }}
            .trends-section {{ margin-top: 30px; }}
            .section-title {{ font-size: 1.3rem; font-weight: 600; color: #1976D2; margin-bottom: 20px; text-align: center; }}
            .trends-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .trend-card {{ background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 25px; backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border: 1px solid rgba(255, 255, 255, 0.2); }}
            .trend-header {{ display: flex; align-items: center; margin-bottom: 20px; }}
            .trend-icon {{ width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 12px; font-weight: bold; color: white; }}
            .trend-title {{ font-size: 1rem; font-weight: 600; color: #2c3e50; }}
            .chart-container {{ position: relative; height: 200px; width: 100%; }}
            @media (max-width: 1200px) {{ .dashboard-main {{ grid-template-columns: 1fr; }} .temp-main {{ min-height: 250px; }} }}
            @media (max-width: 768px) {{ .container {{ padding: 15px; }} .weather-grid {{ grid-template-columns: repeat(2, 1fr); }} .pump-metrics {{ grid-template-columns: 1fr; gap: 20px; }} .trends-grid {{ grid-template-columns: 1fr; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Smart Irrigation Control System v3</h1>
                <div class="subtitle">Real-time monitoring with VPD checking & growth stage tracking</div>
            </div>

            <div class="nav-bar">
                <a href="/dashboard" class="nav-btn active">Dashboard</a>
                <a href="/config" class="nav-btn">Settings</a>
                <a href="/irrigation/history" class="nav-btn">Raw_Data</a>
                <a href="/docs" class="nav-btn">API</a>
            </div>

            <div class="growth-stage-banner">
                <div class="growth-info">
                    <div class="growth-day">Day {current_day}</div>
                    <div>
                        <div class="growth-stage-name">{current_stage}</div>
                        <div style="font-size:0.8rem;opacity:0.8;">Kc: {irrigation_config.get('kc', 0.6):.2f}</div>
                    </div>
                </div>
                <div class="vpd-indicator" style="background:{vpd_status.get('color', '#999')}40;">
                    <div class="vpd-value">{current_vpd:.2f} kPa</div>
                    <div class="vpd-status">{vpd_status.get('status', 'N/A').upper()}</div>
                    <div style="font-size:0.7rem;opacity:0.8;">Range: {vpd_range.get('min', 0)}-{vpd_range.get('max', 2)} kPa</div>
                </div>
            </div>

            <div class="sensor-status-bar">
                <div class="sensor-item">{sensor_indicator('temperature')}Temperature</div>
                <div class="sensor-item">{sensor_indicator('humidity')}Humidity</div>
                <div class="sensor-item">{sensor_indicator('pressure')}Pressure</div>
                <div class="sensor-item">{sensor_indicator('wind')}Wind</div>
                <div class="sensor-item">{sensor_indicator('solar')}Solar</div>
                <div class="sensor-item">{sensor_indicator('rain')}Rain</div>
                <div class="sensor-item">{sensor_indicator('vpd')}VPD</div>
            </div>

            <div class="dashboard-main">
                <div class="temp-main">
                    <div class="location-info">
                        <div class="location" id="farmLocation">Detecting location...</div>
                        <div class="date-time">{datetime.now().strftime("Today %d %B")}</div>
                    </div>
                    <div class="temp-display">
                        <div class="temp-current">{current_temp_max:.0f}deg</div>
                        <div class="temp-status">Current Temperature</div>
                    </div>
                    <div class="temp-range">
                        <div class="temp-range-item"><div class="temp-range-label">Min</div><div class="temp-range-value">{current_temp_min:.0f}deg</div></div>
                        <div class="temp-range-item"><div class="temp-range-label">Max</div><div class="temp-range-value">{current_temp_max:.0f}deg</div></div>
                        <div class="temp-range-item"><div class="temp-range-label">Range</div><div class="temp-range-value">{current_temp_max - current_temp_min:.1f}deg</div></div>
                    </div>
                </div>

                <div>
                    <div class="weather-grid">
                        <div class="metric-card">
                            <div class="metric-header"><div class="metric-icon humidity-icon">H</div><div class="metric-title">Humidity</div></div>
                            <div class="metric-value">{current_humidity:.0f}</div>
                            <div class="metric-unit">% relative humidity</div>
                            <div class="metric-bar"><div class="metric-bar-fill humidity-bar" style="width: {current_humidity}%"></div></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-header"><div class="metric-icon wind-icon">W</div><div class="metric-title">Wind Speed</div></div>
                            <div class="metric-value">{current_wind:.2f}</div>
                            <div class="metric-unit">m/s wind speed</div>
                            <div class="metric-bar"><div class="metric-bar-fill wind-bar" style="width: {min(current_wind * 10, 100)}%"></div></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-header"><div class="metric-icon sun-icon">S</div><div class="metric-title">Sunshine</div></div>
                            <div class="metric-value">{current_sunshine:.2f}</div>
                            <div class="metric-unit">hours of sunshine</div>
                            <div class="metric-bar"><div class="metric-bar-fill sun-bar" style="width: {min(current_sunshine * 4.17, 100)}%"></div></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-header"><div class="metric-icon radiation-icon">R</div><div class="metric-title">Solar Rad</div></div>
                            <div class="metric-value">{current_rad:.1f}</div>
                            <div class="metric-unit">MJ/m2/day</div>
                            <div class="metric-bar"><div class="metric-bar-fill radiation-bar" style="width: {min(current_rad * 3.33, 100)}%"></div></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-header"><div class="metric-icon rain-icon">R</div><div class="metric-title">Rainfall</div></div>
                            <div class="metric-value">{current_rainfall:.1f}</div>
                            <div class="metric-unit">mm today</div>
                            <div class="metric-bar"><div class="metric-bar-fill rain-bar" style="width: {min(current_rainfall * 2, 100)}%"></div></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-header"><div class="metric-icon pressure-icon">P</div><div class="metric-title">Pressure</div></div>
                            <div class="metric-value">{current_pressure:.0f}</div>
                            <div class="metric-unit">hPa</div>
                            <div class="metric-bar"><div class="metric-bar-fill pressure-bar" style="width: {min((current_pressure-950)/100*100, 100) if current_pressure else 0}%"></div></div>
                        </div>
                        <div class="metric-card" style="border: 2px solid {vpd_status.get('color', '#999')};">
                            <div class="metric-header"><div class="metric-icon vpd-icon">V</div><div class="metric-title">VPD</div></div>
                            <div class="metric-value" style="color:{vpd_status.get('color', '#2c3e50')};">{current_vpd:.2f}</div>
                            <div class="metric-unit">kPa ({vpd_status.get('status', 'N/A')})</div>
                            <div class="metric-bar"><div class="metric-bar-fill vpd-bar" style="width: {min(current_vpd * 50, 100)}%"></div></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="pump-section">
                <div class="pump-status">{pump_status_text}</div>
                <div class="pump-metrics">
                    <div class="pump-metric"><div class="pump-metric-value">{irrigation_volume:.1f}</div><div class="pump-metric-label">Liters per Tree</div></div>
                    <div class="pump-metric"><div class="pump-metric-value">{irrigation_time_minutes:.0f}</div><div class="pump-metric-label">Running Time (min)</div></div>
                    <div class="pump-metric water-level-container">
                        <div class="water-level-title">Soil Water Status</div>
                        <div class="battery-gauge"><div class="battery-body"><div class="battery-fill"></div></div><div class="battery-tip"></div></div>
                        <div class="water-level-labels"><div class="water-level-current">{water_level_percent:.0f}% Available</div><div class="water-level-threshold">Deficit: {fmt(deficit_mm, '')} mm</div></div>
                    </div>
                </div>
                <div class="pump-controls">
                    <button class="pump-btn" onclick="controlPump('close')">Stop Pump</button>
                    <button class="pump-btn" onclick="controlPump('open', null)">Open Pump</button>
                    <button class="pump-btn" onclick="controlPump('open', 3600)">Run 60min</button>
                    <button class="pump-btn" onclick="controlPump('open', 7200)">Run 120min</button>
                </div>
                {countdown_section}
            </div>

            <div class="trends-section">
                <div class="section-title">Historical Trends</div>
                <div class="trends-grid">
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon humidity-icon">H</div><div class="trend-title">Humidity Trend</div></div><div class="chart-container"><canvas id="humidityChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon wind-icon">W</div><div class="trend-title">Wind Speed Trend</div></div><div class="chart-container"><canvas id="windChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon sun-icon">S</div><div class="trend-title">Sunshine Trend</div></div><div class="chart-container"><canvas id="sunshineChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon radiation-icon">R</div><div class="trend-title">Solar Radiation</div></div><div class="chart-container"><canvas id="radiationChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon rain-icon">R</div><div class="trend-title">Rainfall Trend</div></div><div class="chart-container"><canvas id="rainfallChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon" style="background:#1976D2;">T</div><div class="trend-title">Temperature Trend</div></div><div class="chart-container"><canvas id="temperatureChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon vpd-icon">V</div><div class="trend-title">VPD Trend</div></div><div class="chart-container"><canvas id="vpdChart"></canvas></div></div>
                    <div class="trend-card"><div class="trend-header"><div class="trend-icon" style="background:#42a5f5;">E</div><div class="trend-title">ETo Trend</div></div><div class="chart-container"><canvas id="etoChart"></canvas></div></div>
                </div>
            </div>
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

            function detectLocation() {{
                const locElement = document.getElementById("farmLocation");
                if (!locElement) return;
                locElement.innerText = "Detecting location...";
                if (!navigator.geolocation) {{ locElement.innerText = "Location unavailable"; return; }}
                navigator.geolocation.getCurrentPosition((pos) => {{
                    const lat = pos.coords.latitude.toFixed(6);
                    const lon = pos.coords.longitude.toFixed(6);
                    fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${{lat}}&lon=${{lon}}`)
                        .then(res => res.json())
                        .then(data => {{
                            if (data && data.address) {{
                                const addr = data.address;
                                let locationText = "";
                                if (addr.village) locationText += addr.village + ", ";
                                else if (addr.town) locationText += addr.town + ", ";
                                else if (addr.city) locationText += addr.city + ", ";
                                if (addr.state) locationText += addr.state + ", ";
                                if (addr.country) locationText += addr.country;
                                locElement.innerText = locationText;
                            }} else {{ locElement.innerText = "Location detected"; }}
                        }}).catch(() => {{ locElement.innerText = "Location detected"; }});
                }}, (err) => {{ locElement.innerText = "Location unavailable"; }}, {{ enableHighAccuracy: true, timeout: 7000 }});
            }}
            detectLocation();

            const chartConfig = {{
                type: 'line',
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ x: {{ display: true, ticks: {{ color: '#666', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,0.1)' }} }}, y: {{ display: true, ticks: {{ color: '#666', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,0.1)' }} }} }},
                    elements: {{ point: {{ radius: 3, hoverRadius: 5 }}, line: {{ borderWidth: 2, tension: 0.4 }} }}
                }}
            }};

            new Chart(document.getElementById('humidityChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['humidity']}, borderColor: '#42a5f5', backgroundColor: 'rgba(66, 165, 245, 0.1)', fill: true }}] }} }});
            new Chart(document.getElementById('windChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['wind_speed']}, borderColor: '#66bb6a', backgroundColor: 'rgba(102, 187, 106, 0.1)', fill: true }}] }} }});
            new Chart(document.getElementById('sunshineChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['sunshine_hours']}, borderColor: '#ffa726', backgroundColor: 'rgba(255, 167, 38, 0.1)', fill: true }}] }} }});
            new Chart(document.getElementById('radiationChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['estimated_rad']}, borderColor: '#ef5350', backgroundColor: 'rgba(239, 83, 80, 0.1)', fill: true }}] }} }});
            new Chart(document.getElementById('rainfallChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['rainfall']}, borderColor: '#2196f3', backgroundColor: 'rgba(33, 150, 243, 0.1)', fill: true }}] }} }});
            new Chart(document.getElementById('temperatureChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ label: 'Max', data: {history_data['temp_max']}, borderColor: '#ef5350', fill: false }}, {{ label: 'Min', data: {history_data['temp_min']}, borderColor: '#42a5f5', fill: false }}] }}, options: {{ ...chartConfig.options, plugins: {{ legend: {{ display: true }} }} }} }});
            new Chart(document.getElementById('vpdChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['vpd']}, borderColor: '#ff5722', backgroundColor: 'rgba(255, 87, 34, 0.1)', fill: true }}] }} }});
            new Chart(document.getElementById('etoChart').getContext('2d'), {{ ...chartConfig, data: {{ labels: {history_data['labels']}, datasets: [{{ data: {history_data['eto_values']}, borderColor: '#42a5f5', backgroundColor: 'rgba(66, 165, 245, 0.1)', fill: true }}] }} }});
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

    def is_selected(field, value):
        return "selected" if str(current_config.get(field)) == str(value) else ""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Farm Configuration v3</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #e3f2fd 0%, #f8f9ff 100%); min-height: 100vh; color: #2c3e50; line-height: 1.6; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; margin-bottom: 25px; padding: 25px 30px; background: rgba(255, 255, 255, 0.9); border-radius: 20px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); }}
            .header h1 {{ font-size: 2.2rem; font-weight: 600; color: #1976D2; margin-bottom: 10px; }}
            .header p {{ color: #666; font-size: 1rem; }}
            .nav-bar {{ display: flex; justify-content: center; gap: 12px; margin-bottom: 30px; flex-wrap: wrap; }}
            .nav-btn {{ padding: 8px 16px; background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(25, 118, 210, 0.2); border-radius: 15px; color: #1976D2; text-decoration: none; font-weight: 500; font-size: 0.85rem; transition: all 0.3s ease; }}
            .nav-btn:hover {{ background: #1976D2; color: white; transform: translateY(-1px); }}
            .nav-btn.active {{ background: #1976D2; color: white; }}
            .current-config {{ background: #4CAF50; color: white; padding: 25px; border-radius: 20px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); }}
            .current-config h3 {{ color: white; margin-bottom: 15px; font-size: 1.3rem; font-weight: 600; }}
            .config-preview {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px; }}
            .config-item {{ background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; text-align: center; }}
            .config-item-label {{ font-size: 0.8rem; opacity: 0.8; margin-bottom: 8px; font-weight: 500; }}
            .config-item-value {{ font-weight: 600; font-size: 0.95rem; line-height: 1.2; }}
            .config-form {{ background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
            .form-section {{ margin-bottom: 30px; padding: 25px; background: rgba(116, 185, 255, 0.05); border-radius: 15px; border-left: 4px solid #1976D2; }}
            .form-section h3 {{ color: #1976D2; margin-bottom: 20px; font-size: 1.3rem; font-weight: 600; }}
            .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
            .form-group {{ margin-bottom: 20px; }}
            .form-group label {{ display: block; margin-bottom: 8px; font-weight: 600; color: #2c3e50; font-size: 0.95rem; }}
            .form-group select, .form-group input {{ width: 100%; padding: 12px 15px; font-size: 14px; border: 2px solid #e0e0e0; border-radius: 10px; background: white; color: #2c3e50; transition: all 0.3s ease; font-family: inherit; }}
            .form-group select:focus, .form-group input:focus {{ outline: none; border-color: #1976D2; box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1); transform: translateY(-1px); }}
            .btn-group {{ display: flex; justify-content: center; gap: 15px; margin-top: 35px; flex-wrap: wrap; }}
            .btn {{ padding: 12px 25px; font-size: 16px; border: none; border-radius: 12px; cursor: pointer; font-weight: 600; transition: all 0.3s ease; text-decoration: none; display: inline-block; font-family: inherit; }}
            .btn-primary {{ background: #4CAF50; color: white; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3); }}
            .btn-primary:hover {{ background: #388E3C; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4); }}
            .btn-secondary {{ background: #1976D2; color: white; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3); }}
            .btn-secondary:hover {{ background: #1565C0; transform: translateY(-2px); }}
            .message {{ margin: 20px 0; padding: 15px 20px; border-radius: 12px; font-weight: 500; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }}
            .success {{ background: #4CAF50; color: white; }}
            .error {{ background: #ef5350; color: white; }}
            @media (max-width: 768px) {{ .form-grid {{ grid-template-columns: 1fr; }} .btn-group {{ flex-direction: column; }} .config-preview {{ grid-template-columns: repeat(2, 1fr); }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Farm Configuration v3</h1>
                <p>Configure irrigation, growth stage, and canopy parameters</p>
            </div>

            <div class="nav-bar">
                <a href="/dashboard" class="nav-btn">Dashboard</a>
                <a href="/config" class="nav-btn active">Settings</a>
                <a href="/irrigation/history" class="nav-btn">History</a>
                <a href="/docs" class="nav-btn">API</a>
            </div>

            <div class="current-config">
                <h3>Current Configuration - Day {current_day}</h3>
                <div class="config-preview">
                    <div class="config-item"><div class="config-item-label">Farm Name</div><div class="config-item-value">{current_config.get('farm_name', 'Not Set')}</div></div>
                    <div class="config-item"><div class="config-item-label">Crop Type</div><div class="config-item-value">{current_config.get('crop_type', 'Not Set')}</div></div>
                    <div class="config-item"><div class="config-item-label">Growth Day</div><div class="config-item-value">Day {current_day}</div></div>
                    <div class="config-item"><div class="config-item-label">Growth Stage</div><div class="config-item-value">{current_config.get('crop_growth_stage', 'Not Set')[:20]}...</div></div>
                    <div class="config-item"><div class="config-item-label">Soil Type</div><div class="config-item-value">{current_config.get('soil_type', 'Not Set')}</div></div>
                    <div class="config-item"><div class="config-item-label">Irrigation Type</div><div class="config-item-value">{current_config.get('irrigation_type', 'Not Set')}</div></div>
                    <div class="config-item"><div class="config-item-label">Emitter Rate</div><div class="config-item-value">{current_config.get('emitter_rate', 'Not Set')} L/h</div></div>
                    <div class="config-item"><div class="config-item-label">Canopy Radius</div><div class="config-item-value">{current_config.get('canopy_radius', 3.0)} m</div></div>
                </div>
            </div>

            <div class="config-form">
                <div id="message"></div>

                <form id="configForm">
                    <div class="form-section">
                        <h3>Farm Information</h3>
                        <div class="form-group">
                            <label for="farm_name">Farm Name:</label>
                            <input type="text" id="farm_name" value="{current_config.get('farm_name', '')}" placeholder="Enter your farm name" required>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Growth Stage Configuration (Phase 2)</h3>
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="growth_day">Current Growth Day (1-365):</label>
                                <input type="number" id="growth_day" value="{current_day}" min="1" max="365" required>
                                <small style="color:#666;">Stage auto-updates based on day number</small>
                            </div>
                            <div class="form-group">
                                <label for="crop_type">Crop Type:</label>
                                <select id="crop_type" required>
                                    <option value="">Select Crop Type</option>
                                    <option value="Durian" {is_selected('crop_type', 'Durian')}>Durian</option>
                                    <option value="Mangosteen" {is_selected('crop_type', 'Mangosteen')}>Mangosteen</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Soil & Irrigation System</h3>
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="soil_type">Soil Type:</label>
                                <select id="soil_type" required>
                                    <option value="">Select Soil Type</option>
                                    <option value="Sandy Loam" {is_selected('soil_type', 'Sandy Loam')}>Sandy Loam</option>
                                    <option value="Loamy Sand" {is_selected('soil_type', 'Loamy Sand')}>Loamy Sand</option>
                                    <option value="Loam" {is_selected('soil_type', 'Loam')}>Loam</option>
                                    <option value="Sandy Clay Loam" {is_selected('soil_type', 'Sandy Clay Loam')}>Sandy Clay Loam</option>
                                    <option value="Clay Loam" {is_selected('soil_type', 'Clay Loam')}>Clay Loam</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="irrigation_type">Irrigation Type:</label>
                                <select id="irrigation_type" required>
                                    <option value="">Select Irrigation Type</option>
                                    <option value="Drip" {is_selected('irrigation_type', 'Drip')}>Drip (90% efficiency)</option>
                                    <option value="Sprinkler" {is_selected('irrigation_type', 'Sprinkler')}>Sprinkler (80% efficiency)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Irrigation System Parameters</h3>
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="emitter_rate">Emitter Rate (L/h):</label>
                                <select id="emitter_rate" required>
                                    <option value="">Select Emitter Rate</option>
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
                                <label for="plant_spacing">Plant Spacing (m):</label>
                                <select id="plant_spacing" required>
                                    <option value="">Select Plant Spacing</option>
                                    <option value="10" {is_selected('plant_spacing', 10)}>10</option>
                                    <option value="11" {is_selected('plant_spacing', 11)}>11</option>
                                    <option value="12" {is_selected('plant_spacing', 12)}>12</option>
                                    <option value="15" {is_selected('plant_spacing', 15)}>15</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="num_sprinklers">Sprinklers per Tree:</label>
                                <select id="num_sprinklers" required>
                                    <option value="">Select Number</option>
                                    <option value="1" {is_selected('num_sprinklers', 1)}>1</option>
                                    <option value="2" {is_selected('num_sprinklers', 2)}>2</option>
                                    <option value="3" {is_selected('num_sprinklers', 3)}>3</option>
                                    <option value="4" {is_selected('num_sprinklers', 4)}>4</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="canopy_radius">Canopy Radius (m) - Phase 2:</label>
                                <input type="number" id="canopy_radius" value="{current_config.get('canopy_radius', 3.0)}" min="0.5" max="10" step="0.5" required>
                                <small style="color:#666;">Used for wetted area calculation</small>
                            </div>
                        </div>
                    </div>

                    <div class="btn-group">
                        <button type="submit" class="btn btn-primary">Save Configuration</button>
                        <button type="button" class="btn btn-secondary" onclick="window.location.reload()">Reload Current</button>
                    </div>
                </form>
            </div>
        </div>

        <script>
            document.getElementById('configForm').addEventListener('submit', async function(e) {{
                e.preventDefault();

                const growthDay = parseInt(document.getElementById('growth_day').value);

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
                    crop_growth_stage: "Auto", // Will be set by growth day
                    soil_type: document.getElementById('soil_type').value,
                    irrigation_type: document.getElementById('irrigation_type').value,
                    emitter_rate: parseFloat(document.getElementById('emitter_rate').value),
                    plant_spacing: parseFloat(document.getElementById('plant_spacing').value),
                    num_sprinklers: parseInt(document.getElementById('num_sprinklers').value),
                    canopy_radius: parseFloat(document.getElementById('canopy_radius').value)
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
