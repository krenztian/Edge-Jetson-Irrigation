"""
ETo Edge Computing + Smart Pump System v2.0
===========================================
Real-time Evapotranspiration prediction and smart pump control with:
- Rainfall integration
- InfluxDB recovery for missed data
- Enhanced validation for Rika sensors (45 m/s wind)

Version: 2.0.0
Date: January 2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
import numpy as np
import uvicorn
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from collections import deque
from contextlib import asynccontextmanager
import requests
import json
import os

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
# Valve Controller Communication
# =========================
class ValveController:
    def __init__(self, ip, port=80):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"

    async def get_valve_status(self):
        """Get current valve status from Arduino controller"""
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
        """Open valve for specified duration"""
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
        """Close valve immediately"""
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

# Initialize valve controller
valve_controller = ValveController(VALVE_CONTROLLER_IP, VALVE_CONTROLLER_PORT)

# =========================
# InfluxDB Recovery Functions
# =========================
class InfluxDBRecovery:
    def __init__(self):
        self.client = None
        self.query_api = None

    def connect(self):
        """Connect to InfluxDB"""
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
        """Query InfluxDB for data since last processed timestamp"""
        if not self.query_api:
            if not self.connect():
                return []

        try:
            # Build query
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
        """Close InfluxDB connection"""
        if self.client:
            self.client.close()

# Initialize recovery client
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

        # Set default irrigation config
        global irrigation_config
        irrigation_config = {
            "crop_type": "Durian",
            "crop_growth_stage": "Fruit growth (76-110 days)",
            "soil_type": "Loam",
            "irrigation_type": "Sprinkler",
            "emitter_rate": 40.0,
            "plant_spacing": 10.0,
            "num_sprinklers": 2,
            "farm_name": "ESP32 Farm"
        }
        logger.info("Default irrigation config set")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# =========================
# Startup Recovery Check
# =========================
async def check_and_recover_missed_data():
    """Check for and recover missed data on startup"""
    global last_processed_timestamp, recovery_status

    logger.info("Checking for missed data to recover...")
    recovery_status["last_recovery_attempt"] = datetime.now().isoformat()

    # Query last 24 hours of data
    records = influx_recovery.query_missed_data(hours_back=24)

    if records:
        recovered_count = 0
        for record in records:
            try:
                # Process each recovered record (calculate ETo, update irrigation)
                if all(record.get(k) is not None for k in ["temp_min", "temp_max", "humidity", "wind_speed", "sunshine_hours"]):
                    # Add to prediction history
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

        # Update last processed timestamp
        if records:
            last_processed_timestamp = records[-1]["timestamp"]
    else:
        logger.info("No missed data to recover")

# =========================
# Lifespan Event Handler
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    # Check initial valve status
    await valve_controller.get_valve_status()
    # Attempt to recover missed data
    await check_and_recover_missed_data()
    yield
    # Shutdown - cleanup
    influx_recovery.close()

# =========================
# FastAPI App with Lifespan
# =========================
app = FastAPI(
    title="ETo Edge Computing + Smart Pump System",
    description="Real-time Evapotranspiration prediction and smart pump control with rainfall integration and data recovery",
    version="2.0.0",
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
    pressure: Optional[float] = None  # NEW: pressure from ESP32
    vpd: Optional[float] = None       # NEW: VPD from ESP32
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
        # FIXED: Updated for Rika RK120-01 sensor (0-45 m/s range)
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

class EtoPrediction(BaseModel):
    eto_mm_day: float
    estimated_rad: float
    input_data: dict
    timestamp: str
    device_id: str
    processing_time_ms: float
    model_info: dict
    validation_warnings: List[str] = []

class IrrigationRecommendation(BaseModel):
    irrigation_needed: bool
    irrigation_volume_l_per_tree: float
    eto_mm_day: float
    etc_mm_day: float
    deficit_mm: float
    RAW_threshold_mm: float
    recommendation: str
    rainfall_mm: float = 0.0
    effective_rainfall_mm: float = 0.0
    farm_config: dict
    timestamp: str

class PumpCommand(BaseModel):
    action: str  # "open" or "close"
    duration: Optional[int] = None  # Duration in seconds for "open" action

class RecoveryRequest(BaseModel):
    hours_back: int = 24
    since_timestamp: Optional[str] = None

# =========================
# Enhanced Irrigation Calculator with FIXED Logic
# =========================
class IrrigationCalculator:
    def __init__(self):
        self.previous_deficit = 0
        self.yesterday_deficit = 0
        self.daily_irrigation_applied = 0
        self.last_irrigation_time = None
        self.soil_depth = 0.5  # root zone

    def get_swhc(self, soil_type):
        """Soil Water Holding Capacity"""
        swhc_values = {
            'Sandy Loam': 120,
            'Loamy Sand': 80,
            'Loam': 160,
            'Sandy Clay Loam': 110,
            'Clay Loam': 160
        }
        return swhc_values.get(soil_type, 160)

    def get_kc(self, crop_type, crop_growth_stage):
        """Crop coefficient values"""
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
        """Wetted diameter based on emitter rate (in meters)"""
        wetted_diameter_values = {
            20: 1.5, 30: 1.8, 35: 2.0, 40: 2.2, 50: 2.5, 58: 2.8, 70: 3.0
        }
        return wetted_diameter_values.get(int(emitter_rate), 2.0)

    def calculate_wetted_area(self, plant_spacing, emitter_rate, num_sprinklers, overlap_factor=0.8):
        """Calculate wetted area - FIXED VERSION"""
        import math
        wetted_diameter = self.get_wetted_diameter(emitter_rate)
        area_per_emitter = math.pi * (wetted_diameter / 2) ** 2
        total_wetted_area = num_sprinklers * area_per_emitter * overlap_factor

        # Use full plant allocation area
        tree_area_allocation = plant_spacing * plant_spacing

        return min(total_wetted_area, tree_area_allocation)

    def calculate_irrigation_volume(self, deficit_mm, wetted_area_m2, irrigation_efficiency):
        """Calculate irrigation volume based on deficit and wetted area"""
        gross_volume = deficit_mm * wetted_area_m2
        required_volume = gross_volume / irrigation_efficiency
        return required_volume

    def calculate_irrigation_time(self, volume, emitter_rate, num_sprinklers):
        """Calculate irrigation time - FIXED VERSION"""
        total_flow_rate = emitter_rate * num_sprinklers

        logger.info(f"Irrigation time calculation: Volume={volume:.1f}L, Rate={emitter_rate}L/h, Sprinklers={num_sprinklers}, Total={total_flow_rate}L/h")

        if total_flow_rate > 0:
            irrigation_time_hours = volume / total_flow_rate
            irrigation_time_seconds = int(irrigation_time_hours * 3600)
        else:
            irrigation_time_seconds = 0

        # Safety limit: Maximum 2 hours
        irrigation_time_seconds = min(irrigation_time_seconds, 7200)

        logger.info(f"Final irrigation time: {irrigation_time_seconds}s = {irrigation_time_seconds/60:.1f}min")

        return irrigation_time_seconds

    def calculate_effective_rainfall(self, rainfall_mm, irrigation_type="Drip"):
        """Calculate effective rainfall based on intensity"""
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
        """Calculate updated soil water deficit with rainfall"""
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
        """Mark that irrigation has been successfully completed"""
        self.previous_deficit = 0
        self.last_irrigation_time = datetime.now()
        logger.info("Irrigation completed - deficit reset to 0")

    def calculate_irrigation_recommendation(self, eto_mm_day, config, rainfall_mm=0.0):
        """Enhanced irrigation calculation with FIXED reset logic"""
        swhc = self.get_swhc(config.soil_type)
        wetted_area = self.calculate_wetted_area(
            config.plant_spacing, config.emitter_rate, config.num_sprinklers
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
            "farm_config": {
                "crop_type": config.crop_type,
                "growth_stage": config.crop_growth_stage,
                "soil_type": config.soil_type,
                "kc": round(kc, 2),
                "wetted_area_m2": round(wetted_area, 2)
            }
        }

# Initialize irrigation calculator
irrigation_calc = IrrigationCalculator()

# =========================
# Feature Engineering Functions
# =========================
def enhance_features_for_rad(tmin, tmax, humidity, wind_speed, sunshine_hours):
    """Apply feature engineering for RAD prediction"""
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
    """Apply nighttime constraints to RAD predictions"""
    if sunshine_hours <= threshold:
        scale = max(0.3, sunshine_hours / threshold)
        constrained_rad = rad_prediction * scale
    else:
        constrained_rad = rad_prediction
    return max(constrained_rad, 1.0)

# =========================
# Helper Function
# =========================
def fmt(value, unit=""):
    """Format numbers with 2 decimals or return N/A"""
    if isinstance(value, (int, float)):
        return f"{value:.2f}{unit}"
    return f"N/A{unit}"

# =========================
# Recovery Endpoints
# =========================
@app.post("/recover-missed-data")
async def recover_missed_data(request: RecoveryRequest):
    """Manually trigger recovery of missed data from InfluxDB"""
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
    """Get current recovery status"""
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
# Enhanced Prediction Endpoint with FIXED Logic
# =========================
@app.post("/predict", response_model=EtoPrediction)
async def predict_eto_endpoint(weather_data: WeatherInput):
    global last_processed_timestamp

    if not all(models[key] is not None for key in ["rf_model", "mlp_model", "scaler_rad", "scaler_eto"]):
        raise HTTPException(status_code=500, detail="Models not loaded")

    start_time = datetime.now()
    warnings = []

    try:
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

        # Step 3: Enhanced irrigation calculation
        irrigation_result = None
        if irrigation_config:
            config = IrrigationConfig(**irrigation_config)
            irrigation_result = irrigation_calc.calculate_irrigation_recommendation(
                eto_prediction, config, weather_data.rainfall_mm
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

        # Get current valve status
        await valve_controller.get_valve_status()

        # Store prediction data
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
            "irrigation_recommendation": irrigation_result
        }

        prediction_history.append(prediction_data)
        cycle_data.update(prediction_data)

        # Update last processed timestamp
        last_processed_timestamp = timestamp

        response = EtoPrediction(
            eto_mm_day=prediction_data["eto_mm_day"],
            estimated_rad=prediction_data["estimated_rad"],
            input_data=prediction_data["input_data"],
            timestamp=timestamp,
            device_id=weather_data.device_id,
            processing_time_ms=prediction_data["processing_time_ms"],
            validation_warnings=warnings,
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
        # Convert ESP32 parameter names to expected format
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

        # Create WeatherInput object
        weather_input = WeatherInput(**converted_data)

        # Call the main prediction function
        return await predict_eto_endpoint(weather_input)

    except Exception as e:
        logger.error(f"ESP32 prediction failed: {e}")
        raise HTTPException(status_code=422, detail=f"ESP32 data conversion failed: {e}")

# =========================
# Valve Control Endpoints
# =========================
@app.post("/pump/control")
async def control_pump(command: PumpCommand):
    """Control the irrigation pump with unlimited mode support"""
    global manual_pump_session, unlimited_pump_session

    try:
        if command.action == "open":
            if command.duration is None:
                # UNLIMITED MODE
                unlimited_pump_session = {
                    "active": True,
                    "start_time": datetime.now(),
                    "total_runtime_seconds": 0
                }
                manual_pump_session = {"active": False, "duration_seconds": 0, "start_time": None}

                result = await valve_controller.open_valve(86400)  # 24 hours max

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
                # TIMED MODE
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
    """Get current pump status"""
    await valve_controller.get_valve_status()
    return {
        "pump_status": valve_status,
        "controller_ip": VALVE_CONTROLLER_IP,
        "last_checked": datetime.now().isoformat()
    }

@app.get("/pump/unlimited-session")
async def get_unlimited_pump_session():
    """Get current unlimited pump session info"""
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
    """Get current manual pump session info"""
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
    """Update valve controller IP configuration"""
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
    """Configure farm parameters for irrigation calculations"""
    global irrigation_config
    irrigation_config = config.dict()
    logger.info(f"Irrigation config updated: {config.farm_name} - {config.crop_type}")
    return {"status": "success", "message": "Irrigation configuration updated", "config": irrigation_config}

@app.get("/irrigation/config")
async def get_irrigation_config():
    """Get current irrigation configuration"""
    return irrigation_config

@app.get("/irrigation/latest")
async def get_latest_irrigation():
    """Get latest irrigation recommendation"""
    if irrigation_history:
        return irrigation_history[-1]
    return {"message": "No irrigation recommendations available"}

@app.get("/irrigation/history")
async def get_irrigation_history():
    """Get irrigation recommendation history"""
    return list(irrigation_history)

# =========================
# Debug Endpoints
# =========================
@app.get("/debug/latest-data")
async def debug_latest_data():
    """Quick debug to see latest data"""
    return {
        "latest_prediction": prediction_history[-1] if prediction_history else "No predictions",
        "cycle_data": cycle_data,
        "irrigation_history_latest": irrigation_history[-1] if irrigation_history else "No irrigation data",
        "prediction_count": len(prediction_history),
        "valve_status": valve_status,
        "current_deficit": irrigation_calc.previous_deficit,
        "last_processed_timestamp": last_processed_timestamp,
        "recovery_status": recovery_status
    }

# =========================
# Health & Info Endpoints
# =========================
@app.get("/health")
async def health():
    await valve_controller.get_valve_status()
    return {
        "status": "ok",
        "version": "2.0.0",
        "models_loaded": all(models.values()),
        "irrigation_enabled": bool(irrigation_config),
        "valve_controller": valve_status,
        "influxdb_recovery": INFLUXDB_AVAILABLE,
        "last_processed_timestamp": last_processed_timestamp
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
            "enhancement": "Applied in API with Rainfall Integration, Data Recovery & Fixed Validation"
        },
        "validation_ranges": {
            "temperature": "-50 to 60 C",
            "humidity": "0 to 100 %",
            "wind_speed": "0 to 45 m/s (Rika RK120-01)",
            "sunshine_hours": "0 to 24 h",
            "rainfall": "0 to 500 mm"
        }
    }

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    uvicorn.run("jetson_edge_v2:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
