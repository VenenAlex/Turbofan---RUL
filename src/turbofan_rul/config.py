"""Settings shared by every part of the turbofan RUL pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"  # the bundle the API serves (model + metadata + drift profile)

DATASET = "FD001"
SETTINGS = ["op1", "op2", "op3"]
COLUMNS = ["unit", "cycle", *SETTINGS] + [f"s{i}" for i in range(1, 22)]

# The 14 sensors that drift as an engine wears out; the other 7 barely move in FD001.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

# Symbol and meaning of each sensor, from Table 2 of Saxena et al. (2008),
# "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".
SENSOR_NAMES = {
    "s2": ("T24", "LPC outlet temperature"),
    "s3": ("T30", "HPC outlet temperature"),
    "s4": ("T50", "LPT outlet temperature"),
    "s7": ("P30", "HPC outlet pressure"),
    "s8": ("Nf", "Fan speed"),
    "s9": ("Nc", "Core speed"),
    "s11": ("Ps30", "HPC outlet static pressure"),
    "s12": ("phi", "Fuel flow / Ps30"),
    "s13": ("NRf", "Corrected fan speed"),
    "s14": ("NRc", "Corrected core speed"),
    "s15": ("BPR", "Bypass ratio"),
    "s17": ("htBleed", "Bleed enthalpy"),
    "s20": ("W31", "HPT coolant bleed"),
    "s21": ("W32", "LPT coolant bleed"),
}

RUL_CAP = 125  # above this many cycles left, an engine counts as fully healthy
WINDOW = 10  # cycles in each rolling mean and rolling std
FEATURES = SENSORS + [f"{s}_{stat}" for s in SENSORS for stat in ("mean", "std")]

# Maintenance policy, chosen from the notebook's threshold sweep.
MAINTENANCE_AT = 30  # an engine truly needs service with this many cycles left or fewer
ALERT_THRESHOLD = 40  # raise an alert when the predicted RUL is at or below this
BAND_EDGES = {"Critical": 30, "Warning": 60}  # upper edges; anything above 60 is Healthy

MLFLOW_EXPERIMENT = "turbofan-rul"
REGISTERED_MODEL = "turbofan-rul-xgb"
PRODUCTION_ALIAS = "production"
