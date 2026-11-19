from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "output" / "battery_labels.csv"
GRAPH_PATH = BASE_DIR / "output" / "graphs"

SENSOR_COLUMNS = [
    "meas_v",
    "meas_a",
    "meas_t1",
    "meas_t2",
    "meas_p",
    "atmp_t",
    "thermal_t",
    "meas_ch4",
    "meas_co",
    "meas_co2",
    "meas_hcn",
    "meas_hcl",
    "meas_hf",
    "meas_n2o",
    "meas_no",
    "meas_no2",
    "meas_so2"
]

RANDOM_STATE = 42