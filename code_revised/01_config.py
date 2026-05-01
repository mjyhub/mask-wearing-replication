from pathlib import Path
import os


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

SOURCE_ROOT = Path(os.environ.get("RSOS_SOURCE_ROOT", PROJECT_ROOT))
OUTPUT_ROOT = Path(os.environ.get("RSOS_OUTPUT_ROOT", PROJECT_ROOT))

RAW_DATA_DIR = SOURCE_ROOT / "raw_data"
DATA_DIR = OUTPUT_ROOT / "data"
RESULTS_DIR = OUTPUT_ROOT / "results"
MODELS_DIR = OUTPUT_ROOT / "models"

YOUGOV_PATH = RAW_DATA_DIR / "australia.csv"
POLICY_PATH = RAW_DATA_DIR / "OxCGRT_AUS_latest.csv"

MISSING_THRESHOLD = 10781
MANDATE_ROLLING_DAYS = 14
MANDATE_THRESHOLD = 3
TRAIN_TEST_SEED = 20240417
MODEL_SEED = 20240627
TEST_SIZE = 0.2

FACE_MASK_ITEMS = [
    "i12_health_1",
    "i12_health_22",
    "i12_health_23",
    "i12_health_25",
]

PHQ4_ITEMS = [f"PHQ4_{i}" for i in range(1, 5)]
COMORBIDITY_ITEMS = [f"d1_health_{i}" for i in range(1, 14)] + [
    "d1_health_98",
    "d1_health_99",
]

FREQUENCY_MAP = {
    "Always": 5,
    "Frequently": 4,
    "Sometimes": 3,
    "Rarely": 2,
    "Not at all": 1,
}

AGREEMENT_MAP = {
    "7 - Agree": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
    "1 – Disagree": 1,
    "1 - Disagree": 1,
}

DUMMY_COLUMNS = [
    "state",
    "gender",
    "i9_health",
    "employment_status",
    "i11_health",
    "WCRex1",
    "WCRex2",
    "PHQ4_1",
    "PHQ4_2",
    "PHQ4_3",
    "PHQ4_4",
    "d1_comorbidities",
]

MODEL_SPECS = {
    "model_1": {
        "target": "face_mask_behaviour_binary",
        "period": "all",
        "drop": [
            "RecordNo",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "face_mask_behaviour_binary",
            "protective_behaviour_binary",
            "endtime",
        ],
    },
    "model_1a": {
        "target": "face_mask_behaviour_binary",
        "period": "before",
        "drop": [
            "RecordNo",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "face_mask_behaviour_binary",
            "protective_behaviour_binary",
            "endtime",
            "within_mandate_period",
        ],
    },
    "model_1b": {
        "target": "face_mask_behaviour_binary",
        "period": "after",
        "drop": [
            "RecordNo",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "face_mask_behaviour_binary",
            "protective_behaviour_binary",
            "endtime",
            "within_mandate_period",
        ],
    },
    "model_2": {
        "target": "protective_behaviour_binary",
        "period": "all",
        "drop": [
            "RecordNo",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "face_mask_behaviour_binary",
            "protective_behaviour_binary",
            "protective_behaviour_nomask_scale",
            "endtime",
        ],
    },
    "model_2a": {
        "target": "protective_behaviour_binary",
        "period": "before",
        "drop": [
            "RecordNo",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "face_mask_behaviour_binary",
            "protective_behaviour_binary",
            "protective_behaviour_nomask_scale",
            "endtime",
            "within_mandate_period",
        ],
    },
    "model_2b": {
        "target": "protective_behaviour_binary",
        "period": "after",
        "drop": [
            "RecordNo",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "face_mask_behaviour_binary",
            "protective_behaviour_binary",
            "protective_behaviour_nomask_scale",
            "endtime",
            "within_mandate_period",
        ],
    },
}


def ensure_output_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

