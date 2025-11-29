from pathlib import Path

# Base directory of the project (the folder that contains this src/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset paths
CLASS_DATASET_DIR = BASE_DIR / "dataset" / "classification_dataset"
DETECT_DATASET_DIR = BASE_DIR / "dataset" / "object_detection_Dataset"

# Model directories
MODELS_DIR = BASE_DIR / "models"
CLASS_MODELS_DIR = MODELS_DIR / "classification"
DETECT_MODELS_DIR = MODELS_DIR / "detection"

# Reports directory
REPORTS_DIR = BASE_DIR / "reports"
YOLO_CONFIG_DIR = BASE_DIR / "yolo_config"

# Make sure folders exist
CLASS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DETECT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "yolo_predictions").mkdir(parents=True, exist_ok=True)

# Common training params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 15
EPOCHS_TRANSFER = 10   # increase if you have more time
