from ultralytics import YOLO
from pathlib import Path
import shutil

# Project root: .../Aerial Object Classification & Detection
BASE_DIR = Path(__file__).resolve().parents[1]

# Path to YOLO weights (from yolo_train.py)
WEIGHTS_PATH = BASE_DIR / "models" / "detection" / "yolov8n_birddrone" / "weights" / "best.pt"

# Dataset root (Roboflow export)
DETECT_DATASET_DIR = BASE_DIR / "dataset" / "object_detection_Dataset"

# Reports output dir
REPORTS_DIR = BASE_DIR / "reports"
PRED_DIR = REPORTS_DIR / "yolo_predictions"


if __name__ == "__main__":
    print("Using weights:", WEIGHTS_PATH)

    # Load model
    model = YOLO(str(WEIGHTS_PATH))

    # Figure out correct test images folder:
    # Prefer: object_detection_Dataset/test/images
    # Fallback: object_detection_Dataset/test
    test_root = DETECT_DATASET_DIR / "test"
    test_images_dir = test_root / "images"
    if not test_images_dir.exists():
        print("test/images not found, falling back to test/ root.")
        test_images_dir = test_root

    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images folder not found: {test_images_dir}")

    print("Running inference on:", test_images_dir)

    # Clean / recreate output dir
    if PRED_DIR.exists():
        shutil.rmtree(PRED_DIR)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Run YOLO prediction
    results = model.predict(
        source=str(test_images_dir),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(PRED_DIR),
        name=".",
        exist_ok=True
    )

    print("Inference complete.")
    print("Annotated images saved in:", PRED_DIR)
