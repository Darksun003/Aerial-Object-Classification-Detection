import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO

# ---------------------------------------------------
# Paths and model loading
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

CLASS_MODEL_PATH = BASE_DIR / "models" / "classification" / "best_transfer_model.h5"
YOLO_WEIGHTS_PATH = BASE_DIR / "models" / "detection" / "yolov8n_birddrone" / "weights" / "best.pt"


@st.cache_resource
def load_classification_model():
    model = tf.keras.models.load_model(CLASS_MODEL_PATH)
    return model


@st.cache_resource
def load_yolo_model():
    model = YOLO(str(YOLO_WEIGHTS_PATH))
    return model


# ---------------------------------------------------
# Inference helpers
# ---------------------------------------------------
def classify_image(img: Image.Image, threshold: float = 0.5):
    model = load_classification_model()
    img_resized = img.resize((224, 224))

    # Let the model's own Rescaling layer handle normalization
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Drone" if prob >= threshold else "Bird"
    conf = prob if label == "Drone" else (1.0 - prob)
    return label, conf, prob


def detect_objects(img: Image.Image, conf_thres: float = 0.25):
    model = load_yolo_model()
    arr = np.array(img)  # RGB

    results = model.predict(source=arr, imgsz=640, conf=conf_thres, verbose=False)[0]
    annotated_bgr = results.plot()          # BGR numpy array
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_img = Image.fromarray(annotated_rgb)

    # Count classes
    class_ids = [int(box.cls[0].item()) for box in results.boxes] if results.boxes is not None else []
    bird_count = class_ids.count(0)   # assuming 0: bird
    drone_count = class_ids.count(1)  # assuming 1: drone

    return annotated_img, results, bird_count, drone_count


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(
    page_title="Aerial Object AI: Bird vs Drone",
    page_icon="🛩️",
    layout="wide"
)

# Custom minimal styling
st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        opacity: 0.85;
    }
    .result-card {
        padding: 1.2rem 1.4rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(250,250,250,0.08);
        background: rgba(255,255,255,0.02);
    }
    .metric-good {
        color: #00f5c4;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .metric-bad {
        color: #ff4b4b;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="big-title">🛰️ Aerial Object AI: Bird vs Drone</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Upload an aerial image and choose between <b>Classification</b> or <b>YOLOv8 Object Detection</b>. '
    'Perfect for demos, capstone presentations, and experiments.'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

# Sidebar controls
st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Mode",
    ["🔍 Classification", "🎯 YOLOv8 Detection"],
    index=0
)

st.sidebar.markdown("---")

if mode == "🔍 Classification":
    thresh = st.sidebar.slider(
        "Decision threshold (Drone vs Bird)",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.01,
        help="Prob ≥ threshold → Drone, otherwise Bird."
    )
else:
    conf_thres = st.sidebar.slider(
        "Detection confidence threshold",
        min_value=0.1,
        max_value=0.7,
        value=0.25,
        step=0.05,
        help="YOLOv8 confidence threshold for showing boxes."
    )

st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.write("- Classifier: Transfer Learning (MobileNetV2)")
st.sidebar.write("- Detector: YOLOv8n (2 classes: bird, drone)")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For images with **both** bird and drone, use YOLOv8 Detection mode.")

# Main layout
col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Use aerial images containing birds and/or drones."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Upload an image to get started.", icon="📎")
        image = None

with col_right:
    st.markdown("### 🔎 Inference Panel")

    if image is not None:
        if mode == "🔍 Classification":
            if st.button("🚀 Run Classification", use_container_width=True):
                label, conf, raw_prob = classify_image(image, threshold=thresh)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### Result (Image-level Classification)")
                if label == "Bird":
                    st.markdown(f"<span class='metric-good'>🕊️ Prediction: {label}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='metric-bad'>🚁 Prediction: {label}</span>", unsafe_allow_html=True)

                st.write(f"Confidence (chosen class): **{conf * 100:.2f}%**")
                st.write(f"Raw model output (Drone probability): `{raw_prob:.4f}`")
                st.write(f"Decision threshold: `{thresh:.2f}`")

                st.caption(
                    "Note: Classification assumes each image contains a single dominant class "
                    "(either bird or drone). For multi-object scenes, use YOLOv8 Detection."
                )
                st.markdown("</div>", unsafe_allow_html=True)

        else:  # YOLOv8 Detection
            if st.button("🎯 Run YOLOv8 Detection", use_container_width=True):
                annotated_img, results, bird_count, drone_count = detect_objects(image, conf_thres=conf_thres)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### Result (Object Detection)")

                st.image(annotated_img, caption="YOLOv8 detections", use_column_width=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("🕊️ Birds detected", bird_count)
                with m2:
                    st.metric("🚁 Drones detected", drone_count)

                if bird_count > 0 and drone_count > 0:
                    st.success("Scene contains **both bird(s) and drone(s)**.", icon="✅")
                elif bird_count > 0:
                    st.info("Only bird(s) detected in this frame.", icon="🕊️")
                elif drone_count > 0:
                    st.info("Only drone(s) detected in this frame.", icon="🚁")
                else:
                    st.warning("No bird or drone detected with the current threshold.", icon="⚠️")

                st.caption(f"YOLO confidence threshold: `{conf_thres:.2f}`")
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="result-card">
            <b>No image uploaded yet.</b><br/>
            Upload an aerial photo on the left to see predictions here.
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.caption("Built by GV Jayanth • Deep Learning · Computer Vision · Streamlit")
