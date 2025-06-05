# streamlit_app.py

import streamlit as st
import tempfile
import time
from pathlib import Path

# Data handling & visualization
import pandas as pd
import numpy as np
import altair as alt

# OpenCV for video / face processing
import cv2

# --- PyTorch / AlexNet for Engagement Model ---
import torch
import pickle
from torchvision import transforms
from torchvision.models.alexnet import AlexNet

# --- Standalone Keras for Emotion Model (.h5) ---
from keras.models import load_model as keras_load_model
from keras.layers import DepthwiseConv2D as KDepthwiseConv2D

# ── CONSTANTS ────────────────────────────────────────────────────────────────────

# Engagement Model: labels and custom index mapping
ENGAGEMENT_TYPES = ["not engaged", "engaged-negative", "engaged-positive"]
ENGAGEMENT_INDEX_MAP = {
    "not engaged": 1,
    "engaged-negative": 3,
    "engaged-positive": 5
}

# Emotion Model: 7 emotion labels
EMOTION_LABELS = [
    "Angry",       # index 0
    "Disgusted",   # index 1
    "Fearful",     # index 2
    "Happy",       # index 3
    "Neutral",     # index 4
    "Sad",         # index 5
    "Surprised"    # index 6
]

# Compute BASE_DIR = parent of this file (so models/ lives at BASE_DIR / "models")
BASE_DIR = Path(__file__).parent.parent


# ── CUSTOM DEPTHWISECONV2D FOR .h5 DESERIALIZATION ──────────────────────────────

class CustomDepthwiseConv2D(KDepthwiseConv2D):
    """
    Subclass of Keras DepthwiseConv2D that strips out any 'groups' kwarg
    during deserialization. This allows loading older H5 models whose config
    includes {"groups": 1}, which standalone Keras cannot handle by default.
    """
    def __init__(self, *args, **kwargs):
        # Remove 'groups' if present
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# ── CACHE / MODEL LOADING ───────────────────────────────────────────────────────

@st.experimental_singleton
def load_engagement_model() -> torch.nn.Module:
    """
    Load the AlexNet‐based engagement model from 'models/alexnet_full_model.pth'.
    If a pickle.UnpicklingError occurs (missing safe global), allowlist AlexNet.
    """
    model_path = BASE_DIR / "models" / "alexnet_full_model.pth"
    if not model_path.exists():
        st.error(
            f"❌ Engagement model not found at:\n  {model_path}\n"
            "Please place 'alexnet_full_model.pth' in the models/ folder and rerun."
        )
        st.stop()

    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
    except pickle.UnpicklingError:
        torch.serialization.add_safe_globals([AlexNet])
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    model.eval()
    return model


# Transformation pipeline for AlexNet (224×224, normalized)
engagement_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@st.experimental_singleton
def load_keras_emotion_model():
    """
    Load the Keras .h5 file from 'models/best_model.h5' using standalone Keras.
    Supplies a custom DepthwiseConv2D that ignores 'groups' in its config.
    """
    model_path = BASE_DIR / "models" / "best_model.h5"
    if not model_path.exists():
        st.error(
            f"❌ Emotion model not found at:\n  {model_path}\n"
            "Please place 'best_model.h5' in the models/ folder and rerun."
        )
        st.stop()

    # Pass custom_objects so that any DepthwiseConv2D(config={"groups":1, ...})
    # falls back to CustomDepthwiseConv2D, which pops "groups" before calling super().
    model = keras_load_model(
        str(model_path),
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
    )
    return model


@st.experimental_singleton
def get_face_cascade() -> cv2.CascadeClassifier:
    """
    Return an OpenCV Haar Cascade classifier for face detection.
    Cached so we don’t reload every time.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("❌ Failed to load Haar Cascade for face detection.")
        st.stop()
    return face_cascade


def preprocess_face_for_emotion(roi_bgr: np.ndarray, target_size=(96, 96)) -> np.ndarray:
    """
    Convert a BGR face crop (OpenCV) → RGB, resize to target_size, normalize [0–1],
    and add a batch dimension → shape (1, H, W, 3).
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(roi_rgb, target_size, interpolation=cv2.INTER_CUBIC)
    face_norm = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_norm, axis=0)


# ── PROCESSING FUNCTIONS: ENGAGEMENT ────────────────────────────────────────────

def process_video_engagement(path: str, frame_skip: int = 1) -> pd.DataFrame:
    """
    Run frame‐by‐frame inference on a video file using the engagement model,
    sampling every `frame_skip` frames to speed up processing.
    Returns DataFrame: ['time_sec', 'engagement_idx', 'engagement_lbl'].
    """
    model = load_engagement_model()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("❌ Could not open the video file for engagement processing.")
        return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times, idxs, lbls = [], [], []
    progress_bar = st.progress(0.0)
    frame_idx = 0

    # We read frames, but only run inference on every `frame_skip`th frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = engagement_transform(img_rgb).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)
                _, pred = torch.max(outputs, 1)
                engagement_label = ENGAGEMENT_TYPES[pred.item()]
                engagement_index = ENGAGEMENT_INDEX_MAP[engagement_label]

            times.append(frame_idx / fps)
            idxs.append(engagement_index)
            lbls.append(engagement_label)

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    progress_bar.empty()

    return pd.DataFrame({
        "time_sec": times,
        "engagement_idx": idxs,
        "engagement_lbl": lbls
    })


def process_camera_engagement(num_frames: int = 200, fps_delay: float = 0.03) -> pd.DataFrame:
    """
    Run real‐time webcam inference for `num_frames` using the engagement model.
    Displays:
      • Live camera feed  
      • Rolling “% engaged” line chart  
      • Running counts text  
    Returns DataFrame: ['time_sec', 'engagement_idx', 'engagement_lbl'].
    """
    model = load_engagement_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam for engagement processing.")
        return pd.DataFrame()

    # Placeholders for live feed, chart, and status
    image_placeholder = st.empty()
    chart_placeholder = st.empty()
    status_text = st.empty()

    times, idxs, lbls = [], [], []
    counts = {e: 0 for e in ENGAGEMENT_TYPES}
    start_time = time.time()
    frame_idx = 0

    with st.spinner("Starting engagement webcam inference..."):
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                status_text.error("⚠️ Lost camera frame during engagement run.")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = engagement_transform(img_rgb).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tensor)
                _, pred = torch.max(outputs, 1)
                engagement_label = ENGAGEMENT_TYPES[pred.item()]
                engagement_index = ENGAGEMENT_INDEX_MAP[engagement_label]
                counts[engagement_label] += 1

            elapsed = time.time() - start_time
            times.append(elapsed)
            idxs.append(engagement_index)
            lbls.append(engagement_label)

            # Display live frame
            image_placeholder.image(
                img_rgb,
                channels="RGB",
                caption=f"Webcam Frame {frame_idx+1}/{num_frames}"
            )

            # Rolling % engaged chart
            df_live = pd.DataFrame({
                "time_sec": times,
                "percent_engaged": [
                    ((counts["engaged-negative"] + counts["engaged-positive"]) / max(1, (i+1))) * 100
                    for i in range(len(times))
                ]
            })
            rolling_chart = (
                alt.Chart(df_live)
                .mark_line(point=True)
                .encode(
                    x=alt.X("time_sec:Q", title="Time (s)"),
                    y=alt.Y("percent_engaged:Q", title="% Engaged So Far")
                )
                .properties(height=250)
            )
            chart_placeholder.altair_chart(rolling_chart, use_container_width=True)

            status_text.text(
                f"Counts → Not engaged: {counts['not engaged']}  |  "
                f"Engaged−: {counts['engaged-negative']}  |  Engaged+: {counts['engaged-positive']}"
            )

            frame_idx += 1
            time.sleep(fps_delay)

    cap.release()
    status_text.success("✅ Engagement webcam run complete.")

    return pd.DataFrame({
        "time_sec": times,
        "engagement_idx": idxs,
        "engagement_lbl": lbls
    })


# ── PROCESSING FUNCTIONS: EMOTION ───────────────────────────────────────────────

def process_video_emotion(path: str, frame_skip: int = 1) -> pd.DataFrame:
    """
    Open a video file, detect faces on each frame, classify emotion on the largest face,
    sampling every `frame_skip` frames. Returns DataFrame: ['time_sec', 'emotion_idx', 'emotion_lbl'].
    """
    model = load_keras_emotion_model()
    face_cascade = get_face_cascade()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("❌ Could not open the video file for emotion processing.")
        return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times, idxs, lbls = [], [], []
    progress_bar = st.progress(0.0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi_bgr = frame[y : y + h, x : x + w]
                face_input = preprocess_face_for_emotion(roi_bgr, target_size=(96, 96))
                preds = model.predict(face_input, verbose=0)
                emotion_index = int(np.argmax(preds[0]))
                emotion_label = EMOTION_LABELS[emotion_index]

                times.append(frame_idx / fps)
                idxs.append(emotion_index)
                lbls.append(emotion_label)

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    progress_bar.empty()

    return pd.DataFrame({
        "time_sec": times,
        "emotion_idx": idxs,
        "emotion_lbl": lbls
    })


def process_camera_emotion(num_frames: int = 200, fps_delay: float = 0.03) -> pd.DataFrame:
    """
    Capture `num_frames` from the webcam, detect largest face each frame, classify emotion,
    draw a bounding box + label on the frame, and display live. Returns DataFrame:
    ['time_sec', 'emotion_idx', 'emotion_lbl'].
    """
    model = load_keras_emotion_model()
    face_cascade = get_face_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam for emotion processing.")
        return pd.DataFrame()

    image_placeholder = st.empty()
    status_text = st.empty()

    times, idxs, lbls = [], [], []
    start_time = time.time()
    frame_idx = 0

    with st.spinner("Starting emotion webcam inference..."):
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                status_text.error("⚠️ Lost camera frame during emotion run.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi_bgr = frame[y : y + h, x : x + w]
                face_input = preprocess_face_for_emotion(roi_bgr, target_size=(96, 96))
                preds = model.predict(face_input, verbose=0)
                emotion_index = int(np.argmax(preds[0]))
                emotion_label = EMOTION_LABELS[emotion_index]

                # Draw bounding box + label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                elapsed = time.time() - start_time
                times.append(elapsed)
                idxs.append(emotion_index)
                lbls.append(emotion_label)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(img_rgb, channels="RGB")
            status_text.text(f"Frame {frame_idx+1}/{num_frames} | Recognized faces: {len(times)}")

            frame_idx += 1
            time.sleep(fps_delay)

    cap.release()
    status_text.success("✅ Emotion webcam run complete.")

    return pd.DataFrame({
        "time_sec": times,
        "emotion_idx": idxs,
        "emotion_lbl": lbls
    })


# ── STREAMLIT UI LAYOUT ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Unified Facial Model App", layout="wide")
st.title("📹 Facial Model Playground: Engagement vs. Emotion")

st.markdown("""
Welcome! This app lets you pick **two** different facial analysis models and run them on either:
1. An uploaded video file  
2. A live webcam feed  

**Models Available**:
- 🟢 **Engagement Model** (`alexnet_full_model.pth`) → classifies each frame as:
  - not engaged (index 1)
  - engaged-negative (index 3)
  - engaged-positive (index 5)

- 🔵 **Emotion Model** (`best_model.h5`) → classifies each detected face into one of 7 emotions:
  `Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.`

Use the sidebar to select a model and input mode. Then you can customize frame sampling or webcam frames before processing.  
Results include time‐series plots, summary statistics, and raw data download.
""")

# ── SIDEBAR: Configuration ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # 1) Choose model
    model_choice = st.radio(
        "Select a model:",
        ("Engagement Model (AlexNet)", "Emotion Model (Keras)")
    )

    # 2) Choose input mode
    input_mode = st.radio(
        "Choose input mode:",
        ("Upload Video File", "Use Webcam Live")
    )

    # 3) If video mode, let user pick frame skip
    if input_mode == "Upload Video File":
        st.subheader("🗜️ Video Sampling")
        frame_skip = st.slider(
            "Process one frame every N frames:",
            min_value=1, max_value=10, value=1, step=1,
            help="Increase N to skip frames and speed up processing"
        )
        st.caption("Higher skip = fewer frames processed (faster).")

    # 4) If webcam mode, let user pick number of frames
    if input_mode == "Use Webcam Live":
        st.subheader("📷 Webcam Settings")
        webcam_frames = st.slider(
            "Number of frames to capture:",
            min_value=50, max_value=500, value=200, step=50
        )
        st.caption("More frames = longer capture but smoother charts.")


# ── MAIN WORKFLOW ───────────────────────────────────────────────────────────────

if input_mode == "Upload Video File":
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4, MOV, AVI, MKV)", 
        type=["mp4", "mov", "avi", "mkv"]
    )
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.info("🔄 Processing uploaded video—please wait…")

        if model_choice.startswith("Engagement"):
            df = process_video_engagement(tmp_path, frame_skip=frame_skip)
            if df.empty:
                st.error("⚠️ Engagement processing failed or no frames found.")
            else:
                # 1) Time‐series line chart: engagement_idx vs. time_sec
                st.subheader("⏱️ Engagement Index Over Time (Video)")
                line_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#1f77b4")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y(
                            "engagement_idx:Q",
                            title="Engagement Index (1=Not, 3=Neg, 5=Pos)",
                            scale=alt.Scale(domain=[0, 6])
                        ),
                        tooltip=["time_sec", "engagement_lbl"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(line_chart, use_container_width=True)

                # 2) Final bar chart: counts per label
                st.subheader("📊 Final Engagement Counts")
                counts = df["engagement_lbl"].value_counts().reindex(ENGAGEMENT_TYPES, fill_value=0)
                df_counts = pd.DataFrame({
                    "engagement": counts.index,
                    "count": counts.values
                })
                bar_chart = (
                    alt.Chart(df_counts)
                    .mark_bar()
                    .encode(
                        x="engagement",
                        y="count",
                        color=alt.Color("engagement", legend=None),
                        tooltip=["count"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(bar_chart, use_container_width=True)

                # 3) Summary statistics
                avg_idx = df["engagement_idx"].mean()
                percent_engaged = (
                    df["engagement_lbl"].isin(["engaged-negative", "engaged-positive"]).mean() * 100
                )
                st.markdown(f"**Average Engagement Index:** {avg_idx:.2f}")
                st.markdown(f"**% of frames engaged** (neg+pos): {percent_engaged:.1f}%")

                # 4) Download CSV + preview
                st.download_button(
                    label="📥 Download Engagement Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="engagement_results.csv",
                    mime="text/csv"
                )
                with st.expander("Show raw engagement data"):
                    st.dataframe(df)

        else:  # Emotion Model
            df = process_video_emotion(tmp_path, frame_skip=frame_skip)
            if df.empty:
                st.error("⚠️ Emotion processing failed or no faces detected.")
            else:
                # 1) Time‐series line chart: emotion_idx vs. time_sec
                st.subheader("⏱️ Emotion Index Over Time (Video)")
                line_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#ff7f0e")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y(
                            "emotion_idx:Q",
                            title="Emotion Index (0=Angry … 6=Surprised)",
                            scale=alt.Scale(domain=[-0.5, 6.5])
                        ),
                        tooltip=["time_sec", "emotion_lbl"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(line_chart, use_container_width=True)

                # 2) Final bar chart: counts per emotion
                st.subheader("📊 Final Emotion Counts")
                counts = df["emotion_lbl"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
                df_counts = pd.DataFrame({
                    "emotion": counts.index,
                    "count": counts.values
                })
                bar_chart = (
                    alt.Chart(df_counts)
                    .mark_bar()
                    .encode(
                        x="emotion",
                        y="count",
                        color=alt.Color("emotion", legend=None),
                        tooltip=["count"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(bar_chart, use_container_width=True)

                # 3) Summary statistics: distribution + most frequent
                total = len(df)
                distribution = (
                    df["emotion_lbl"].value_counts() / total * 100
                ).round(1)
                most_freq = distribution.idxmax()
                most_pct = distribution.max()
                st.markdown("**Emotion Distribution (%)**")
                dist_table = pd.DataFrame({
                    "emotion": distribution.index,
                    "percent (%)": distribution.values
                })
                st.table(dist_table)

                st.markdown(f"**Most Frequent Emotion:** {most_freq} ({most_pct:.1f}%)")

                # 4) Download CSV + preview
                st.download_button(
                    label="📥 Download Emotion Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="emotion_results.csv",
                    mime="text/csv"
                )
                with st.expander("Show raw emotion data"):
                    st.dataframe(df)


elif input_mode == "Use Webcam Live":
    st.info("▶️ Click the button below to start webcam capture.")
    if st.button("▶️ Start Webcam"):
        if model_choice.startswith("Engagement"):
            df = process_camera_engagement(num_frames=webcam_frames, fps_delay=0.03)
            if df.empty:
                st.error("⚠️ Engagement webcam processing failed or no frames found.")
            else:
                st.subheader("⏱️ Engagement Index Over Time (Webcam Run)")
                line_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#1f77b4")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y(
                            "engagement_idx:Q",
                            title="Engagement Index (1=Not, 3=Neg, 5=Pos)",
                            scale=alt.Scale(domain=[0, 6])
                        ),
                        tooltip=["time_sec", "engagement_lbl"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(line_chart, use_container_width=True)

                st.subheader("📊 Final Engagement Counts (Webcam)")
                counts = df["engagement_lbl"].value_counts().reindex(ENGAGEMENT_TYPES, fill_value=0)
                df_counts = pd.DataFrame({
                    "engagement": counts.index,
                    "count": counts.values
                })
                bar_chart = (
                    alt.Chart(df_counts)
                    .mark_bar()
                    .encode(
                        x="engagement",
                        y="count",
                        color=alt.Color("engagement", legend=None),
                        tooltip=["count"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(bar_chart, use_container_width=True)

                # Summary:
                avg_idx = df["engagement_idx"].mean()
                percent_engaged = (
                    df["engagement_lbl"].isin(["engaged-negative", "engaged-positive"]).mean() * 100
                )
                st.markdown(f"**Average Engagement Index:** {avg_idx:.2f}")
                st.markdown(f"**% of frames engaged** (neg+pos): {percent_engaged:.1f}%")

                st.download_button(
                    label="📥 Download Engagement (Webcam) Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="engagement_webcam_results.csv",
                    mime="text/csv"
                )
                with st.expander("Show raw webcam engagement data"):
                    st.dataframe(df)

        else:  # Emotion Model
            df = process_camera_emotion(num_frames=webcam_frames, fps_delay=0.03)
            if df.empty:
                st.error("⚠️ Emotion webcam processing failed or no faces detected.")
            else:
                st.subheader("⏱️ Emotion Index Over Time (Webcam Run)")
                line_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#ff7f0e")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y(
                            "emotion_idx:Q",
                            title="Emotion Index (0=Angry … 6=Surprised)",
                            scale=alt.Scale(domain=[-0.5, 6.5])
                        ),
                        tooltip=["time_sec", "emotion_lbl"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(line_chart, use_container_width=True)

                st.subheader("📊 Final Emotion Counts (Webcam)")
                counts = df["emotion_lbl"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
                df_counts = pd.DataFrame({
                    "emotion": counts.index,
                    "count": counts.values
                })
                bar_chart = (
                    alt.Chart(df_counts)
                    .mark_bar()
                    .encode(
                        x="emotion",
                        y="count",
                        color=alt.Color("emotion", legend=None),
                        tooltip=["count"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(bar_chart, use_container_width=True)

                # Summary distribution
                total = len(df)
                distribution = (df["emotion_lbl"].value_counts() / total * 100).round(1)
                most_freq = distribution.idxmax()
                most_pct = distribution.max()

                st.markdown("**Emotion Distribution (%)**")
                dist_table = pd.DataFrame({
                    "emotion": distribution.index,
                    "percent (%)": distribution.values
                })
                st.table(dist_table)

                st.markdown(f"**Most Frequent Emotion:** {most_freq} ({most_pct:.1f}%)")

                st.download_button(
                    label="📥 Download Emotion (Webcam) Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="emotion_webcam_results.csv",
                    mime="text/csv"
                )
                with st.expander("Show raw webcam emotion data"):
                    st.dataframe(df)