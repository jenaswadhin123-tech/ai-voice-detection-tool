from pathlib import Path
import tempfile

import joblib
import streamlit as st

from src.inference import predict_with_confidence
from train import load_dataset
from src.model import train_model

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "model.joblib"

st.set_page_config(page_title="Voice Authenticity Check", page_icon="🎙️", layout="centered")
st.title("Voice Authenticity Check")
st.caption("Screen an audio recording for patterns associated with AI-generated speech.")
st.warning("This is a research prototype. A prediction is not proof that a voice is cloned or human.")

with st.sidebar:
    st.header("Model")
    if MODEL_PATH.exists():
        st.success("Trained model loaded")
    else:
        st.info("No trained model yet")

    if st.button("Train model", use_container_width=True):
        try:
            files, features, labels = load_dataset()
            if len(files) == 0:
                st.error("No audio files found in data/sample_audio.")
            else:
                model, accuracy = train_model(features, labels)
                joblib.dump(model, MODEL_PATH)
                st.success(f"Model trained. Validation accuracy: {accuracy:.1%}")
                st.rerun()
        except ValueError as error:
            st.error(str(error))

st.subheader("Record or upload audio")
recorded_audio = st.audio_input("Record a voice sample")
uploaded_audio = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])
audio = recorded_audio or uploaded_audio

if audio:
    st.audio(audio)
    if st.button("Analyze voice", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.error("Train a model first using the sidebar button.")
        else:
            suffix = Path(audio.name).suffix if getattr(audio, "name", None) else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(audio.getvalue())
                temporary_path = temporary_file.name
            try:
                model = joblib.load(MODEL_PATH)
                label, confidence = predict_with_confidence(model, temporary_path)
                if label == "AI":
                    st.error(f"Likely AI-generated: {confidence:.1%} model confidence")
                else:
                    st.success(f"Likely human: {confidence:.1%} model confidence")
                st.progress(confidence, text="Prediction confidence")
            finally:
                Path(temporary_path).unlink(missing_ok=True)
else:
    st.info("Record a sample or choose an audio file to begin.")

st.divider()
st.caption("Training labels use filenames: include 'ai' for AI-generated samples; other files are treated as human.")
