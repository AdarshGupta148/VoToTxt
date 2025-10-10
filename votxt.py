import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import time
import whisper
from transformers import pipeline
import uuid

# CONFIG
SAMPLE_RATE = 16000
MAX_DURATION = 180  # seconds
FFMPEG_PATH = r"C:\ffmpeg\bin"
DEVICE = "cuda"  # or "cpu"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# ---- Streamlit UI ----
st.title("🎙️ Voice to Text + Summarization")
st.write("Record, Pause, Resume, Upload, and Save your voice instantly!")

# ---- Session State Initialization ----
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "pause" not in st.session_state:
    st.session_state.pause = False
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = np.array([], dtype=np.float32)
if "start_time" not in st.session_state:
    st.session_state.start_time = 0
if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = 0
if "uploaded_done" not in st.session_state:
    st.session_state.uploaded_done = False

# Select Whisper model
whisper_model_choice = st.selectbox("Choose Whisper Model", ["tiny", "base", "small", "medium", "large"])

# ---- Upload Audio Section ----
uploaded_audio = st.file_uploader("📤 Upload a Recorded Audio File", type=["wav", "mp3", "m4a"])

def transcribe_and_summarize(file_path):
    """Load Whisper, transcribe audio, summarize text using LLM."""
    st.info("Loading Whisper model...")
    model = whisper.load_model(whisper_model_choice, device=DEVICE)
    st.success("Model loaded!")

    # Transcribe
    st.info("Transcribing audio... ⏳")
    result = model.transcribe(file_path)
    transcript = result["text"]

    st.subheader("Transcribed Text")
    st.write(transcript)
    st.download_button(
        "📥 Download Transcription",
        transcript,
        "transcription.txt",
        key=f"transcription_{uuid.uuid4()}"
    )

    # Summarize
    if len(transcript.split()) < 90:
        st.subheader("Summary")
        st.write("Text is too short to summarize.")
    else:
        st.info("Summarizing text using local LLM... ⏳")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if DEVICE == "cuda" else -1
        )
        summary = summarizer(
            transcript,
            max_length=220,
            min_length=90,
            do_sample=False
        )[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)
        st.download_button(
            "📥 Download Summary",
            summary,
            "summary.txt",
            key=f"summary_{uuid.uuid4()}"
        )

# Handle uploaded audio
if uploaded_audio is not None and not st.session_state.uploaded_done:
    st.audio(uploaded_audio)
    # Save uploaded file temporarily
    temp_uploaded = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_uploaded.write(uploaded_audio.read())
    temp_uploaded.flush()
    transcribe_and_summarize(temp_uploaded.name)
    st.session_state.uploaded_done = True

# ---- Recording Buttons ----
col1, col2, col3, col4 = st.columns(4)
record_btn = col1.button("🎙️ Record", use_container_width=True)
pause_btn = col2.button("⏸️ Pause", use_container_width=True)
resume_btn = col3.button("▶️ Resume", use_container_width=True)
save_btn = col4.button("💾 Save", use_container_width=True)

# Timer display
st.session_state["timer_placeholder"] = st.empty()
st.session_state["timer_placeholder"].text(f"⏱️ Recording: {int(st.session_state.elapsed_time)}s")

# ---- Recording Logic ----
def record_audio():
    """Record audio in small blocks and append to buffer."""
    block_size = 1  # seconds per capture
    while st.session_state.is_recording and not st.session_state.pause:
        audio_block = sd.rec(int(block_size * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        st.session_state.recorded_audio = np.concatenate((st.session_state.recorded_audio, audio_block.flatten()))
        st.session_state.elapsed_time = time.time() - st.session_state.start_time
        st.session_state.elapsed_time = min(st.session_state.elapsed_time, MAX_DURATION)
        st.session_state["timer_placeholder"].text(f"⏱️ Recording: {int(st.session_state.elapsed_time)}s")
        if st.session_state.elapsed_time >= MAX_DURATION:
            st.session_state.is_recording = False
            break

# ---- Record Button ----
if record_btn and not st.session_state.is_recording:
    st.session_state.is_recording = True
    st.session_state.pause = False
    st.session_state.start_time = time.time() - st.session_state.elapsed_time
    with st.spinner("Recording... Speak now! 🎤"):
        record_audio()

# ---- Pause Button ----
if pause_btn and st.session_state.is_recording:
    st.session_state.pause = True
    st.session_state.is_recording = False
    st.info(f"⏸️ Recording paused at {int(st.session_state.elapsed_time)}s")

# ---- Resume Button ----
if resume_btn and not st.session_state.is_recording and st.session_state.pause:
    st.session_state.pause = False
    st.session_state.is_recording = True
    st.session_state.start_time = time.time() - st.session_state.elapsed_time
    st.success("▶️ Resumed recording!")
    record_audio()

# ---- Save Button ----
if save_btn and len(st.session_state.recorded_audio) > 0:
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_wav.name, st.session_state.recorded_audio, SAMPLE_RATE)
    st.audio(temp_wav.name)
    st.success(f"💾 Saved Recording: {temp_wav.name}")
    transcribe_and_summarize(temp_wav.name)
