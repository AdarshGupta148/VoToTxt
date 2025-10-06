import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import whisper
import tempfile
import time
# -----------------------------
# CONFIG
# -----------------------------
DURATION = 10      # seconds (you can change this in UI too)
SAMPLE_RATE = 16000
FFMPEG_PATH = r"C:\ffmpeg\bin"

# Add ffmpeg to PATH
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎤 Voice to Text using Whisper")
st.write("Record your voice and transcribe it using OpenAI's Whisper model.")

# User input controls
duration = st.slider("Recording Duration (seconds)", 5, 30, DURATION)
model_choice = st.selectbox("Choose Whisper Model", ["tiny", "base", "small", "medium", "large"])

# Record button
if st.button("🎙️ Record Audio"):
    st.info(f"Recording for {duration} seconds...")
    with st.spinner("Recording..."):
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        
        # Save to temp file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, recording, SAMPLE_RATE)
        st.success(f"✅ Recording saved: {temp_wav.name}")
        st.audio(temp_wav.name)

        # -----------------------------
        # Loading Whisper model with progress bar
        # -----------------------------
        st.info("Loading Whisper model... please wait ⏳")
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for percent_complete in range(100):
            time.sleep(0.03)  # smooth animation
            progress_bar.progress(percent_complete + 1)
            progress_text.text(f"Model loading... {percent_complete + 1}%")

        model = whisper.load_model(model_choice)
        progress_bar.empty()
        progress_text.empty()
        st.success("✅ Whisper model loaded successfully!")

        # -----------------------------
        # Transcribe audio
        # -----------------------------
        st.info("Transcribing... This may take a few moments ⏳")
        result = model.transcribe(temp_wav.name)

        # Display output
        st.subheader("📝 Transcribed Text:")
        st.write(result["text"])

        # Option to download transcription
        st.download_button(
            label="📥 Download Transcription as TXT",
            data=result["text"],
            file_name="transcription.txt",
            mime="text/plain"
        )

