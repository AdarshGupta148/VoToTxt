import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import whisper
import tempfile
import numpy as np
import time
from transformers import pipeline

#CONFIG
DURATION = 10
SAMPLE_RATE = 16000
FFMPEG_PATH = r"C:\\ffmpeg\\bin"

os.environ["PATH"] += os.pathsep + FFMPEG_PATH


#STREAMLIT PAGE SETTINGS
st.set_page_config(page_title="VoTXT - Voice to Text + Summarizer", layout="centered")
st.title("🎙️ VoTXT - Voice to Text + Summarization")
st.write("Record, upload, and transcribe your voice instantly using Whisper + Transformer Summarizer.")

#SESSION STATE VARIABLES
if "recording" not in st.session_state:
    st.session_state.recording = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "mode" not in st.session_state:
    st.session_state.mode = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "recording_data" not in st.session_state:
    st.session_state.recording_data = np.array([])
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

 
#SIDEBAR SETTINGS 
st.sidebar.header("Settings ⚙️")
duration = st.sidebar.slider("Recording Duration (seconds)", 5, 180, DURATION)
model_choice = st.sidebar.selectbox("Choose Whisper Model", ["tiny", "base", "small", "medium", "large"])
max_summary_words = st.sidebar.slider("Summary Length (words)", 30, 200, 100)
st.sidebar.markdown("---")

 
#MODE SELECTION 
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Record Mode"):
        st.session_state.mode = "record"
        st.session_state.recording = False
        st.session_state.paused = False
        st.session_state.transcribed_text = ""
        st.session_state.recording_data = np.array([])
        st.session_state.summary_text = ""
with col2:
    if st.button("📤 Upload Mode"):
        st.session_state.mode = "upload"
        st.session_state.recording = False
        st.session_state.paused = False
        st.session_state.transcribed_text = ""
        st.session_state.recording_data = np.array([])
        st.session_state.summary_text = ""

 
#RECORD MODE 
if st.session_state.mode == "record":
    st.subheader("🎙️ Record Audio")

    #Record / Pause / Resume buttons
    col_rec, col_pause, col_reset = st.columns(3)
    with col_rec:
        if st.button("▶️ Start Recording"):
            st.session_state.recording = True
            st.session_state.paused = False
            st.session_state.recording_data = np.array([])
            st.info(f"Recording for {duration} seconds...")
            with st.spinner("Recording..."):
                recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
                sd.wait()
                st.session_state.recording_data = recording
            st.session_state.recording = False

    with col_pause:
        if st.session_state.recording and not st.session_state.paused:
            if st.button("⏸️ Pause"):
                sd.stop()
                st.session_state.paused = True
                st.warning("Recording paused.")
        elif st.session_state.paused:
            if st.button("▶️ Resume"):
                st.session_state.paused = False
                st.info("Recording resumed.")
                remaining_time = duration - len(st.session_state.recording_data) / SAMPLE_RATE
                if remaining_time > 0:
                    new_part = sd.rec(int(remaining_time * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
                    sd.wait()
                    st.session_state.recording_data = np.concatenate((st.session_state.recording_data, new_part))
                st.session_state.recording = False

    with col_reset:
        if st.button("🔄 Reset"):
            st.session_state.recording = False
            st.session_state.paused = False
            st.session_state.recording_data = np.array([])
            st.session_state.transcribed_text = ""
            st.session_state.summary_text = ""
            st.success("Recording reset!")

    #After recording
    if st.session_state.recording_data.size > 0:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, st.session_state.recording_data, SAMPLE_RATE)
        st.audio(temp_wav.name)
        st.success("✅ Recording complete!")

        #Transcribe
        st.info("Loading Whisper model... please wait ⏳")
        model = whisper.load_model(model_choice)
        st.info("Transcribing your audio...")

        result = model.transcribe(temp_wav.name)
        st.session_state.transcribed_text = result["text"]
        st.success("✅ Transcription complete!")

    #Show transcription
    if st.session_state.transcribed_text:
        st.subheader("📝 Transcribed Text:")
        st.write(st.session_state.transcribed_text)
        st.download_button(
            label="📥 Download as TXT",
            data=st.session_state.transcribed_text,
            file_name="transcription.txt",
            mime="text/plain"
        )

         
        #SUMMARY GENERATOR         
        st.markdown("---")
        st.subheader("🧠 Generate Summary")
        if st.button("✨ Summarize Text"):
            with st.spinner("Generating summary using transformer model... ⏳"):
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                #Truncate text to avoid token overflow
                input_text = st.session_state.transcribed_text[:3000]
                summary = summarizer(
                    input_text,
                    max_length=max_summary_words,
                    min_length=max(20, max_summary_words // 2),
                    do_sample=False
                )[0]["summary_text"]
                st.session_state.summary_text = summary
            st.success("✅ Summary generated!")

        if st.session_state.summary_text:
            st.write("###🧾 Summary:")
            st.write(st.session_state.summary_text)
            st.download_button(
                label="📥 Download Summary as TXT",
                data=st.session_state.summary_text,
                file_name="summary.txt",
                mime="text/plain"
            )

 
#UPLOAD MODE 
elif st.session_state.mode == "upload":
    st.subheader("📤 Upload a Recorded Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        st.info("Transcribing your audio... please wait ⏳")

        #Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        model = whisper.load_model(model_choice)
        result = model.transcribe(temp_path)
        st.session_state.transcribed_text = result["text"]

        st.success("✅ Transcription complete!")

    if st.session_state.transcribed_text:
        st.subheader("📝 Transcribed Text:")
        st.write(st.session_state.transcribed_text)
        st.download_button(
            label="📥 Download as TXT",
            data=st.session_state.transcribed_text,
            file_name="transcription.txt",
            mime="text/plain"
        )

         
        #SUMMARY GENERATOR         
        st.markdown("---")
        st.subheader("🧠 Generate Summary")
        if st.button("✨ Summarize Text (Upload Mode)"):
            with st.spinner("Generating summary using transformer model... ⏳"):
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                input_text = st.session_state.transcribed_text[:3000]
                summary = summarizer(
                    input_text,
                    max_length=max_summary_words,
                    min_length=max(20, max_summary_words // 2),
                    do_sample=False
                )[0]["summary_text"]
                st.session_state.summary_text = summary
            st.success("✅ Summary generated!")

        if st.session_state.summary_text:
            st.write("###🧾 Summary:")
            st.write(st.session_state.summary_text)
            st.download_button(
                label="📥 Download Summary as TXT",
                data=st.session_state.summary_text,
                file_name="summary.txt",
                mime="text/plain"
            )

 
#DEFAULT VIEW 
else:
    st.info("👆 Choose a mode above to start recording or uploading audio.")
