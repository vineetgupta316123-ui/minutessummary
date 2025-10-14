import streamlit as st
from openai import OpenAI
from pathlib import Path
import os
import subprocess
import shutil


# Load API keys from secrets
openai_api_key = st.secrets.get('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OpenAI API key is missing. Please configure 'OPENAI_API_KEY' in Streamlit secrets.")
    st.stop()

api_key = st.secrets.get('API_KEY')
if not api_key:
    st.error("OpenRouter API key is missing. Please configure 'API_KEY' in Streamlit secrets.")
    st.stop()

# Client for OpenAI Whisper (audio translation)
client = OpenAI(api_key=openai_api_key)

# Client for OpenRouter (Qwen summarization)
client_openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
# Streamlit UI
st.title("🎧 Auto Audio Translator to English (with Chunking)")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "webm", "ogg", "flac"])

# Function to split audio into ~10 min chunks using ffmpeg
def split_audio(audio_path, chunk_length_sec=600):
    chunks = []
    temp_dir = Path("temp_chunks")
    temp_dir.mkdir(exist_ok=True)

    # Run ffmpeg to split audio
    output_pattern = str(temp_dir / "chunk_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-c", "copy",
        output_pattern
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        chunks = list(temp_dir.glob("chunk_*.mp3"))
        if not chunks:
            raise RuntimeError("No chunks created by ffmpeg")
        chunks.sort()  # Ensure chunks are in order
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg error: {e.stderr}")
    return chunks, temp_dir

# Function to generate summary using Qwen model

# Function to generate summary using Qwen model via OpenRouter
def generate_summary(text):
    prompt = (
        "You are an expert summarizer. Generate a point-wise summary of the following meeting transcript "
        "using markdown bullet points. Structure it as:\n"
        "- **Primary Focus**: [Main topic in 1 sentence].\n"
        "- **Key Decisions/Additions**: [Bullet sub-points for decisions, e.g., new programs].\n"
        "- **Discussions/Plans**: [Bullet sub-points for other topics, e.g., website features].\n"
        "- **Action Items**: [Bullet sub-points for tasks, owners if mentioned, deadlines].\n"
        "- **Next Steps**: [Any follow-ups, e.g., meetings].\n\n"
        "Keep each bullet focused and concise (1-2 sentences max). Transcript:\n\n"
        f"{text}"
    )
    try:
        response = client_openrouter.chat.completions.create(
            model="qwen/qwen-2.5-72b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Increased slightly for structured output
            temperature=0.5  # Lowered for more consistent structure
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"Failed to generate summary: {e}")
        return "Summary generation failed."


# Main processing
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    audio_path = Path("temp_audio") / uploaded_file.name
    audio_path.parent.mkdir(exist_ok=True)

    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    st.write(f"File size: {file_size_mb:.2f} MB")
    full_translation = ""

    with st.spinner("Translating..."):
        try:
            if file_size_mb <= 24:
                # Small file — translate directly
                with open(audio_path, "rb") as audio_file:
                    response = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                    full_translation = response.text
            else:
                # Large file — split into chunks and translate each
                st.warning("File is large. Splitting into chunks...")
                chunks, temp_dir = split_audio(audio_path)
                progress_bar = st.progress(0)
                for i, chunk_path in enumerate(chunks):
                    st.write(f"Translating chunk {i + 1}/{len(chunks)}...")
                    with open(chunk_path, "rb") as audio_chunk:
                        response = client.audio.translations.create(
                            model="whisper-1",
                            file=audio_chunk
                        )
                        full_translation += f"\n\n--- Chunk {i + 1} ---\n{response.text}"
                    progress_bar.progress((i + 1) / len(chunks))
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            # Clean up temporary fprompiles
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as e:
                    st.warning(f"Failed to delete temporary audio file: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except OSError as e:
                    st.warning(f"Failed to delete temporary chunk directory: {e}")

    st.subheader("📝 Translated English Text")
    st.text_area("Translation", full_translation.strip(), height=300)

    if full_translation.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(full_translation)
            st.subheader("📋 Meeting Summary")
            st.text_area("Summary", summary, height=150)    
