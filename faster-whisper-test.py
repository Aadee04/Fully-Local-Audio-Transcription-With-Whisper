import torch
import torchaudio
import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import tempfile
import os
import subprocess

# Load VAD model
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils


# Config
MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
BLOCK_DURATION = 3

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print("Model loaded.")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

print("Listening with VAD... Press Ctrl+C to stop.")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=int(SAMPLE_RATE * BLOCK_DURATION)):
    try:
        while True:
            audio_chunk = audio_queue.get()
            wav_tensor = torch.from_numpy(audio_chunk).float().squeeze()

            # Check if speech is present
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=SAMPLE_RATE)
            if not speech_timestamps:
                continue  # skip silent/noise-only chunks

            # Save chunk & transcribe
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                wav_path = tmpfile.name
            subprocess.run([
                "ffmpeg", "-y", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-i", "pipe:0", "-ar", str(SAMPLE_RATE), wav_path
            ], input=audio_chunk.tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            segments, _ = model.transcribe(wav_path, language="en")
            text = " ".join([seg.text for seg in segments]).strip()
            if text:
                print(f"You said: {text}")

            os.remove(wav_path)

    except KeyboardInterrupt:
        print("\nStopped.")
