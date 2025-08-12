import pvporcupine
import pyaudio
import struct
import torch
import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import tempfile
import os
import subprocess
import time
import threading
from collections import deque

# ===== Wake Word Config =====
try:
    from my_secrets import ACCESS_KEY
except ImportError:
    print("Please create a my_secrets.py file with your ACCESS_KEY.")
    exit(1)

keyword_paths = ['models/Hey-Desktop_en_windows_v3_0_0.ppn']

# ===== Whisper + VAD Config =====
MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
BLOCK_DURATION = 3
SILENCE_TIMEOUT = 5.0  # seconds after last speech to stop Whisper

# ===== Load VAD =====
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils

# ===== Wake Word Init =====
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=keyword_paths
)
pa = pyaudio.PyAudio()
wake_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

def record_audio_to_buffer(stop_event, buffer_queue):
    """Continuously record audio into a buffer queue."""
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        buffer_queue.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        callback=audio_callback,
                        blocksize=int(SAMPLE_RATE * BLOCK_DURATION)):
        while not stop_event.is_set():
            sd.sleep(50)  # small sleep to yield CPU

def process_audio_stream(model, buffer_queue, stop_event):
    """Process audio from buffer queue in real-time until silence timeout."""
    last_speech_time = time.time()

    while not stop_event.is_set():
        if buffer_queue:
            audio_chunk = buffer_queue.popleft()
        else:
            if time.time() - last_speech_time > SILENCE_TIMEOUT:
                print("Silence detected. Returning to wake word mode.")
                break
            time.sleep(0.05)
            continue

        wav_tensor = torch.from_numpy(audio_chunk).float().squeeze()
        speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=SAMPLE_RATE)

        if not speech_timestamps:
            continue  # skip silence/noise

        last_speech_time = time.time()

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

def run_whisper_session():
    print("Wake word detected! Starting session...")

    # Shared queue for recorded audio
    buffer_queue = deque()

    # Start recording immediately
    stop_event = threading.Event()
    record_thread = threading.Thread(target=record_audio_to_buffer, args=(stop_event, buffer_queue))
    record_thread.start()

    # Load model in parallel
    print("Loading Whisper model...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    print("Model loaded.")

    # Process backlog + continue in real-time
    process_audio_stream(model, buffer_queue, stop_event)

    # Stop recording
    stop_event.set()
    record_thread.join()

print("Listening for wake word...")
try:
    while True:
        pcm = wake_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
        result = porcupine.process(pcm_unpacked)

        if result >= 0:
            run_whisper_session()

except KeyboardInterrupt:
    print("Stopping...")

finally:
    wake_stream.stop_stream()
    wake_stream.close()
    pa.terminate()
    porcupine.delete()
