import os
import sys
import struct
import subprocess
import time
import pvporcupine
import pyaudio

# Import your access key from a separate file
try:
    from my_secrets import ACCESS_KEY
except ImportError:
    print("Please create a my_secrets.py file with your ACCESS_KEY.")
    exit(1)

# Paths
keyword_paths = ['models/Hey-Desktop_en_windows_v3_0_0.ppn']
WHISPER_CPP_PATH = r"C:\path\to\whisper.cpp"
WHISPER_MODEL = r"ggml-base.en.bin"

# Stop conditions
IDLE_TIMEOUT = 5  # seconds without transcription
STOP_PHRASE = "thank you"

# --- Porcupine wake word ---
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=keyword_paths
)

pa = pyaudio.PyAudio()

stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for wake word...")

def run_whisper():
    whisper_cmd = [
        os.path.join(WHISPER_CPP_PATH, "main.exe"),
        "-m", os.path.join(WHISPER_CPP_PATH, WHISPER_MODEL),
        "--capture",      # mic mode
        "-t", "4"         # number of threads
    ]
    
    proc = subprocess.Popen(
        whisper_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    last_activity = time.time()

    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"[Whisper] {line}")
            last_activity = time.time()
            
            if STOP_PHRASE.lower() in line.lower():
                print("Stop phrase detected — stopping Whisper.")
                proc.terminate()
                break
        
        if time.time() - last_activity > IDLE_TIMEOUT:
            print("Idle timeout reached — stopping Whisper.")
            proc.terminate()
            break

    proc.wait()

try:
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

        result = porcupine.process(pcm_unpacked)
        if result >= 0:
            print("Wake word detected! Starting real-time transcription...")
            run_whisper()

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
    print("Exited...")