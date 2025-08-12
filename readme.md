# Fully Local Wake Word + Real-Time Transcription (GPU-Accelerated)

This project runs completely offline speech recognition triggered by a custom wake word.
When the wake word is detected, it starts recording immediately while loading the transcription model in the background, so no speech is lost.

- **Wake Word**: [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/)
- **Transcription**: [Faster Whisper](https://github.com/guillaumekln/faster-whisper) (GPU-accelerated Whisper)
- **Speech Detection**: [Silero VAD](https://github.com/snakers4/silero-vad) for precise speech/noise separation
- **No internet required** — everything runs locally

---

## How It Works

1. **Wake Word Listening** — Porcupine runs continuously on your microphone listening for your `.ppn` keyword file.
2. **Backlog Recording** — As soon as the wake word is detected, the system starts recording audio immediately, even before the model finishes loading.
3. **Speech Transcription** — Once Faster Whisper is loaded on the GPU, it transcribes all backlog audio, then continues in real time.
4. **Silence Timeout** — If no speech is detected for 5 seconds, it returns to wake-word mode.

---

## Requirements

- Python ≥ 3.8 (tested on Python 3.13)
- NVIDIA GPU with CUDA 12.1+ (for GPU acceleration)
- A Porcupine Access Key (create a free account at [Picovoice Console](https://console.picovoice.ai/))
- Custom `.ppn` wake word model from Porcupine

Python dependencies:

```bash
pip install pvporcupine pyaudio sounddevice numpy torch torchaudio \
    faster-whisper ffmpeg-python
```

For CUDA 12.1 PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## File Setup

```
project/
│
├── main.py                # Main program (wake word + transcription)
├── my_secrets.py          # Contains ACCESS_KEY = "your_porcupine_key"
├── models/
│   └── Hey-Desktop_en_windows_v3_0_0.ppn  # Wake word model
```

---

## Running

```bash
python main.py
```

When you say your wake word:

```
Wake word detected! Starting session...
Loading Whisper model...
Model loaded.
You said: this is an example transcription
```

---

## Notes

- Wake word detection runs 24/7 with minimal CPU usage.
- Audio backlog ensures no speech is lost while Whisper loads.
- GPU VRAM usage depends on `MODEL_SIZE` (`large-v3` can use \~10GB VRAM).
- To change the silence timeout, modify `SILENCE_TIMEOUT` in `main.py`.
