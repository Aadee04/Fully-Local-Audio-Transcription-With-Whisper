# WIP - Not even close to completion

---

# Fully Local Wake Word + Audio Transcription

This is a mini-project that performs **completely local** speech recognition.
It uses a lightweight wake-word detection model to listen for a trigger phrase, and once activated, switches to [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for full transcription.

No cloud APIs. No internet dependency. Just your CPU, microphone, and a couple of models.

> ⚠️ Note: Whisper.cpp can be RAM-hungry depending on the model size you choose.

---

## How It Works

1. **Wake Word Detection** — A low-resource model continuously monitors audio from your microphone for a specific keyword (e.g., _"Hey Jarvis"_).
2. **Full Transcription** — When the wake word is detected, the pipeline hands over to Whisper.cpp for detailed speech-to-text transcription.

---

## Models Used

1. **Wake Word Model** — [OpenWakeWord](https://github.com/dscripka/openWakeWord) (`.tflite` or `.onnx`)
2. **Speech-to-Text Model** — [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) (choose any size: `tiny`, `base`, `small`, etc.)

---

## Requirements (Subject to Change)

- Python ≥ 3.8 (tested on 3.13 with ONNXRuntime)
- `sounddevice`, `numpy`, `onnxruntime` (for ONNX models) or `tflite-runtime` (for TFLite models)
- Compiled Whisper.cpp binary for your platform

---

## Running

-

## Notes

- For the lowest latency, run the wake-word model at 16 kHz mono.
- Whisper.cpp performance scales heavily with CPU power — larger models will be slower and use more RAM.
- Models are **not** downloaded automatically; place them in your project directory or update the paths in `wake_word.py`.
