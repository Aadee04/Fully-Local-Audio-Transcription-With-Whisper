import sounddevice as sd
import numpy as np
from openwakeword.model import Model

# Force ONNXRuntime by telling Model not to expect TFLite
oww_model = Model(wakeword_models=["hey_mycroft"], inference_framework="onnx")

# Other included wake words: "hey_mycroft", "alexa", "hey_firefox"

samplerate = 16000  # openwakeword expects 16k mono
blocksize = 512

with sd.InputStream(
    samplerate=samplerate,
    channels=1,
    dtype='int16',
    blocksize=blocksize
) as stream:
    print("Listening for wake word... (ONNX mode)")
    while True:
        audio_block, _ = stream.read(blocksize)
        audio_block = audio_block.flatten().astype(np.float32) / 32768.0  # int16 → float32 [-1, 1]

        detection_scores = oww_model.predict(audio_block)
        for key, score in detection_scores.items():
            if score > 0.5:  # adjust threshold for sensitivity
                print(f"Wake word '{key}' detected! (score: {score:.2f})")
