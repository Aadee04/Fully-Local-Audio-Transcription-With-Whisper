# This is a simple microphone volume tester script.
# It uses the sounddevice library to capture audio input and prints the volume level.

import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume_norm = int(np.linalg.norm(indata) * 10)
    print("Volume:", volume_norm * "#")

with sd.InputStream(callback=callback):
    print("Speak into mic — press Ctrl+C to quit")
    import time
    while True:
        time.sleep(0.1)
