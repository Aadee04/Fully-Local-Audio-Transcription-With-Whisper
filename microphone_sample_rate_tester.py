import sounddevice as sd

# List all devices
print(sd.query_devices())

# Or check the default input device's sample rate
default_device = sd.default.device[0]  # index of default input
info = sd.query_devices(default_device, 'input')
print(f"\nDefault input device: {info['name']}")
print(f"Default sample rate: {info['default_samplerate']} Hz\n")

TARGET_RATE = 16000
device_index = sd.default.device[0]  # default input device index

try:
    # Just try opening the stream at 16k
    with sd.InputStream(samplerate=TARGET_RATE, channels=1, dtype='int16', device=device_index):
        print(f"Device supports {TARGET_RATE} Hz directly")
except Exception as e:
    print(f"Device cannot open at {TARGET_RATE} Hz: {e}")
