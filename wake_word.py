import pvporcupine
import pyaudio
import struct


# Import your access key from a separate file
try:
    from my_secrets import ACCESS_KEY
except ImportError:
    print("Please create a my_secrets.py file with your ACCESS_KEY.")
    exit(1)

# Replace with your actual Access Key
access_key = ACCESS_KEY

# Path to your custom wake word model
keyword_paths = ['models/Hey-Desktop_en_windows_v3_0_0.ppn']

# Initialize Porcupine with your custom wake word
porcupine = pvporcupine.create(
    access_key=access_key,
    keyword_paths=keyword_paths
)

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Open the audio stream
stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for wake word...")

try:
    while True:
        # Read audio frame
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

        # Process the audio frame
        result = porcupine.process(pcm_unpacked)
        if result >= 0:
            print("Wake word detected!")
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
