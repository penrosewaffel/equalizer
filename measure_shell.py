import subprocess
import tempfile
import numpy as np
import wave
import os

# This was aided by ChatGPT 4

def measure(signal):
    """
    Measure a signal by calling an according script.
    """

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_wav, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_wav:
        input_path = input_wav.name
        output_path = output_wav.name
    
        write_wav_file(input_path, signal.T, 48000, 2)

        try:
            print("Calling script")

            script_path="./measure.sh"

            try:

                result=subprocess.run(
                    ["bash",script_path,input_path,output_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )

                print("shell stdout:", result.stdout.decode())
                print("shell stderr:", result.stderr.decode())
            except subprocess.CalledProcessError as e:
                print("Error in script:", e.stderr.decode())
                
            # Read the result
            result = read_wav_file(output_path).T

        finally:
            
            # delete temprary files
            
            os.unlink(input_path)
            os.unlink(output_path)

    return result

def write_wav_file(path, signal, samplerate, channels):
    """Save a signal as a WAV file"""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes((signal * 32760).astype(np.int16).tobytes())

def read_wav_file(path):
    """Load a signal from a WAV file"""
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        signal = np.frombuffer(frames, dtype=np.int16) / 32767
        return signal
