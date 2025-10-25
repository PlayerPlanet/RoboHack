import requests
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import os
import time
import keyboard
import tempfile
from pathlib import Path

BASE_URL = "http://localhost:8000"

def record_audio(duration=5, fs=16000):
    """Record audio from microphone"""
    print("🎤 Press and hold SPACE to start recording...")
    while not keyboard.is_pressed('space'):
        time.sleep(0.1)
    
    print("Recording... (Release SPACE to stop)")
    recorded_frames = []
    stream = sd.InputStream(samplerate=fs, channels=1, dtype='float32')
    stream.start()
    
    while keyboard.is_pressed('space'):
        frame, overflowed = stream.read(fs // 10)  # Read 100ms chunks
        if not overflowed:
            recorded_frames.append(frame)
        if len(recorded_frames) * (fs // 10) >= fs * duration:  # Check max duration
            break
    
    stream.stop()
    stream.close()
    
    if not recorded_frames:
        raise ValueError("No audio recorded - recording too short")
        
    audio = np.concatenate(recorded_frames)
    print("✅ Recording complete!")
    
    # Convert to WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        wav_path = temp_wav.name
        audio = (audio * 32767).astype(np.int16)
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(fs)
            wav_file.writeframes(audio.tobytes())
    
    return wav_path

def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code != 200:
            print("❌ Server is not running at", BASE_URL)
            return False
        print("✅ Server is running")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running at", BASE_URL)
        return False

def test_stt():
    """Interactive test of the speech-to-text endpoint"""
    print("\n🎤 Testing Speech-to-Text")
    print("-------------------------")
    
    # Record audio
    wav_path = record_audio()
    print("\nSending to STT service...")
    
    try:
        # Send to STT endpoint
        with open(wav_path, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/stt",
                files={"audio": f}
            )
        
        if response.status_code == 200:
            text = response.json().get('text', '')
            print("\n📝 Transcription:", text)
        else:
            print("❌ Error:", response.json().get('detail', 'Unknown error'))
    
    finally:
        # Cleanup
        if os.path.exists(wav_path):
            os.unlink(wav_path)

def test_tts():
    """Interactive test of the text-to-speech endpoint"""
    print("\n🔊 Testing Text-to-Speech")
    print("-------------------------")
    
    # Get text from user
    print("\nEnter text to speak (press Enter):")
    text = input("> ")
    
    if not text:
        print("No text entered, skipping TTS test")
        return
    
    print("\nSending to TTS service...")
    
    # Send to TTS endpoint
    response = requests.post(
        f"{BASE_URL}/tts",
        json={"text": text}
    )
    
    if response.status_code == 200:
        # Save and play the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav.write(response.content)
            wav_path = temp_wav.name

        print("✅ Playing audio...")
        tts_audio = None
        tts_fs = None
        try:
            # Read WAV file using wave + numpy (sounddevice has no read())
            with wave.open(wav_path, 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                tts_fs = wf.getframerate()
                nframes = wf.getnframes()
                frames = wf.readframes(nframes)

            # Determine numpy dtype from sample width
            if sampwidth == 1:
                dtype = np.uint8
            elif sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                dtype = np.int16

            audio_array = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)

            # Normalize integer audio to float32 in [-1.0, 1.0] for sounddevice
            if np.issubdtype(dtype, np.integer):
                max_val = float(2 ** (8 * sampwidth - 1))
                tts_audio = audio_array.astype(np.float32) / max_val
            else:
                tts_audio = audio_array.astype(np.float32)

            # Play the audio and wait until finished
            sd.play(tts_audio, tts_fs)
            sd.wait()

        except Exception as e:
            print("❌ Playback failed:", e)
        finally:
            # Cleanup temp file
            if os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
    else:
        print("❌ Error:", response.json().get('detail', 'Unknown error'))

def main():
    """Run interactive tests for STT and TTS endpoints"""
    print("🧪 Interactive API Testing")
    print("=========================")
    
    if not check_server():
        print("\nPlease start the server first:")
        print("cd RoboHack\\RoboHack")
        print("& ..\\..\\venv\\Scripts\\python.exe server.py")
        return
    
    while True:
        print("\nSelect a test:")
        print("1. Test Speech-to-Text (record and transcribe)")
        print("2. Test Text-to-Speech (type and listen)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == '1':
            test_stt()
        elif choice == '2':
            test_tts()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice, please try again")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")