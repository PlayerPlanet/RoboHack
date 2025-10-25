import speech_recognition as sr
import keyboard
import time

def record_audio():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("Press and hold SPACE + R to start recording...")
    while True:
        if keyboard.is_pressed('space') and keyboard.is_pressed('r'):
            break
        time.sleep(0.1)
    
    print("🎤 Recording... (Release keys to stop)")
    
    with microphone as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Record until keys are released
        audio = recognizer.listen(source)
    
    print("✅ Recording complete!")
    return audio

def transcribe(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Could not request results from the speech recognition service"

# ---- Step 3: Run Everything ----
if __name__ == "__main__":
    while True:
        try:
            audio = record_audio()
            print("\nTranscribing...")
            text = transcribe(audio)
            print("\n🗣️ Transcription:", text)
            print("\nPress Ctrl+C to exit or press SPACE + R to record again...")
        except KeyboardInterrupt:
            print("\nExiting...")
            break