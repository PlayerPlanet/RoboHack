import pyttsx3
import keyboard
import time

def init_tts_engine():
    """Initialize the text-to-speech engine with default settings"""
    engine = pyttsx3.init()
    # Set default properties
    engine.setProperty('rate', 150)    # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    return engine

def speak_text(engine, text):
    """Convert text to speech and play it"""
    print("🔊 Speaking...")
    engine.say(text)
    engine.runAndWait()
    print("✅ Done speaking!")

def get_user_input():
    """Get text input from user"""
    print("\nEnter the text you want to speak (press Enter to submit):")
    return input("> ")

# ---- Main Program ----
if __name__ == "__main__":
    print("Text-to-Speech System")
    print("====================")
    print("Controls:")
    print("- Type your text and press Enter to speak")
    print("- Press SPACE + S to speak the last text again")
    print("- Press Ctrl+C to exit")
    print("")

    engine = init_tts_engine()
    last_text = ""

    while True:
        try:
            if keyboard.is_pressed('space') and keyboard.is_pressed('s'):
                if last_text:
                    print("\n🔄 Repeating last text:", last_text)
                    speak_text(engine, last_text)
                    # Wait for keys to be released to prevent multiple triggers
                    while keyboard.is_pressed('space') or keyboard.is_pressed('s'):
                        time.sleep(0.1)
                else:
                    print("\n⚠️ No previous text to repeat!")
                    time.sleep(0.5)
            
            text = get_user_input()
            if text:
                last_text = text
                speak_text(engine, text)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
            time.sleep(1)
