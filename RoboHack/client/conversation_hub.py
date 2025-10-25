import ollama
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import requests
import io
import os
import json
import base64
from typing import Optional


# --- Configuration ---
STT_URL = "http://localhost:8000/stt"
TTS_URL = "http://localhost:8000/tts"
OLLAMA_MODEL = "qwen2.5vl:7b"


SAMPLE_RATE = 16000
RECORD_DURATION = 5
TASK_FILE = "../final_task.txt"

SYSTEM_PROMPT = """
You are a helpful robotic hand assistant. Your goal is to have a 
brief, natural conversation with a user to identify a single, 
clear, physical task you can perform. 

When you are 100% certain you have identified a clear, actionable 
task (e.g., "pick up the red block", "pass me the screwdriver", 
"wave goodbye"), you MUST respond *only* with a JSON object 
in the following format:
{"task": "the specific task description"}

If you are not certain, or if you are just continuing the conversation 
(e.g., "Hello", "I'm not sure", "Could you repeat that?"), 
respond with normal, friendly text. DO NOT use JSON.

Example conversation:
User: Hello robot!
You: Hello! How can I help you today?
User: Can you grab that small blue cube for me?
You: {"task": "grab the small blue cube"}
"""

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def attach_image_to_history(image_path: str) -> None:
    """Read an image file and append a base64-encoded placeholder to the conversation history.

    This is a conservative, non-breaking enhancement: it does not attempt to render or upload
    full multimodal messages to Ollama (which may not be enabled). Instead it stores the
    image inline as base64 so a downstream VLM or human operator can access it.
    """
    try:
        with open(image_path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("ascii")
        # Insert a short marker message that an image was provided
        conversation_history.append({"role": "user", "content": f"[image_base64:{b64}]"})
        print(f"Attached image '{image_path}' to conversation history.")
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
    except Exception as e:
        print(f"Failed to attach image: {e}")

def record_audio(duration, fs):
    """Records audio from the default microphone."""
    print("Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to complete
    print("Finished listening.")
    return recording

def speech_to_text(audio_data, fs):
    """Sends audio data to the STT server and gets text."""
    print("Transcribing audio...")
    try:
        # Save recording to a byte buffer
        wav_buffer = io.BytesIO()
        write(wav_buffer, fs, audio_data)
        wav_buffer.seek(0)

        files = {'audio': ('recording.wav', wav_buffer, 'audio/wav')}
        response = requests.post(STT_URL, files=files, timeout=15)

        if response.status_code == 200:
            text = response.json().get("text")
            print(f"User said: {text}")
            return text
        else:
            print(f"STT Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"STT connection error: {e}")
        return None

def get_ai_response(user_text):
    """Sends text to Ollama and gets a response."""
    print("Robot is thinking...")

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_text})

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=conversation_history
        )
        ai_text = response['message']['content']

        # Add AI response to history
        conversation_history.append({"role": "assistant", "content": ai_text})

        return ai_text
    except Exception as e:
        print(f"Ollama error: {e}")
        return "I'm sorry, I'm having trouble thinking right now."

def text_to_speech_and_play(text):
    """Sends text to the TTS server, gets audio, and plays it."""
    print(f"Robot says: {text}")
    try:
        response = requests.post(TTS_URL, json={"text": text}, timeout=15)

        if response.status_code == 200:
            audio_data, samplerate = sf.read(io.BytesIO(response.content))
            sd.play(audio_data, samplerate)
            sd.wait()
        else:
            print(f"TTS Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"TTS connection error: {e}")

def main_loop(last_instruction: Optional[str] = None):
    """The main conversation loop."""
    text_to_speech_and_play("Hello! I am ready to help.")

    while True:
        # 1. Listen
        audio = record_audio(RECORD_DURATION, SAMPLE_RATE)

        # 2. Transcribe (STT)
        user_text = speech_to_text(audio, SAMPLE_RATE)

        if not user_text:
            text_to_speech_and_play("I'm sorry, I didn't catch that.")
            continue
        if last_instruction:
            full_text = ("The latest task you tried is: "
            +last_instruction
            +"\n Keeping this in mind, here's what the user said next: "
            +user_text)
        else:
            full_text = user_text
        # 3. Think (Ollama)
        ai_response = get_ai_response(full_text)

        # 4. Check for Task
        try:
            # Try to parse the response as JSON
            task_data = json.loads(ai_response)
            if isinstance(task_data, dict) and "task" in task_data:
                final_task = task_data["task"]
                print(f"--- TASK IDENTIFIED ---")
                print(f"Task: {final_task}")

                # Save the task to a file
                with open(TASK_FILE, "w") as f:
                    f.write(final_task)
                print(f"Task saved to {TASK_FILE}")


                # Confirm task and end
                text_to_speech_and_play(f"Okay, I will {final_task}.")
                print("Conversation ended.")
                break
        except json.JSONDecodeError:
            # It's not JSON, so it's a normal chat response
            # 5. Speak (TTS)
            text_to_speech_and_play(ai_response)
    with open(TASK_FILE, "r") as f:
        task = f.read()
    return task

if __name__ == "__main__":
    while True:
        main_loop()