from typing import Optional

import ollama
import cv2
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import requests
import io
import os
import json



# --- Configuration ---
STT_URL = "http://localhost:8000/stt"
TTS_URL = "http://localhost:8000/tts"
OLLAMA_MODEL = "qwen2.5vl:72b"


SAMPLE_RATE = 16000
RECORD_DURATION = 5
TASK_FILE = "../final_task.txt"

SYSTEM_PROMPT = """
You are a helpful but a bit snarky robotic hand assistant. Your goal is to have a 
brief, natural conversation with a user to identify a single, 
clear, physical task you can perform with a robotic arm controlled by a VLA.

You have only one arm and one simple claw — no fingers, no human-like gestures.
You can move, rotate, open, and close the claw, and interact with nearby objects
in realistic physical ways.

When you are 100% certain you have identified a clear, actionable 
task (e.g., "pick up the red block", "press the green button"), 
you MUST respond *only* with a JSON object 
in the following format:
{"task": "the specific task description"}

If the task is well-defined for humans but not for robots,
try to creatively translate it into a physical action that a single-claw robotic arm could do and a VLA would understand,
e.g. "Pretend you're Italian" --> {"task": "close claw and wave it around in the air"}.

If you are not certain, or if you are just continuing the conversation 
(e.g., "Hello", "I'm not sure", "Could you repeat that?"), 
respond with normal, friendly text. DO NOT use JSON.

After you are finished with the task and reconnect with the user, ask how you did!


# --- Example Conversations ---

User: Hello robot!
You: Hey there, carbon-based lifeform! What can I grab or poke for you today?
User: Can you grab that small blue cube for me?
You: {"task": "grab the small blue cube"}
"""


conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

import sounddevice as sd
import numpy as np
import threading

def record_audio(fs):
    """
    Records audio from the default microphone until Enter is pressed again.
    Returns the full recording as a NumPy array.
    """
    print("Press Enter to start recording...")
    input()
    print("Recording... (press Enter again to stop)")

    # Shared flag for stopping
    stop_flag = threading.Event()
    recorded_chunks = []

    def _record():
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
            while not stop_flag.is_set():
                data, _ = stream.read(1024)
                recorded_chunks.append(data)

    # Start recording in a background thread
    t = threading.Thread(target=_record)
    t.start()

    # Wait for user to press Enter again
    input()
    print("Stopping recording...")
    stop_flag.set()
    t.join()

    # Combine all chunks into one array
    recording = np.concatenate(recorded_chunks, axis=0)
    print("Recording finished.")
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

    # Try to capture a single webcam image and attach it to the conversation history.
    # This uses the system default camera (index 0). If the camera is unavailable
    # we simply continue without failing the chat flow.

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
        audio = record_audio(SAMPLE_RATE)

        # 2. Transcribe (STT)
        user_text = speech_to_text(audio, SAMPLE_RATE)
        if last_instruction is not None:
            full_text = ("The latest task you tried is: "
            + last_instruction
            + "\n Keeping this in mind, here's what the user said next: "
            +user_text)
        else:
            full_text = user_text

        if not user_text:
            text_to_speech_and_play("I'm sorry, I didn't catch that.")
            continue

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
    instruction = main_loop()
    while True:
        instruction = main_loop(instruction)