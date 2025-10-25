import ollama
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import requests
import io
import os
import json
import base64
import asyncio
from datetime import datetime
from typing import Optional
from RoboHack.common.event_router import EventRouter


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


# Optional in-process EventRouter instance. Call `set_event_router(router)` from the
# main program to receive turn-status updates from this module.
_event_router: EventRouter | None = None
# Event loop associated with the EventRouter. When set via `set_event_router`
# callers can optionally provide the asyncio loop that will service router
# operations. This allows synchronous code running on other threads to publish
# safely using `asyncio.run_coroutine_threadsafe`.
_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_router(router: EventRouter, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Set a shared EventRouter instance to publish status updates.

    If `loop` is provided it will be used for cross-thread publishing via
    `asyncio.run_coroutine_threadsafe`. If omitted the helper will attempt
    reasonable fallbacks.
    """
    global _event_router, _event_loop
    _event_router = router
    if loop is not None:
        _event_loop = loop
    else:
        # Try to capture a running loop if present; otherwise leave None.
        try:
            _event_loop = asyncio.get_running_loop()
        except RuntimeError:
            _event_loop = None


def get_event_router() -> tuple[EventRouter | None, asyncio.AbstractEventLoop | None]:
    """Return the configured EventRouter and its associated event loop (if any).

    This is a small helper so other modules (e.g., robot clients) can subscribe
    to shared topics when a router has been installed via `set_event_router`.
    """
    return _event_router, _event_loop


def _publish_turn_status(turn: str, message: str | None = None) -> None:
    """Publish a small status message to topic 'turn_status'.

    Behavior:
      - If `_event_router` is not set: no-op.
      - If called from within an asyncio loop: schedule a task on that loop.
      - If called from a different thread and `_event_loop` is set and running:
        use `run_coroutine_threadsafe` to publish into that loop.
      - Otherwise fall back to running a short-lived asyncio.run() to publish.

    The payload contains `turn`, `message`, and an ISO timestamp.
    """
    if _event_router is None:
        return

    payload = {
        "turn": turn,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread: try to use the recorded event loop
        if _event_loop is not None and _event_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(_event_router.publish("turn_status", payload), _event_loop)
                return
            except Exception:
                # fall back to running a new loop
                pass
        # Fallback: create a short-lived loop to publish
        try:
            asyncio.run(_event_router.publish("turn_status", payload))
        except Exception:
            # best-effort: swallow exceptions in status publishing
            return
    else:
        # We are inside an asyncio loop: schedule the publish
        try:
            loop.create_task(_event_router.publish("turn_status", payload))
        except Exception:
            try:
                asyncio.ensure_future(_event_router.publish("turn_status", payload))
            except Exception:
                # swallow any publish failures
                return

def record_audio(duration, fs):
    """Records audio from the default microphone."""
    press = True
    while press:
        key = input("Press enter to start listening....")
        if key == "": press = False
        pass
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
    _publish_turn_status("idle", "greeting_played")

    while True:
        # 1. Listen
        _publish_turn_status("user", "listening_start")
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
        _publish_turn_status("assistant", "thinking")
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
                _publish_turn_status("assistant", "task_identified")
                text_to_speech_and_play(f"Okay, I will {final_task}.")
                print("Conversation ended.")
                break
        except json.JSONDecodeError:
            # It's not JSON, so it's a normal chat response
            # 5. Speak (TTS)
            _publish_turn_status("assistant", "speaking")
            text_to_speech_and_play(ai_response)
    with open(TASK_FILE, "r") as f:
        task = f.read()
    _publish_turn_status("idle", "conversation_end")
    return task

if __name__ == "__main__":
    while True:
        main_loop()