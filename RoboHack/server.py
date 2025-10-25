from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import speech_recognition as sr
import concurrent.futures
import pyttsx3
import io
import tempfile
import os
import uvicorn

app = FastAPI(title="RoboHack STT/TTS API")

# Initialize the TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

class TTSRequest(BaseModel):
    text: str

@app.post("/stt", tags=["Speech to Text"])
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert speech to text from an audio file.

    Accepts WAV audio files and returns the transcribed text. Recognition is
    executed in a background thread with a timeout to avoid blocking the
    server if the external recognition service is slow or unreachable.
    """
    temp_audio_path = None
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Load audio data
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)

        # Run recognition in a thread with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(recognizer.recognize_google, audio_data)
            try:
                text = future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise HTTPException(status_code=504, detail="STT recognition timed out")
            except sr.UnknownValueError:
                raise HTTPException(status_code=422, detail="Could not understand the audio")
            except sr.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Speech recognition service error: {e}")

        return {"text": text}

    except HTTPException:
        # Re-raise HTTPExceptions from above
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Ensure temporary file is removed
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass

@app.post("/tts", tags=["Text to Speech"])
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech.
    
    Returns an audio file stream containing the synthesized speech.
    """
    try:
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_path = temp_wav.name

        # Generate speech
        engine.save_to_file(request.text, temp_path)
        engine.runAndWait()

        # Read the generated audio file
        def iterfile():
            with open(temp_path, 'rb') as file:
                yield from file
            os.unlink(temp_path)  # Clean up after streaming

        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={
                'Content-Disposition': 'attachment;filename=speech.wav'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)