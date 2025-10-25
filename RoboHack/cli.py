"""Command-line interface to run STT and TTS features from the terminal.

Usage examples:
  python cli.py stt                # start interactive recording (SPACE+R)
  python cli.py stt --file audio.wav   # transcribe an existing WAV file

  python cli.py tts --text "Hello"     # speak text immediately
  python cli.py tts --file message.txt  # read text from file and speak
  python cli.py tts --text "Hi" --save out.wav  # save spoken audio to WAV

This script expects to be run from the same directory that contains
`stt.py` and `tts.py` (the RoboHack package directory).
"""
import argparse
import sys
from pathlib import Path

try:
    import stt
    import tts
except Exception:
    # If this file is executed from project root, adjust sys.path
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import stt
    import tts

import speech_recognition as sr


def run_stt(args):
    """Run speech-to-text. Either transcribe a file or do an interactive recording."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return 1
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(file_path)) as source:
            audio_data = recognizer.record(source)
        text = stt.transcribe(audio_data)
        print("\n🗣️ Transcription:", text)
        return 0

    # Interactive recording using keyboard shortcut (SPACE + R)
    audio = stt.record_audio()
    print("\nTranscribing...")
    text = stt.transcribe(audio)
    print("\n🗣️ Transcription:", text)
    return 0


def run_tts(args):
    """Run text-to-speech. Speak provided text or text from a file. Optionally save to WAV."""
    engine = tts.init_tts_engine()

    text = args.text
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return 1
        text = file_path.read_text(encoding='utf-8')

    if not text:
        print("No text provided. Use --text or --file to specify text to speak.")
        return 1

    if args.save:
        out = Path(args.save)
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        print(f"Saved spoken audio to {out}")
    else:
        tts.speak_text(engine, text)

    return 0


def main():
    parser = argparse.ArgumentParser(description="RoboHack CLI for STT and TTS")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # STT subcommand
    p_stt = sub.add_parser('stt', help='Speech-to-text commands')
    p_stt.add_argument('--file', '-f', help='Path to WAV file to transcribe')
    p_stt.set_defaults(func=run_stt)

    # TTS subcommand
    p_tts = sub.add_parser('tts', help='Text-to-speech commands')
    p_tts.add_argument('--text', '-t', help='Text to speak')
    p_tts.add_argument('--file', '-f', help='Path to text file to read and speak')
    p_tts.add_argument('--save', '-s', help='Path to save synthesized WAV (optional)')
    p_tts.set_defaults(func=run_tts)

    args = parser.parse_args()
    try:
        return_code = args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return_code = 130

    sys.exit(return_code)


if __name__ == '__main__':
    main()
