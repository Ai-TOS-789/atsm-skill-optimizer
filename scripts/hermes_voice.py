#!/usr/bin/env python3
"""Hermes Voice Assistant for Linux.

Listens for voice commands, transcribes with faster-whisper,
executes via Hermes CLI, and responds with text-to-speech.

Integrates with hermes_wake.py for wake word detection when available.

Usage:
    python3 hermes_voice.py              # start listening
    python3 hermes_voice.py --once       # single command
    python3 hermes_voice.py --text       # text input mode
    python3 hermes_voice.py --wake       # use wake word detection (requires hermes_wake.py)
"""



import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- Config ---
WHISPER_MODEL = os.environ.get("HERMES_WHISPER_MODEL", "base")
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
SILENCE_THRESHOLD = 500  # RMS threshold for silence
MIN_AUDIO_SECONDS = 0.5
MAX_AUDIO_SECONDS = 30
SILENCE_DURATION = 1.5  # seconds of silence to stop recording

HERMES_CLI = os.environ.get("HERMES_CLI", "hermes")
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def speak(text: str):
    """Text-to-speech via speech-dispatcher."""
    try:
        subprocess.run(
            ["spd-say", "-w", "-l", "th", text],
            capture_output=True, timeout=30
        )
    except Exception as e:
        print(f"TTS error: {e}", file=sys.stderr)


def record_audio() -> bytes:
    """Record audio from microphone until silence or max duration."""
    print("🎤 Listening...", end="", flush=True)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    
    # Use arecord with silence detection
    cmd = [
        "arecord",
        "-D", "hw:0,0",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "--duration", str(MAX_AUDIO_SECONDS),
        wav_path
    ]
    
    # Start recording
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Monitor for silence (simplified - just wait for user to stop)
    try:
        proc.wait(timeout=MAX_AUDIO_SECONDS)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait()
    
    print(" ✓", flush=True)
    
    # Read the recorded audio
    try:
        with open(wav_path, "rb") as f:
            audio = f.read()
        os.unlink(wav_path)
        return audio
    except Exception:
        return b""


def transcribe(audio_path: str) -> str:
    """Transcribe audio file using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, language="th", beam_size=5)
        
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    except Exception as e:
        print(f"STT error: {e}", file=sys.stderr)
        return ""


def record_and_transcribe() -> str:
    """Record audio and transcribe to text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    
    # Record with arecord - use plughw for compatibility
    cmd = [
        "arecord",
        "-D", "plughw:0,0",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "--duration", str(MAX_AUDIO_SECONDS),
        wav_path
    ]
    
    print("🎤 Listening...", end="", flush=True)
    try:
        subprocess.run(cmd, capture_output=True, timeout=MAX_AUDIO_SECONDS + 5)
    except subprocess.TimeoutExpired:
        pass
    
    print(" done", flush=True)
    
    # Check if file has content
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        return ""
    
    # Transcribe
    text = transcribe(wav_path)
    os.unlink(wav_path)
    return text


def execute_hermes(command: str) -> str:
    """Execute command via Hermes CLI and return response."""
    print(f"🤖 Executing: {command}")
    
    try:
        result = subprocess.run(
            [HERMES_CLI, "chat", "-q", command],
            capture_output=True, text=True,
            timeout=120,
            env={**os.environ, "HERMES_HOME": str(HERMES_HOME)}
        )
        
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            return f"Error: {error[:200]}"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Execution error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Hermes Voice Assistant")
    parser.add_argument("--once", action="store_true", help="Single command mode")
    parser.add_argument("--text", action="store_true", help="Text input mode")
    parser.add_argument("--wake", action="store_true", help="Use wake word detection (via hermes_wake.py)")
    parser.add_argument("--command", "-c", help="Direct command to execute")
    parser.add_argument("--wake-words", nargs="+", default=None, help="Custom wake words")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🤖 Hermes Voice Assistant")
    print("=" * 50)
    
    if args.command:
        response = execute_hermes(args.command)
        print(f"\n{response}\n")
        speak(response)
        return
    
    # Wake word detection mode (via hermes_wake.py)
    if args.wake:
        wake_script = Path(__file__).parent / "hermes_wake.py"
        if not wake_script.exists():
            print("❌ hermes_wake.py not found. Falling back to continuous mode.")
        else:
            print("🔊 Using wake word detection\n")
            cmd = [sys.executable, str(wake_script)]
            if args.wake_words:
                for w in args.wake_words:
                    cmd.extend(["--wake-word", w])
            if args.once:
                cmd.append("--once")
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
            return
    
    if args.text:
        print("\nText mode (type 'quit' to exit):")
        while True:
            try:
                text = input("\n📝 > ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if not text:
                    continue
                
                response = execute_hermes(text)
                print(f"\n{response}\n")
                speak(response)
            except (EOFError, KeyboardInterrupt):
                break
        return
    
    if args.once:
        text = record_and_transcribe()
        if text:
            print(f"\n🗣️  You said: {text}\n")
            response = execute_hermes(text)
            print(f"\n{response}\n")
            speak(response)
        else:
            print("No speech detected")
        return
    
    # Continuous listening mode
    print("\n🎤 Continuous mode (Ctrl+C to exit)")
    print("Speak after the 'Listening...' prompt\n")
    
    while True:
        try:
            text = record_and_transcribe()
            if text:
                print(f"\n🗣️  You said: {text}\n")
                response = execute_hermes(text)
                print(f"\n{response}\n")
                speak(response)
                print("\n" + "-" * 40)
            else:
                print("No speech detected, try again...")
            
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
