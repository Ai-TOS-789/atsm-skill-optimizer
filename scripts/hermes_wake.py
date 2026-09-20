#!/usr/bin/env python3
"""Wake Word Detection module for Hermes Voice Assistant.

Provides continuous wake word listening with two backends:
1. Porcupine (pvporcupine) - if available, uses proper keyword spotting
2. Energy-based VAD fallback - uses arecord + RMS energy detection

On wake word detection: plays a chime, records command audio,
transcribes, and executes via Hermes CLI.

Usage:
    python3 hermes_wake.py                    # start daemon
    python3 hermes_wake.py --once             # single wake+command cycle
    python3 hermes_wake.py --status           # check daemon status
    python3 hermes_wake.py --stop             # stop daemon
"""

import argparse
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# --- Config ---
WAKE_WORDS_DEFAULT = ["hey hermes", "สวัสดีเฮอร์เมส"]
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
CHUNK_DURATION = 0.5  # seconds per analysis chunk
SILENCE_THRESHOLD = 800  # RMS threshold for speech detection
MIN_SPEECH_CHUNKS = 2  # minimum chunks to consider as speech
MAX_SPEECH_DURATION = 10  # max seconds of speech before forced stop
SILENCE_DURATION = 1.5  # seconds of silence to end command recording
PRE_WAKE_BUFFER = 0.5  # seconds of audio to keep before wake detection

HERMES_CLI = os.environ.get("HERMES_CLI", "hermes")
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
SCRIPT_DIR = Path(__file__).parent.resolve()
PID_FILE = SCRIPT_DIR / ".hermes_wake.pid"
CHIME_FILE = SCRIPT_DIR / "chime.wav"

# --- Audio Utilities ---


def generate_chime(path: Path):
    """Generate a subtle two-tone chime WAV file using stdlib only."""
    sample_rate = 44100
    duration = 0.12
    samples = int(sample_rate * duration)

    # Two-tone chime: A5 (880Hz) then E6 (1319Hz)
    data = []
    split = samples // 2
    for i in range(samples):
        if i < split:
            freq = 880.0
            env = math.sin(math.pi * i / split) * 0.25
        else:
            freq = 1319.0
            env = math.sin(math.pi * (i - split) / (samples - split)) * 0.25
        val = int(32767 * env * math.sin(2 * math.pi * freq * i / sample_rate))
        data.append(max(-32768, min(32767, val)))

    # Pack as signed 16-bit little-endian
    raw = struct.pack(f"<{len(data)}h", *data)

    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(raw)


def play_chime():
    """Play the wake confirmation chime."""
    if not CHIME_FILE.exists():
        generate_chime(CHIME_FILE)
    try:
        subprocess.run(
            ["aplay", "-q", str(CHIME_FILE)],
            timeout=5,
            capture_output=True,
        )
    except Exception:
        pass  # Non-critical


def record_chunk(duration: float, device: str = "plughw:0,0") -> bytes:
    """Record a short audio chunk using arecord. Returns raw PCM bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        cmd = [
            "arecord",
            "-D", device,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "--duration", str(duration),
            wav_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=duration + 2)
        with wave.open(wav_path, "r") as w:
            return w.readframes(w.getnframes())
    except Exception:
        return b""
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def rms_energy(pcm_data: bytes) -> float:
    """Compute RMS energy of 16-bit PCM audio data."""
    if not pcm_data:
        return 0.0
    count = len(pcm_data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", pcm_data[: count * 2])
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / count)


def record_until_silence(
    silence_duration: float = SILENCE_DURATION,
    max_duration: float = MAX_SPEECH_DURATION,
    device: str = "plughw:0,0",
) -> bytes:
    """Record audio until silence is detected or max duration reached.
    Returns raw PCM bytes."""
    chunks = []
    silence_chunks = 0
    max_silence_chunks = int(silence_duration / CHUNK_DURATION)
    max_chunks = int(max_duration / CHUNK_DURATION)

    while len(chunks) < max_chunks:
        chunk = record_chunk(CHUNK_DURATION, device)
        if not chunk:
            break

        energy = rms_energy(chunk)
        chunks.append(chunk)

        if energy < SILENCE_THRESHOLD:
            silence_chunks += 1
            if silence_chunks >= max_silence_chunks and len(chunks) > MIN_SPEECH_CHUNKS:
                break
        else:
            silence_chunks = 0

    return b"".join(chunks)


def save_pcm_to_wav(pcm_data: bytes, path: str):
    """Save raw 16-bit PCM data to a WAV file."""
    with wave.open(path, "w") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_data)


# --- Transcription ---


def transcribe(audio_path: str) -> str:
    """Transcribe audio file using faster-whisper."""
    try:
        from faster_whisper import WhisperModel

        model_name = os.environ.get("HERMES_WHISPER_MODEL", "base")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, language=None, beam_size=5)
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    except Exception as e:
        print(f"STT error: {e}", file=sys.stderr)
        return ""


# --- Wake Word Detection ---


class EnergyVADDetector:
    """Energy-based Voice Activity Detection fallback.

    Records chunks continuously, detects speech by RMS energy,
    and transcribes to check for wake words.
    """

    def __init__(self, wake_words: list[str], device: str = "plughw:0,0"):
        self.wake_words = [w.lower() for w in wake_words]
        self.device = device
        self.running = False

    def listen_for_wake(self) -> bool:
        """Listen for wake word. Returns True if detected."""
        print("👂 Listening for wake word (energy VAD)...", flush=True)
        speech_chunks = []

        while self.running:
            chunk = record_chunk(CHUNK_DURATION, self.device)
            if not chunk:
                time.sleep(0.1)
                continue

            energy = rms_energy(chunk)

            if energy >= SILENCE_THRESHOLD:
                speech_chunks.append(chunk)
            else:
                if len(speech_chunks) >= MIN_SPEECH_CHUNKS:
                    # We had speech - transcribe and check for wake word
                    pcm_data = b"".join(speech_chunks)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav_path = f.name
                    try:
                        save_pcm_to_wav(pcm_data, wav_path)
                        text = transcribe(wav_path)
                        if text:
                            text_lower = text.lower()
                            print(f"  Heard: '{text}'", flush=True)
                            for ww in self.wake_words:
                                if ww in text_lower:
                                    print(f"  ✅ Wake word detected: '{ww}'", flush=True)
                                    return True
                    finally:
                        try:
                            os.unlink(wav_path)
                        except OSError:
                            pass

                speech_chunks = []

        return False

    def record_command(self) -> str:
        """Record command audio after wake word. Returns transcribed text."""
        print("🎤 Recording command...", end="", flush=True)
        pcm_data = record_until_silence()
        print(" done", flush=True)

        if not pcm_data:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            save_pcm_to_wav(pcm_data, wav_path)
            return transcribe(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


class PorcupineDetector:
    """Porcupine-based wake word detection (preferred backend)."""

    def __init__(self, wake_words: list[str], device: str = "plughw:0,0"):
        self.wake_words = wake_words
        self.device = device
        self.running = False
        self._init_porcupine()

    def _init_porcupine(self):
        import pvporcupine
        import sounddevice as sd

        self.sd = sd
        keywords = self.wake_words
        self.handle = pvporcupine.create(keywords=keywords)
        self.stream = sd.RawInputStream(
            samplerate=self.handle.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.handle.frame_length,
        )

    def listen_for_wake(self) -> bool:
        """Listen for wake word using Porcupine. Returns True if detected."""
        print("👂 Listening for wake word (Porcupine)...", flush=True)
        self.stream.start()
        try:
            while self.running:
                pcm = self.stream.read(self.handle.frame_length)
                pcm_unpacked = struct.unpack(f"{self.handle.frame_length}h", pcm[0])
                result = self.handle.process(pcm_unpacked)
                if result >= 0:
                    print(f"  ✅ Wake word detected (keyword index: {result})", flush=True)
                    return True
        finally:
            self.stream.stop()
        return False

    def record_command(self) -> str:
        """Record command audio after wake word. Returns transcribed text."""
        print("🎤 Recording command...", end="", flush=True)
        pcm_data = record_until_silence()
        print(" done", flush=True)

        if not pcm_data:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            save_pcm_to_wav(pcm_data, wav_path)
            return transcribe(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def cleanup(self):
        """Release Porcupine resources."""
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        try:
            self.handle.delete()
        except Exception:
            pass


def create_detector(wake_words: list[str], device: str = "plughw:0,0"):
    """Create the best available detector. Returns (detector, backend_name)."""
    try:
        import pvporcupine  # noqa: F401

        print("Using Porcupine wake word engine", file=sys.stderr)
        return PorcupineDetector(wake_words, device), "porcupine"
    except ImportError:
        pass

    print("Porcupine not available, using energy-based VAD fallback", file=sys.stderr)
    return EnergyVADDetector(wake_words, device), "energy_vad"


# --- Command Execution ---


def execute_hermes(command: str) -> str:
    """Execute command via Hermes CLI and return response."""
    print(f"🤖 Executing: {command}")
    try:
        result = subprocess.run(
            [HERMES_CLI, "chat", "-q", command],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "HERMES_HOME": str(HERMES_HOME)},
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


def speak(text: str):
    """Text-to-speech via speech-dispatcher."""
    try:
        subprocess.run(
            ["spd-say", "-w", "-l", "th", text],
            capture_output=True,
            timeout=30,
        )
    except Exception as e:
        print(f"TTS error: {e}", file=sys.stderr)


# --- Daemon Management ---


def daemonize():
    """Fork into a background daemon process."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    # Redirect stdio
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = open(os.devnull, "r")
    os.dup2(devnull.fileno(), sys.stdin.fileno())


def write_pid():
    """Write PID file."""
    PID_FILE.write_text(str(os.getpid()))


def read_pid() -> int | None:
    """Read PID from file. Returns None if not running."""
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def stop_daemon():
    """Stop the running daemon."""
    pid = read_pid()
    if pid is None:
        print("No daemon running")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        PID_FILE.unlink(missing_ok=True)
        print(f"Daemon stopped (PID {pid})")
        return True
    except OSError as e:
        print(f"Error stopping daemon: {e}")
        return False


# --- Main Loop ---


def run_wake_loop(detector, backend_name: str, single_cycle: bool = False):
    """Main wake word detection loop."""
    detector.running = True

    def handle_signal(signum, frame):
        detector.running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while detector.running:
            # Listen for wake word
            wake_detected = detector.listen_for_wake()
            if not wake_detected:
                if single_cycle:
                    break
                continue

            # Play confirmation chime
            play_chime()

            # Record command
            command = detector.record_command()
            if not command:
                print("No command detected")
                if single_cycle:
                    break
                continue

            print(f"🗣️  Command: {command}")

            # Execute
            response = execute_hermes(command)
            print(f"🤖 Response: {response}")
            speak(response)

            if single_cycle:
                break

            time.sleep(0.5)
    finally:
        if backend_name == "porcupine" and hasattr(detector, "cleanup"):
            detector.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Hermes Wake Word Detector")
    parser.add_argument(
        "--once", action="store_true", help="Single wake+command cycle"
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run as background daemon"
    )
    parser.add_argument("--stop", action="store_true", help="Stop daemon")
    parser.add_argument("--status", action="store_true", help="Check daemon status")
    parser.add_argument(
        "--wake-word",
        action="append",
        help="Wake word(s) to listen for (default: 'hey hermes' + Thai)",
    )
    parser.add_argument(
        "--device",
        default="plughw:0,0",
        help="ALSA device for recording (default: plughw:0,0)",
    )
    args = parser.parse_args()

    wake_words = args.wake_word or WAKE_WORDS_DEFAULT

    if args.stop:
        stop_daemon()
        return

    if args.status:
        pid = read_pid()
        if pid:
            print(f"Daemon running (PID {pid})")
        else:
            print("Daemon not running")
        return

    if args.daemon:
        daemonize()
        write_pid()

    # Create detector
    detector, backend_name = create_detector(wake_words, args.device)

    print("=" * 50)
    print("🔊 Hermes Wake Word Detector")
    print(f"   Backend: {backend_name}")
    print(f"   Wake words: {wake_words}")
    print(f"   Device: {args.device}")
    print("=" * 50)

    run_wake_loop(detector, backend_name, single_cycle=args.once)


if __name__ == "__main__":
    main()
