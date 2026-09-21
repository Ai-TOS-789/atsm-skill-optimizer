#!/usr/bin/env python3
"""ATSM Multimodal: Extend ATSM ranking to image/audio/video inputs.

Usage:
    python3 atsm_multimodal.py image /path/to/image.jpg
    python3 atsm_multimodal.py audio /path/to/audio.mp3
    python3 atsm_multimodal.py video /path/to/video.mp4
    python3 atsm_multimodal.py status

Flow:
    1. Extract text from media (describe / transcribe / extract frames)
    2. Feed text to ATSM rank_skills()
    3. Print ranked results
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency detection (lazy — import only when the modality is requested)
# ---------------------------------------------------------------------------

def check_dependencies() -> dict[str, bool]:
    """Probe all optional dependencies and return availability map."""
    deps: dict[str, bool] = {}

    # Python packages
    try:
        import PIL  # noqa: F401
        deps["PIL"] = True
    except ImportError:
        deps["PIL"] = False

    try:
        import cv2  # noqa: F401
        deps["cv2"] = True
    except ImportError:
        deps["cv2"] = False

    try:
        import faster_whisper  # noqa: F401
        deps["faster_whisper"] = True
    except ImportError:
        deps["faster_whisper"] = False

    # System binaries
    deps["ffmpeg"] = shutil.which("ffmpeg") is not None

    return deps


# ---------------------------------------------------------------------------
# ATSM integration — import from sibling script
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from atsm import rank_skills  # noqa: E402


# ---------------------------------------------------------------------------
# Image → text
# ---------------------------------------------------------------------------

def describe_image(path: str) -> str:
    """Produce a text description of an image using PIL (and cv2 if available).

    Description includes: format, dimensions, dominant colors, brightness,
    aspect ratio hints. This text is then used to rank skills.
    """
    from PIL import Image, ImageStat

    img = Image.open(path)
    w, h = img.size
    fmt = img.format or "unknown"
    mode = img.mode

    # Basic descriptors
    parts = [f"{fmt} image", f"{w}x{h}", f"mode {mode}"]

    # Aspect ratio hint
    ratio = w / h if h else 0
    if ratio > 2:
        parts.append("panoramic wide")
    elif ratio < 0.5:
        parts.append("portrait tall")
    elif 0.9 < ratio < 1.1:
        parts.append("square")
    else:
        parts.append("landscape" if ratio > 1 else "portrait")

    # Brightness
    try:
        if mode == "L":
            stat = ImageStat.Stat(img)
        else:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]
        if brightness < 64:
            parts.append("dark")
        elif brightness > 192:
            parts.append("bright")
        else:
            parts.append("medium brightness")
    except Exception:
        pass

    # Dominant colors via histogram quantization
    try:
        small = img.convert("RGB").resize((64, 64))
        pixels = list(small.getdata())
        # Quantize to 4 bits per channel
        quantized = [(r >> 4 << 4, g >> 4 << 4, b >> 4 << 4) for r, g, b in pixels]
        from collections import Counter
        top = Counter(quantized).most_common(3)
        color_names = {
            (0, 0, 0): "black", (255, 255, 255): "white",
            (255, 0, 0): "red", (0, 255, 0): "green", (0, 0, 255): "blue",
            (255, 255, 0): "yellow", (255, 0, 255): "magenta", (0, 255, 255): "cyan",
            (128, 128, 128): "gray", (128, 0, 0): "dark red",
            (0, 128, 0): "dark green", (0, 0, 128): "dark blue",
            (255, 165, 0): "orange", (165, 42, 42): "brown",
        }
        for color, _ in top:
            name = color_names.get(color, "")
            if name and name not in parts:
                parts.append(name)
    except Exception:
        pass

    # If cv2 is available, add edge/detail info
    try:
        import cv2
        import numpy as np
        cv_img = cv2.imread(path)
        if cv_img is not None:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = edges.sum() / (255 * edges.size)
            if edge_ratio > 0.15:
                parts.append("high detail")
            elif edge_ratio < 0.03:
                parts.append("low detail simple")
            # Face detection (very basic cascade check)
            cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                # Try alternative locations
                import glob
                candidates = glob.glob("/usr/share/**/haarcascade_frontalface_default.xml", recursive=True)
                if candidates:
                    cascade_path = candidates[0]
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                faces = cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    parts.append(f"contains {len(faces)} face(s)")
    except Exception:
        pass

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Audio → text
# ---------------------------------------------------------------------------

def transcribe_audio(path: str) -> str:
    """Transcribe audio file using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, beam_size=5, language=None)
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
    transcript = " ".join(text_parts)
    lang_info = f"[lang={info.language}, prob={info.language_probability:.2f}]"
    return f"{lang_info} {transcript}"


# ---------------------------------------------------------------------------
# Video → text
# ---------------------------------------------------------------------------

def extract_video_frames(path: str, num_frames: int = 5) -> list[str]:
    """Extract keyframes from video using ffmpeg, then describe each with PIL.

    Returns a list of frame descriptions.
    """
    tmpdir = tempfile.mkdtemp(prefix="atsm_video_")
    try:
        # Get video duration
        probe = subprocess.run(
            ["ffmpeg", "-i", path],
            capture_output=True, text=True
        )
        duration = 0.0
        for line in probe.stderr.splitlines():
            if "Duration" in line:
                # Format: Duration: 00:01:23.45
                try:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = time_str.split(":")
                    duration = int(h) * 3600 + int(m) * 60 + float(s)
                except (ValueError, IndexError):
                    pass
                break

        if duration <= 0:
            duration = 10.0  # fallback

        # Extract frames at evenly spaced timestamps
        interval = duration / (num_frames + 1)
        frame_paths = []
        for i in range(num_frames):
            ts = interval * (i + 1)
            out_path = os.path.join(tmpdir, f"frame_{i:03d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", path,
                 "-frames:v", "1", "-q:v", "3", out_path],
                capture_output=True, timeout=30
            )
            if os.path.exists(out_path):
                frame_paths.append((ts, out_path))

        # Describe each frame
        descriptions = []
        for ts, fp in frame_paths:
            desc = describe_image(fp)
            descriptions.append(f"t={ts:.1f}s: {desc}")

        return descriptions
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_status(deps: dict[str, bool]):
    """Print dependency status table."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║        ATSM Multimodal — Dependency Status       ║")
    print("╠══════════════════════════════════════════════════╣")
    for name, available in deps.items():
        icon = "✅" if available else "❌"
        status = "available" if available else "NOT FOUND"
        print(f"║  {icon}  {name:<20} {status:<25} ║")
    print("╠══════════════════════════════════════════════════╣")

    # Modality readiness
    image_ok = deps["PIL"]
    audio_ok = deps["faster_whisper"]
    video_ok = deps["ffmpeg"] and deps["PIL"]

    print("║  Modality support:                               ║")
    for mod, ok in [("image", image_ok), ("audio", audio_ok), ("video", video_ok)]:
        icon = "✅" if ok else "❌"
        print(f"║    {icon}  {mod:<10} {'ready' if ok else 'missing deps':<33} ║")
    print("╚══════════════════════════════════════════════════╝\n")


def print_results(results: list[dict], elapsed: float, source_text: str):
    """Print ranked skill results."""
    if not results:
        print("No skills found.")
        return

    print(f"\n📋 Extracted text: {source_text[:120]}{'...' if len(source_text) > 120 else ''}")
    print(f"\n{'Rank':<5} {'Score':<8} {'Rel':<6} {'P(success)':<11} {'N':<4} {'Skill'}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['final_score']:<8} {r['relevance']:<6} {r['success_prob']:<11} {r['observations']:<4} {r['name']}")
        if r["description"]:
            print(f"      └─ {r['description'][:60]}")
    print(f"\n  ⚡ {elapsed*1000:.2f}ms")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "status":
        deps = check_dependencies()
        print_status(deps)
        return

    if len(sys.argv) < 3:
        print(f"Usage: atsm_multimodal.py {command} <file_path>")
        sys.exit(1)

    file_path = sys.argv[2]
    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    deps = check_dependencies()

    if command == "image":
        if not deps["PIL"]:
            print("❌ PIL (Pillow) is required for image analysis.")
            print("   Install: pip install Pillow")
            sys.exit(1)
        print(f"🖼  Analyzing image: {file_path}")
        text = describe_image(file_path)
        results, elapsed = rank_skills(text)
        print_results(results, elapsed, text)

    elif command == "audio":
        if not deps["faster_whisper"]:
            print("❌ faster-whisper is required for audio transcription.")
            print("   Install: pip install faster-whisper")
            sys.exit(1)
        print(f"🎤 Transcribing audio: {file_path}")
        text = transcribe_audio(file_path)
        results, elapsed = rank_skills(text)
        print_results(results, elapsed, text)

    elif command == "video":
        if not deps["ffmpeg"]:
            print("❌ ffmpeg is required for video frame extraction.")
            print("   Install: sudo apt install ffmpeg")
            sys.exit(1)
        if not deps["PIL"]:
            print("❌ PIL (Pillow) is required for frame analysis.")
            print("   Install: pip install Pillow")
            sys.exit(1)
        print(f"🎬 Extracting frames from video: {file_path}")
        frame_descs = extract_video_frames(file_path)
        text = " ".join(frame_descs)
        results, elapsed = rank_skills(text)
        print_results(results, elapsed, text)

    else:
        print(f"Unknown command: {command}")
        print("Commands: image, audio, video, status")
        sys.exit(1)


if __name__ == "__main__":
    main()
