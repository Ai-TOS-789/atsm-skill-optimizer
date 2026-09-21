# Voice Assistant Reference

## Pipeline

```
User speaks → arecord (plughw:0,0) → WAV file
    ↓
faster-whisper (base model, CPU) → text transcript
    ↓
hermes chat -q "<text>" → agent response
    ↓
spd-say -w -l th "<response>" → TTS output
```

## Key Commands

```bash
# Record
arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 --duration 30 output.wav

# Transcribe
python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('base', device='cpu', compute_type='int8')
segments, _ = model.transcribe('output.wav', language='th')
print(' '.join(s.text for s in segments))
"

# TTS
spd-say -w -l th "ข้อความที่จะพูด"
```

## Dependencies

- `faster-whisper` — STT (installed in hermes-agent venv)
- `spd-say` — TTS via speech-dispatcher (system package)
- `arecord` — ALSA capture (system package)

## Modes

| Mode | Command | Behavior |
|------|---------|----------|
| Continuous | `hermes_voice.py` | Records → transcribes → executes loop |
| Single | `hermes_voice.py --once` | One command then exit |
| Text | `hermes_voice.py --text` | Type instead of speak |
| Wake | `hermes_voice.py --wake` | Wait for wake word, then command |

## Wake Word Detection

Uses `hermes_wake.py`:
- Primary: Porcupine engine (if `pvporcupine` installed)
- Fallback: Energy-based VAD (always available)
- Bilingual: "hey hermes" + "สวัสดีเฮอร์เมส"
- Chime feedback via generated `chime.wav`

## Pitfalls

- **ALSA device**: Use `plughw:0,0` not `hw:0,0` — the latter fails on many systems
- **Port conflict**: Default CDP port 9222 conflicts with Hermes — use 9223 for browser_bot
- **spd-say blocking**: Use `-w` flag to wait for speech completion, otherwise TTS overlaps
- **Whisper model**: `base` model is CPU-friendly but less accurate than `small` or `medium`
