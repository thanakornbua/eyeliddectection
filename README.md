# Eyelid Detection

CPU-only eyelid state monitor using OpenCV + MediaPipe. Provides live state (open/closed), eyelid distance via eye-aspect ratio (EAR), and blink rate per minute.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main --camera 0 --threshold 0.21 --window 60 --fps 30 --pitch-threshold 10 --landmarks
```

Press `q` to exit the preview window. Use `--headless` to log stats in the console and `--landmarks` to draw landmarks.

## Notes

- EAR threshold `0.21` works for many setups; adjust per user by lowering when false positives occur or raising if blinks are missed.
- Blink rate window determines how many seconds of history contribute to blinks/min computation. `60` seconds mimics a rolling minute average.
- Mediapipe face mesh runs fully on CPU by default; ensure adequate lighting for stable landmarks.
- Use `--pitch-threshold` (degrees) and `--pitch-alpha` for nod detection sensitivity. HUD size and margins are
  customizable via `--hud-*` flags.
