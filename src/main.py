import argparse
import time
import cv2

from .blink_tracker import BlinkTracker
from .eyelid_metrics import MediapipeEyelidDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eyelid distance and blink rate monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=0.21, help="EAR threshold for closed eye")
    parser.add_argument("--window", type=float, default=60.0, help="Blink rate window in seconds")
    parser.add_argument("--fps", type=float, default=30.0, help="Assumed camera fps for debounce")
    parser.add_argument("--headless", action="store_true", help="Disable video preview, log to stdout")
    parser.add_argument("--noshow", action="store_true", help="Alias for --headless")
    parser.add_argument("--pitch-threshold", type=float, default=0.025, help="Pitch delta to classify nodding")
    parser.add_argument("--landmarks", action="store_true", help="Overlay landmarks for debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = MediapipeEyelidDetector(eyelid_threshold=args.threshold)
    tracker = BlinkTracker(
        blink_confirm_frames=max(1, int(args.fps * 0.1)),
        window_seconds=args.window,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            measurement = detector.process_frame(frame, include_landmarks=args.landmarks)
            if measurement is None:
                continue

            blink_state = tracker.update(measurement.eye.is_closed)
            eyelid_state = "closed" if blink_state.is_closed else "open"
            is_nodding = measurement.pitch_deg > args.pitch_threshold
            nod_state = "nodding" if is_nodding else "steady"
            caption = (
                f"state: {eyelid_state}/{nod_state} | EAR: {measurement.eye.eye_aspect_ratio:.3f} | "
                f"blink/min: {blink_state.blink_rate_per_min:.1f} | total: {blink_state.total_blinks}"
            )

            if args.headless or args.noshow:
                print(f"{time.time():.3f} {caption}")
            else:
                overlay = frame.copy()
                if args.landmarks and measurement.landmarks_px is not None:
                    for x, y in measurement.landmarks_px.astype(int):
                        cv2.circle(overlay, (x, y), 1, (255, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Eyelid Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
