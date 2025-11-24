from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import time


@dataclass
class BlinkState:
    is_closed: bool
    blink_rate_per_min: float
    total_blinks: int


class BlinkTracker:
    def __init__(self, blink_confirm_frames: int = 2, window_seconds: float = 60.0) -> None:
        self.blink_confirm_frames = blink_confirm_frames
        self.window_seconds = window_seconds
        self.state_history: Deque[tuple[bool, float]] = deque()
        self.pending_closed_frames = 0
        self.is_closed = False
        self.total_blinks = 0

    def _cleanup(self, now: float) -> None:
        while self.state_history and now - self.state_history[0][1] > self.window_seconds:
            self.state_history.popleft()

    def update(self, is_closed: bool, timestamp: Optional[float] = None) -> BlinkState:
        timestamp = timestamp or time.time()

        if is_closed:
            self.pending_closed_frames += 1
        else:
            if self.is_closed and self.pending_closed_frames >= self.blink_confirm_frames:
                self.total_blinks += 1
                self.state_history.append((True, timestamp))
            self.pending_closed_frames = 0

        self.is_closed = is_closed
        self._cleanup(timestamp)
        blink_rate = len(self.state_history) / (self.window_seconds / 60.0)
        return BlinkState(is_closed=is_closed, blink_rate_per_min=blink_rate, total_blinks=self.total_blinks)
