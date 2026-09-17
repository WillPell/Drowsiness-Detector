import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

BLINK_WARMUP_S = 20.0
BLINK_RATE_WINDOW_S = 60.0
EAR_HISTORY_S = 120.0


class AlertState(IntEnum):
    AWAKE = 0
    SLIGHTLY_DROWSY = 1
    DROWSY = 2
    CRITICAL = 3


@dataclass
class MetricsConfig:
    ear_threshold: float = 0.22
    perclos_window: float = 30.0
    ema_alpha: float = 0.30
    score_weights: dict = field(default_factory=lambda: {
        "perclos": 0.45,
        "ear_deviation": 0.30,
        "blink_rate": 0.15,
        "head_nod": 0.10,
    })
    state_durations: dict = field(default_factory=lambda: {
        AlertState.SLIGHTLY_DROWSY: 2.0,
        AlertState.DROWSY: 3.0,
        AlertState.CRITICAL: 4.0,
    })
    score_thresholds: dict = field(default_factory=lambda: {
        AlertState.AWAKE: 25,
        AlertState.SLIGHTLY_DROWSY: 50,
        AlertState.DROWSY: 75,
    })
    head_nod_threshold: float = 15.0


class DrowsinessMetrics:
    def __init__(self, config: MetricsConfig | None = None):
        self.cfg = config or MetricsConfig()
        now = time.time()

        self._session_start = now
        self._ear_history: deque[tuple[float, float]] = deque()
        self._ema_ear: float | None = None

        self._blink_timestamps: deque[float] = deque()
        self._in_blink = False
        self._blink_cooldown = 0.15
        self._last_blink_end = 0.0

        self._closure_samples: deque[tuple[float, bool]] = deque()

        self._state = AlertState.AWAKE
        self._candidate_state = AlertState.AWAKE
        self._candidate_start = now

        self._calibration_ears: list[float] = []
        self._calibrated = False
        self._baseline_ear = 0.30

        self.smoothed_ear = 0.0
        self.perclos = 0.0
        self.blink_rate = 0.0
        self.drowsiness_score = 0.0
        self.state = AlertState.AWAKE

    def update(self, ear: float, pitch: float = 0.0) -> AlertState:
        now = time.time()

        self.smoothed_ear = self._apply_ema(ear)

        is_closed = self.smoothed_ear < self.cfg.ear_threshold
        self._closure_samples.append((now, is_closed))
        self._prune_deque(self._closure_samples, now, self.cfg.perclos_window)
        self.perclos = self._compute_perclos()

        self._detect_blink(is_closed, now)
        self.blink_rate = self._compute_blink_rate(now)

        self.drowsiness_score = self._compute_score(now, pitch)
        self.state = self._update_state(now)

        self._ear_history.append((now, ear))
        self._prune_deque(self._ear_history, now, EAR_HISTORY_S)

        return self.state

    def add_calibration_sample(self, ear: float):
        self._calibration_ears.append(ear)

    def reset_calibration(self):
        self._calibration_ears.clear()

    def finalise_calibration(self):
        if len(self._calibration_ears) < 10:
            return
        sorted_ears = sorted(self._calibration_ears)
        median = sorted_ears[len(sorted_ears) // 2]
        self._baseline_ear = median
        self.cfg.ear_threshold = round(median * 0.75, 3)
        self._calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def baseline_ear(self) -> float:
        return self._baseline_ear

    def _apply_ema(self, raw: float) -> float:
        if self._ema_ear is None:
            self._ema_ear = raw
        self._ema_ear = self.cfg.ema_alpha * raw + (1 - self.cfg.ema_alpha) * self._ema_ear
        return self._ema_ear

    @staticmethod
    def _prune_deque(dq: deque, now: float, window: float):
        cutoff = now - window
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _compute_perclos(self) -> float:
        if not self._closure_samples:
            return 0.0
        closed = sum(1 for _, c in self._closure_samples if c)
        return closed / len(self._closure_samples)

    def _detect_blink(self, is_closed: bool, now: float):
        if is_closed and not self._in_blink:
            self._in_blink = True
        elif not is_closed and self._in_blink:
            self._in_blink = False
            if (now - self._last_blink_end) > self._blink_cooldown:
                self._blink_timestamps.append(now)
                self._last_blink_end = now

    def _compute_blink_rate(self, now: float) -> float:
        self._prune_blinks(now)
        elapsed = min(now - self._session_start, BLINK_RATE_WINDOW_S)
        if elapsed < 1.0:
            return 0.0
        return len(self._blink_timestamps) * (60.0 / elapsed)

    def _prune_blinks(self, now: float):
        cutoff = now - BLINK_RATE_WINDOW_S
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

    def _compute_score(self, now: float, pitch: float) -> float:
        w = self.cfg.score_weights

        perclos_score = min(self.perclos / 0.40, 1.0) * 100

        ear_dev = max(0.0, self._baseline_ear - self.smoothed_ear) / self._baseline_ear
        ear_dev_score = min(ear_dev / 0.40, 1.0) * 100

        if (now - self._session_start) < BLINK_WARMUP_S:
            blink_score = 0.0
        elif self.blink_rate < 5:
            blink_score = min((5 - self.blink_rate) / 5.0, 1.0) * 100
        elif self.blink_rate > 25:
            blink_score = min((self.blink_rate - 25) / 15.0, 1.0) * 100
        else:
            blink_score = 0.0

        if pitch > self.cfg.head_nod_threshold:
            head_score = min((pitch - self.cfg.head_nod_threshold) / 15.0, 1.0) * 100
        else:
            head_score = 0.0

        raw = (
            w["perclos"] * perclos_score
            + w["ear_deviation"] * ear_dev_score
            + w["blink_rate"] * blink_score
            + w["head_nod"] * head_score
        )
        return round(max(0.0, min(100.0, raw)), 1)

    def _update_state(self, now: float) -> AlertState:
        thresholds = self.cfg.score_thresholds

        if self.drowsiness_score >= thresholds[AlertState.DROWSY]:
            target = AlertState.CRITICAL
        elif self.drowsiness_score >= thresholds[AlertState.SLIGHTLY_DROWSY]:
            target = AlertState.DROWSY
        elif self.drowsiness_score >= thresholds[AlertState.AWAKE]:
            target = AlertState.SLIGHTLY_DROWSY
        else:
            target = AlertState.AWAKE

        if target < self._state:
            self._state = target
            self._candidate_state = target
            self._candidate_start = now
            return self._state

        if target > self._state:
            if target != self._candidate_state:
                self._candidate_state = target
                self._candidate_start = now
            else:
                required = self.cfg.state_durations.get(target, 2.0)
                if (now - self._candidate_start) >= required:
                    self._state = target
        else:
            self._candidate_state = self._state
            self._candidate_start = now

        return self._state

    def predict_fatigue_trend(self, horizon_minutes: float = 5.0) -> float | None:
        if len(self._ear_history) < 30:
            return None

        t0 = self._ear_history[0][0]
        xs, ys = [], []
        for t, ear_val in self._ear_history:
            xs.append(t - t0)
            deviation = max(0.0, self._baseline_ear - ear_val) / self._baseline_ear
            ys.append(min(deviation / 0.40, 1.0) * 100)

        xs, ys = np.array(xs), np.array(ys)
        if xs[-1] - xs[0] < 5.0:
            return None

        slope, intercept = np.polyfit(xs, ys, 1)
        predicted = slope * (xs[-1] + horizon_minutes * 60.0) + intercept
        return round(max(0.0, min(100.0, float(predicted))), 1)
