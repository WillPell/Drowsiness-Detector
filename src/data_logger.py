# logs data to a csv file, useful if doing any sort of research/ML training
# functionally optional though

import csv
import os
import time
from datetime import datetime

FIELDNAMES = [
    "timestamp",
    "elapsed_s",
    "raw_ear",
    "smoothed_ear",
    "perclos",
    "blink_rate",
    "drowsiness_score",
    "state",
    "head_pitch",
    "head_yaw",
]


class DataLogger:
    
    # appends one row per frame to a CSV log file.

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"session_{ts}.csv")
        self._start_time = time.time()

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._row_count = 0

    def log(
        self,
        raw_ear: float,
        smoothed_ear: float,
        perclos: float,
        blink_rate: float,
        drowsiness_score: float,
        state: str,
        head_pitch: float = 0.0,
        head_yaw: float = 0.0,
    ):
        """Write a single row to the CSV."""
        now = time.time()
        self._writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(now - self._start_time, 3),
            "raw_ear": round(raw_ear, 4),
            "smoothed_ear": round(smoothed_ear, 4),
            "perclos": round(perclos, 4),
            "blink_rate": round(blink_rate, 1),
            "drowsiness_score": round(drowsiness_score, 1),
            "state": state,
            "head_pitch": round(head_pitch, 1),
            "head_yaw": round(head_yaw, 1),
        })
        self._row_count += 1
        # forces every 50 rows to be written to disk in case of crash (this has happened before)
        if self._row_count % 50 == 0:
            self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def rows_written(self) -> int:
        return self._row_count
