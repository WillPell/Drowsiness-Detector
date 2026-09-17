from collections import deque

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

STATE_COLOURS = {
    "AWAKE": "#22c55e",
    "SLIGHTLY_DROWSY": "#eab308",
    "DROWSY": "#f97316",
    "CRITICAL": "#ef4444",
}

BG_DARK = "#1e1e2e"
BG_CARD = "#2a2a3c"
FG_TEXT = "#e2e2e8"
FG_DIM = "#9090a0"
GRAPH_LINE = "#60a5fa"


class DrowsinessGUI:
    def __init__(self, width: int = 720, height: int = 480, graph_seconds: float = 60.0):
        self.vid_w = width
        self.vid_h = height
        self.graph_h = 120
        self._graph_seconds = graph_seconds
        self._score_history: deque[tuple[float, float]] = deque()
        self._running = True
        self._photo_image = None

        self.root = tk.Tk()
        self.root.title("Drowsiness Detection System")
        self.root.configure(bg=BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root, width=self.vid_w, height=self.vid_h,
            bg="#000000", highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=(10, 0))

        self.status_frame = tk.Frame(self.root, bg=BG_CARD, height=50)
        self.status_frame.pack(fill="x", padx=10, pady=5)
        self.status_frame.pack_propagate(False)

        self.lbl_state = self._status_label("AWAKE", ("Consolas", 16, "bold"),
                                            STATE_COLOURS["AWAKE"], "left")
        self.lbl_score = self._status_label("Score: 0.0", ("Consolas", 13), FG_TEXT, "left")
        self.lbl_ear = self._status_label("EAR: 0.000", ("Consolas", 13), FG_DIM, "left")
        self.lbl_perclos = self._status_label("PERCLOS: 0.0%", ("Consolas", 13), FG_DIM, "left")
        self.lbl_fps = self._status_label("FPS: --", ("Consolas", 13), FG_DIM, "right")
        self.lbl_blink = self._status_label("Blinks: 0/min", ("Consolas", 13), FG_DIM, "right")

        self.graph_canvas = tk.Canvas(
            self.root, width=self.vid_w, height=self.graph_h,
            bg=BG_CARD, highlightthickness=0,
        )
        self.graph_canvas.pack(padx=10, pady=5)

        self.btn_frame = tk.Frame(self.root, bg=BG_DARK)
        self.btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Style().configure("Dark.TButton", font=("Consolas", 11))

        self.btn_calibrate = ttk.Button(self.btn_frame, text="Calibrate", style="Dark.TButton")
        self.btn_calibrate.pack(side="left", padx=5)

        self.lbl_calibration = tk.Label(
            self.btn_frame, text="", font=("Consolas", 11), bg=BG_DARK, fg=FG_DIM,
        )
        self.lbl_calibration.pack(side="left", padx=10)

        self.btn_quit = ttk.Button(
            self.btn_frame, text="Quit", style="Dark.TButton", command=self._on_close,
        )
        self.btn_quit.pack(side="right", padx=5)

    def update_frame(
        self,
        frame: np.ndarray,
        state: str = "AWAKE",
        score: float = 0.0,
        ear: float = 0.0,
        perclos: float = 0.0,
        blink_rate: float = 0.0,
        fps: float = 0.0,
        elapsed_s: float = 0.0,
    ):
        if not self._running:
            return

        display = cv2.resize(frame, (self.vid_w, self.vid_h))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._photo_image = ImageTk.PhotoImage(image=Image.fromarray(display))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

        self.lbl_state.config(text=state, fg=STATE_COLOURS.get(state, FG_TEXT))
        self.lbl_score.config(text=f"Score: {score:.1f}")
        self.lbl_ear.config(text=f"EAR: {ear:.3f}")
        self.lbl_perclos.config(text=f"PERCLOS: {perclos * 100:.1f}%")
        self.lbl_fps.config(text=f"FPS: {fps:.0f}")
        self.lbl_blink.config(text=f"Blinks: {blink_rate:.0f}/min")

        self._score_history.append((elapsed_s, score))
        cutoff = elapsed_s - self._graph_seconds
        while self._score_history and self._score_history[0][0] < cutoff:
            self._score_history.popleft()
        self._draw_graph()

    def set_calibration_callback(self, callback):
        self.btn_calibrate.config(command=callback)

    def set_calibration_status(self, text: str):
        self.lbl_calibration.config(text=text)

    def tick(self):
        if self._running:
            self.root.update_idletasks()
            self.root.update()

    @property
    def is_running(self) -> bool:
        return self._running

    def destroy(self):
        self._running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _status_label(self, text, font, colour, side) -> tk.Label:
        label = tk.Label(self.status_frame, text=text, font=font, bg=BG_CARD, fg=colour)
        label.pack(side=side, padx=15)
        return label

    def _on_close(self):
        self._running = False

    def _draw_graph(self):
        c = self.graph_canvas
        c.delete("all")
        w, h = self.vid_w, self.graph_h
        pad = 30

        for threshold, label, colour in [
            (25, "Slight", STATE_COLOURS["SLIGHTLY_DROWSY"]),
            (50, "Drowsy", STATE_COLOURS["DROWSY"]),
            (75, "Critical", STATE_COLOURS["CRITICAL"]),
        ]:
            y = pad + (1 - threshold / 100) * (h - 2 * pad)
            c.create_line(pad, y, w - 5, y, fill=colour, dash=(4, 4), width=1)
            c.create_text(pad - 3, y, text=label, anchor="e", fill=colour,
                          font=("Consolas", 7))

        if len(self._score_history) < 2:
            return

        t_min = self._score_history[0][0]
        t_max = self._score_history[-1][0]
        t_range = max(t_max - t_min, 1.0)

        points = []
        for t, s in self._score_history:
            x = pad + (t - t_min) / t_range * (w - pad - 5)
            y = pad + (1 - s / 100) * (h - 2 * pad)
            points.extend((x, y))

        c.create_line(*points, fill=GRAPH_LINE, width=2, smooth=True)
