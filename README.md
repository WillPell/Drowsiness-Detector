# Drowsiness Detection System

Real-time driver drowsiness detection using computer vision and embedded hardware.

A webcam feed is converted into per-frame eye geometry with MediaPipe Face Landmarker, reduced to established fatigue metrics (EAR, PERCLOS, blink rate, head pose), scored by a weighted classifier with a hysteretic state machine, and surfaced both on screen and through an Arduino-driven RGB LED and buzzer. Runs at 25–30 FPS on a typical laptop, and runs without the hardware attached.

---

## System Architecture

```
┌─────────────┐    USB     ┌─────────────────────────────────────────────┐
│   Webcam    │───────────▶│              Python Host                    │
└─────────────┘            │                                             │
                           │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
                           │  │vision.py │─▶│metrics.py│─▶│serial_comm│  │
                           │  │          │  │          │  │   .py     │  │
                           │  │Face mesh │  │ PERCLOS  │  │           │  │
                           │  │EAR calc  │  │ Score    │  │ State cmd │  │
                           │  │Head pose │  │ State    │  │ over UART │  │
                           │  └──────────┘  └──────────┘  └─────┬─────┘  │
                           │  ┌──────────┐  ┌───────────┐       │        │
                           │  │  gui.py  │  │data_logger│       │        │
                           │  │Live feed │  │ CSV logs  │       │        │
                           │  │Score graph│ │           │       │        │
                           │  └──────────┘  └───────────┘       │        │
                           └────────────────────────────────────┼────────┘
                                                                │ Serial
                           ┌────────────────────────────────────▼────────┐
                           │           Arduino Mega2560                  │
                           │  ┌─────────────┐  ┌──────────────────────┐  │
                           │  │ RGB LED     │  │ Piezo buzzer         │  │
                           │  │ (PWM fade)  │  │ (critical alert)     │  │
                           │  └─────────────┘  └──────────────────────┘  │
                           └─────────────────────────────────────────────┘
```

## Project Structure

```
drowsiness_detection_system/
├── src/
│   ├── main.py            # Entry point and capture loop
│   ├── vision.py          # Face landmarks, EAR, head pose
│   ├── metrics.py         # PERCLOS, score, state machine
│   ├── serial_comm.py     # Arduino serial link
│   ├── data_logger.py     # CSV session logging
│   ├── gui.py             # Tkinter dashboard
│   └── config.py          # Tuneable parameters
├── arduino/
│   ├── drowsiness_controller/   # Alert firmware
│   └── circuit_tester/          # Hardware bring-up sketch
├── analysis/
│   └── analyse_session.py # Post-session figure and report
├── logs/                  # Session CSVs (created on first run)
├── requirements.txt
└── README.md
```

## Algorithms

### Eye Aspect Ratio (EAR)

Based on [Soukupová & Čech, 2016](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf). Six landmarks define each eye contour; the ratio of vertical to horizontal distances gives a scalar that drops sharply when the eye closes:

```
EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 × ‖p1 − p4‖)
```

- Open eye ≈ 0.25–0.35
- Closed eye ≈ 0.05–0.15
- Default threshold 0.22, personalised by calibration

### PERCLOS (Percentage of Eye Closure)

The standard fatigue metric from FHWA research: the fraction of time the eyes are below the EAR threshold over a sliding window (default 30 seconds).

```
PERCLOS = (frames with EAR < threshold) / (total frames in window)
```

PERCLOS above 40% is strongly associated with impaired driving performance.

### Drowsiness Score (0–100)

A weighted composite of four signals:

| Signal         | Weight | Interpretation                             |
|----------------|--------|--------------------------------------------|
| PERCLOS        | 45%    | Primary drowsiness indicator               |
| EAR deviation  | 30%    | How far below baseline the eyes are        |
| Blink rate     | 15%    | Too few or too many blinks suggests fatigue |
| Head nod       | 10%    | Downward pitch beyond threshold            |

The blink term is held at zero for the first 20 seconds of a session: until a reasonable share of the one-minute window has elapsed, a low measured rate reflects a short observation rather than a drowsy driver.

### State Machine (with hysteresis)

Four states with time-based transition guards to suppress false alarms:

| State            | Score Range | Required Duration | LED    | Buzzer |
|------------------|-------------|-------------------|--------|--------|
| AWAKE            | 0–24        | instant (down)    | Green  | Off    |
| SLIGHTLY_DROWSY  | 25–49       | 2 seconds         | Yellow | Off    |
| DROWSY           | 50–74       | 3 seconds         | Orange | Off    |
| CRITICAL         | 75–100      | 4 seconds         | Red    | On     |

Upward transitions require a sustained score; downward transitions are instant, so recovery feels immediate.

### Signal Smoothing

EAR values are smoothed with an exponential moving average (α = 0.30):

```
EAR_smooth[t] = α × EAR_raw[t] + (1 − α) × EAR_smooth[t−1]
```

This suppresses frame-to-frame jitter while keeping sub-second responsiveness.

### Head Pose Estimation

`cv2.solvePnP` fits a generic 3D face model to six reference landmarks to recover pitch, yaw and roll. The model is expressed in the OpenCV camera convention (x right, y down, z away from the camera) so a forward-facing head yields angles near zero; a y-up model instead returns angles near ±180° that wrap unpredictably across the boundary. Pitch is positive when the head tilts down, and a sustained pitch beyond +15° contributes to the score.

### Fatigue Trend Prediction

Least-squares extrapolation of the EAR-derived score over the last two minutes, projected five minutes forward, logged every 30 seconds as an early warning of deteriorating alertness.

## Setup

### Prerequisites

- Python 3.10+
- Webcam
- Arduino IDE and a Mega2560 with an RGB LED and piezo buzzer (optional — the system runs in stub mode without hardware)

### Installation

```bash
cd drowsiness_detection_system
pip install -r requirements.txt
```

The MediaPipe face landmarker model (~3 MB) downloads automatically on first run.

To flash the board, open `arduino/drowsiness_controller/drowsiness_controller.ino` in the Arduino IDE, select Mega2560, and upload. `arduino/circuit_tester/circuit_tester.ino` cycles the LED and buzzer to verify wiring first.

### Wiring

```
Arduino Mega2560
    Pin 9  ──[220Ω]──▶ RGB LED red
    Pin 10 ──[220Ω]──▶ RGB LED green
    Pin 11 ──[220Ω]──▶ RGB LED blue
    Pin 8  ──[100Ω]──▶ Piezo buzzer (+)
    GND    ───────────▶ LED common cathode / buzzer (−)
```

### Running

Run from the project root:

```bash
# Full system with GUI and Arduino
python -m src.main --port COM3

# Without the Arduino
python -m src.main --no-serial

# Headless (OpenCV window only, press q to quit)
python -m src.main --no-gui --no-serial
```

| Flag | Purpose |
|------|---------|
| `--port` | Arduino serial port (default `COM3`) |
| `--baud` | Baud rate (default 9600) |
| `--camera` | Camera index (default 0) |
| `--no-serial` | Skip the Arduino link |
| `--no-gui` | Run without the Tkinter window |

### Calibration

1. Click **Calibrate** in the GUI.
2. Look at the camera with eyes open for 5 seconds.
3. The EAR threshold is set to 75% of your median open-eye EAR.

Calibration is optional; without it the system uses the default 0.22 threshold.

### Post-Session Analysis

```bash
python -m analysis.analyse_session logs/session_*.csv
```

For each session this writes `*_analysis.png` (EAR, score, PERCLOS and blink rate over time, with state regions shaded) and `*_report.txt` (duration, frame rate, metric averages, state distribution and a risk assessment) next to the CSV.

## Serial Protocol

Single-character commands at 9600 baud:

| Byte | Meaning |
|------|---------|
| `0`–`3` | Alert state (AWAKE … CRITICAL) |
| `P` | Ping; the board replies `OK` |

The host sends a command when the state changes and then re-sends it about once a second. That heartbeat is what makes the firmware watchdog meaningful: if nothing arrives for 5 seconds the host is assumed to have died and the board reverts to AWAKE rather than latching an alert on forever.

## Performance

Target ≥ 15 FPS; typically 25–30 FPS on modern hardware.

- Landmarks are computed once per frame and reused for both EAR and head pose
- EMA smoothing and the PERCLOS window are O(1) per frame
- Serial traffic is limited to state changes plus a 1 Hz heartbeat
- CSV writes are flushed every 50 rows rather than every frame
- The loop is capped at `TARGET_FPS` so a fast camera does not burn CPU

## Known Limitations

- Single face only; the landmarker is configured for one subject
- Assumes reasonable lighting — performance degrades in the dark and in direct glare
- Head pose uses a generic face model, so absolute angles are approximate; the nod signal is a relative cue rather than a measurement
- Glasses with strong reflections can destabilise the EAR signal
- The score weights are hand-tuned, not trained on labelled data

## Further Improvements

- **Yawn detection** — add Mouth Aspect Ratio as a fifth signal
- **ML classifier** — replace the heuristic score with a model trained on labelled drowsiness data (e.g. the NTHU-DDD dataset)
- **Multi-face support** — pick the largest face in multi-occupant scenarios
- **Bluetooth** — replace USB serial with an HC-05 for wireless operation
- **Raspberry Pi deployment** — port to a Pi 4 with a camera module as a standalone unit
- **Web dashboard** — replace Tkinter with a Flask/WebSocket UI for remote monitoring
- **Night vision** — add IR illumination and NIR camera support

## Acknowledgements

This project implements published methods rather than inventing them:

- **Eye Aspect Ratio** — Soukupová & Čech, *Real-Time Eye Blink Detection using Facial Landmarks* (CVWW 2016). The EAR formula and the six-point eye contour come from this paper.
- **PERCLOS** — Wierwille et al., *Research on Vehicle-Based Driver Status/Performance Monitoring* (FHWA, 1994), which established proportion of eye closure as a validated fatigue measure.
- **Head pose via solvePnP** — the generic 3D face model coordinates are the anthropometric constants used throughout the standard OpenCV head-pose recipe.
- **MediaPipe Face Landmarker** — Google's face mesh model supplies the 478 facial landmarks. The model file is downloaded at run time under its own terms and is not redistributed here.

The scoring weights, state machine, hysteresis behaviour, calibration routine, firmware, GUI and analysis tooling are original to this project.

### Third-party dependencies

All permissive and installed via pip rather than vendored: MediaPipe and OpenCV (Apache-2.0), NumPy, pandas and pyserial (BSD), Pillow (HPND), matplotlib (PSF-based).

## License

MIT — see [LICENSE](LICENSE). This covers the source in this repository; the referenced papers, models and libraries remain under their own terms.
