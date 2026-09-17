import argparse
import logging
import time

import cv2

from .config import (
    CALIBRATION_DURATION_S,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    EAR_THRESHOLD,
    EMA_ALPHA,
    FACE_DETECTION_CONFIDENCE,
    FACE_TRACKING_CONFIDENCE,
    GRAPH_HISTORY_S,
    GUI_HEIGHT,
    GUI_WIDTH,
    LOG_DIR,
    PERCLOS_WINDOW_S,
    SERIAL_BAUD,
    SERIAL_PORT,
    TARGET_FPS,
)
from .data_logger import DataLogger
from .metrics import DrowsinessMetrics, MetricsConfig
from .serial_comm import ArduinoSerial
from .vision import VisionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

HUD_COLOURS = {
    "AWAKE": (34, 197, 94),
    "SLIGHTLY_DROWSY": (8, 179, 234),
    "DROWSY": (22, 115, 249),
    "CRITICAL": (68, 68, 239),
}
FPS_WINDOW_FRAMES = 30
TREND_LOG_INTERVAL_S = 30.0


def parse_args():
    parser = argparse.ArgumentParser(description="Drowsiness Detection System")
    parser.add_argument("--port", default=SERIAL_PORT, help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="Baud rate")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--no-serial", action="store_true", help="Disable the Arduino link")
    parser.add_argument("--no-gui", action="store_true", help="Run without the Tkinter window")
    return parser.parse_args()


def draw_hud(frame, state, score, ear, fps, face_found):
    h, w = frame.shape[:2]
    colour = HUD_COLOURS.get(state, (255, 255, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 46), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    label = state if face_found else "NO FACE"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, f"Score: {score:.0f}", (220, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (226, 226, 232), 1)
    cv2.putText(frame, f"EAR: {ear:.3f}", (380, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (226, 226, 232), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 144, 160), 1)

    bar_y = h - 20
    cv2.rectangle(frame, (0, bar_y), (int(score / 100.0 * w), h), colour, -1)
    cv2.rectangle(frame, (0, bar_y), (w, h), (60, 60, 80), 1)
    return frame


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", index)
        return None
    return cap


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("Initialising system...")

    vision = VisionProcessor(
        min_detection_confidence=FACE_DETECTION_CONFIDENCE,
        min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
    )
    metrics = DrowsinessMetrics(config=MetricsConfig(
        ear_threshold=EAR_THRESHOLD,
        perclos_window=PERCLOS_WINDOW_S,
        ema_alpha=EMA_ALPHA,
    ))

    arduino = ArduinoSerial(port=args.port, baud=args.baud)
    if not args.no_serial:
        arduino.connect()
        if arduino.is_connected and arduino.ping():
            logger.info("Arduino handshake OK")

    data_log = DataLogger(log_dir=LOG_DIR)

    gui = None
    if not args.no_gui:
        from .gui import DrowsinessGUI

        gui = DrowsinessGUI(width=GUI_WIDTH, height=GUI_HEIGHT, graph_seconds=GRAPH_HISTORY_S)

    cap = open_camera(args.camera)
    if cap is None:
        data_log.close()
        vision.release()
        if gui:
            gui.destroy()
        return

    calibrating = False
    calibration_start = 0.0

    def start_calibration():
        nonlocal calibrating, calibration_start
        calibrating = True
        calibration_start = time.time()
        metrics.reset_calibration()
        if gui:
            gui.set_calibration_status("Look at the camera with eyes open...")
        logger.info("Calibration started")

    if gui:
        gui.set_calibration_callback(start_calibration)

    frame_times: list[float] = []
    fps = 0.0
    last_trend_log = 0.0
    frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.0

    logger.info("System running, press 'q' or close the window to stop")

    try:
        while True:
            loop_start = time.time()

            if gui and not gui.is_running:
                break

            ok, frame = cap.read()
            if not ok:
                logger.warning("Frame capture failed")
                continue

            landmarks = vision.process_frame(frame)
            pitch = yaw = 0.0
            raw_ear = 0.0

            if landmarks is not None:
                _, _, raw_ear = vision.get_ear(landmarks)
                pitch, yaw, _ = vision.estimate_head_pose(landmarks, frame.shape)

                if calibrating:
                    metrics.add_calibration_sample(raw_ear)
                    elapsed_cal = time.time() - calibration_start
                    if elapsed_cal >= CALIBRATION_DURATION_S:
                        metrics.finalise_calibration()
                        calibrating = False
                        status = (
                            f"Calibrated: threshold={metrics.cfg.ear_threshold:.3f} "
                            f"baseline={metrics.baseline_ear:.3f}"
                        )
                        if gui:
                            gui.set_calibration_status(status)
                        logger.info(status)
                    elif gui:
                        gui.set_calibration_status(
                            f"Calibrating... {CALIBRATION_DURATION_S - elapsed_cal:.1f}s remaining"
                        )

                metrics.update(ear=raw_ear, pitch=pitch)
                vision.draw_eye_landmarks(frame, landmarks)

            state_name = metrics.state.name
            score = metrics.drowsiness_score

            arduino.send_state(int(metrics.state))

            now = time.time()
            frame_times.append(now)
            if len(frame_times) > FPS_WINDOW_FRAMES:
                frame_times = frame_times[-FPS_WINDOW_FRAMES:]
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            if now - last_trend_log > TREND_LOG_INTERVAL_S:
                trend = metrics.predict_fatigue_trend()
                if trend is not None:
                    logger.info("Projected drowsiness in 5 min: %.0f/100", trend)
                last_trend_log = now

            frame = draw_hud(frame, state_name, score, metrics.smoothed_ear, fps,
                             landmarks is not None)

            data_log.log(
                raw_ear=raw_ear,
                smoothed_ear=metrics.smoothed_ear,
                perclos=metrics.perclos,
                blink_rate=metrics.blink_rate,
                drowsiness_score=score,
                state=state_name,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            if gui:
                gui.update_frame(
                    frame=frame,
                    state=state_name,
                    score=score,
                    ear=metrics.smoothed_ear,
                    perclos=metrics.perclos,
                    blink_rate=metrics.blink_rate,
                    fps=fps,
                    elapsed_s=now - start_time,
                )
                gui.tick()
            else:
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed_loop = time.time() - loop_start
            if frame_interval > elapsed_loop:
                time.sleep(frame_interval - elapsed_loop)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        logger.info("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        vision.release()
        arduino.disconnect()
        data_log.close()
        if gui:
            gui.destroy()
        logger.info(
            "Session complete, %d rows logged to %s", data_log.rows_written, data_log.filepath
        )


if __name__ == "__main__":
    main()
