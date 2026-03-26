# global constants all in one file for easy tuning


# camera 
CAMERA_INDEX = 0 # default webcam
CAMERA_WIDTH = 640 # resolution width
CAMERA_HEIGHT = 480 # resolution (height)

# serial 
SERIAL_PORT = "COM3" # i use COM3, this only works on windows as far as i'm aware
SERIAL_BAUD = 9600

# vision
FACE_DETECTION_CONFIDENCE = 0.6
FACE_TRACKING_CONFIDENCE = 0.6

# metrics (look into these more)
EAR_THRESHOLD = 0.22 # if below this eyes closed
PERCLOS_WINDOW_S = 30.0 # window for perlocs
EMA_ALPHA = 0.30 # EAR smoothing factor

# calibration
CALIBRATION_DURATION_S = 5.0  # length of calibration (s), 5 is usually pretty good (can be shorter)

# GUI
GUI_WIDTH = 720
GUI_HEIGHT = 480
GRAPH_HISTORY_S = 60.0 # graph shows a minute

# logging
LOG_DIR = "logs"

# performance (on slow machines skip frames)
FRAME_SKIP = 0                # process every Nth frame (0 = no skip)
TARGET_FPS = 30               # cap frame rate to save CPU
