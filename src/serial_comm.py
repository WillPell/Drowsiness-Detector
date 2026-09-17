import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

STATE_COMMANDS = {0: b"0", 1: b"1", 2: b"2", 3: b"3"}

HEARTBEAT_INTERVAL_S = 1.0


class ArduinoSerial:
    def __init__(
        self,
        port: str = "COM3",
        baud: int = 9600,
        timeout: float = 1.0,
        min_send_interval: float = 0.1,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_S,
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.min_send_interval = min_send_interval
        self.heartbeat_interval = heartbeat_interval

        self._serial = None
        self._connected = False
        self._last_send_time = 0.0
        self._last_state_sent: Optional[int] = None

    def connect(self) -> bool:
        try:
            import serial as pyserial

            self._serial = pyserial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
            )
            time.sleep(2.0)
            self._connected = True
            logger.info("Arduino connected on %s @ %d baud", self.port, self.baud)
            return True
        except Exception as exc:
            logger.warning("Could not open %s: %s, running in STUB mode", self.port, exc)
            self._connected = False
            return False

    def disconnect(self):
        if self._serial and self._serial.is_open:
            self._safe_write(b"0")
            self._serial.close()
            logger.info("Arduino disconnected")
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_state(self, state_value: int) -> bool:
        now = time.time()
        changed = state_value != self._last_state_sent
        due = (now - self._last_send_time) >= self.heartbeat_interval

        if not changed and not due:
            return False
        if changed and (now - self._last_send_time) < self.min_send_interval:
            return False

        cmd = STATE_COMMANDS.get(state_value, b"0")
        if not self._safe_write(cmd):
            return False

        self._last_state_sent = state_value
        self._last_send_time = now
        logger.debug("Sent state %d (%s)", state_value, cmd)
        return True

    def ping(self) -> bool:
        if not self._connected:
            return False
        self._safe_write(b"P")
        try:
            return self._serial.readline().decode().strip() == "OK"
        except Exception:
            return False

    def _safe_write(self, data: bytes) -> bool:
        if not self._connected:
            logger.debug("STUB send: %s", data)
            return False
        try:
            self._serial.write(data)
            return True
        except Exception as exc:
            logger.error("Serial write error: %s", exc)
            self._connected = False
            return False
