/*
 * drowsiness_controller.ino - Arduino Mega2560 alert controller
 *
 * Serial protocol (9600 baud):
 *   '0' AWAKE     LED green,  buzzer off
 *   '1' SLIGHT    LED yellow, buzzer off
 *   '2' DROWSY    LED orange, buzzer off
 *   '3' CRITICAL  LED red,    buzzer pulsing
 *   'P' PING      replies "OK\n"
 *
 * Wiring:
 *   Pin 9  -> RGB LED red    (PWM, 220R)
 *   Pin 10 -> RGB LED green  (PWM, 220R)
 *   Pin 11 -> RGB LED blue   (PWM, 220R)
 *   Pin 8  -> piezo buzzer   (100R)
 *   GND    -> LED common cathode and buzzer ground
 *
 * The loop never blocks: colour fades and buzzer pulses are driven by millis()
 * so incoming serial is read promptly. The host re-sends the current state at
 * roughly 1 Hz, so a gap longer than WATCHDOG_TIMEOUT means the host has gone
 * away and the board falls back to AWAKE.
 */

const int PIN_RED    = 9;
const int PIN_GREEN  = 10;
const int PIN_BLUE   = 11;
const int PIN_BUZZER = 8;

enum AlertState {
  STATE_AWAKE    = 0,
  STATE_SLIGHT   = 1,
  STATE_DROWSY   = 2,
  STATE_CRITICAL = 3
};

const int COLOURS[][3] = {
  {  0, 255,   0},
  {255, 200,   0},
  {255, 100,   0},
  {255,   0,   0},
};

const unsigned long BUZZ_INTERVAL    = 500;
const unsigned long LED_FADE_TIME    = 200;
const unsigned long FADE_STEP_MS     = 10;
const unsigned long WATCHDOG_TIMEOUT = 5000;

AlertState currentState = STATE_AWAKE;

float curR = 0, curG = 255, curB = 0;

bool buzzerOn = false;
unsigned long lastBuzz = 0;
unsigned long lastValidCmd = 0;
unsigned long lastFadeUpdate = 0;


void setup() {
  pinMode(PIN_RED,    OUTPUT);
  pinMode(PIN_GREEN,  OUTPUT);
  pinMode(PIN_BLUE,   OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  setLED(0, 255, 0);
  digitalWrite(PIN_BUZZER, LOW);

  Serial.begin(9600);
  lastValidCmd = millis();
}


void loop() {
  unsigned long now = millis();

  while (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'P') {
      Serial.println("OK");
      lastValidCmd = now;
    } else if (cmd >= '0' && cmd <= '3') {
      currentState = (AlertState)(cmd - '0');
      lastValidCmd = now;
    }
    // Anything else is line noise and is ignored.
  }

  if ((now - lastValidCmd) > WATCHDOG_TIMEOUT) {
    currentState = STATE_AWAKE;
  }

  if (now - lastFadeUpdate >= FADE_STEP_MS) {
    lastFadeUpdate = now;
    float step = (float)FADE_STEP_MS / LED_FADE_TIME;
    curR += (COLOURS[currentState][0] - curR) * step;
    curG += (COLOURS[currentState][1] - curG) * step;
    curB += (COLOURS[currentState][2] - curB) * step;
    setLED((int)curR, (int)curG, (int)curB);
  }

  if (currentState == STATE_CRITICAL) {
    if (now - lastBuzz >= BUZZ_INTERVAL) {
      buzzerOn = !buzzerOn;
      digitalWrite(PIN_BUZZER, buzzerOn ? HIGH : LOW);
      lastBuzz = now;
    }
  } else if (buzzerOn) {
    buzzerOn = false;
    digitalWrite(PIN_BUZZER, LOW);
  }
}


void setLED(int r, int g, int b) {
  analogWrite(PIN_RED,   constrain(r, 0, 255));
  analogWrite(PIN_GREEN, constrain(g, 0, 255));
  analogWrite(PIN_BLUE,  constrain(b, 0, 255));
}
