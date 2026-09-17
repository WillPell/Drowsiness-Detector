/*
 * circuit_tester.ino - hardware bring-up sketch
 *
 * Cycles the RGB LED through red, green and blue, then beeps the buzzer,
 * to confirm the wiring before flashing drowsiness_controller.ino.
 * Pinout matches the main sketch.
 */

const int PIN_RED    = 9;
const int PIN_GREEN  = 10;
const int PIN_BLUE   = 11;
const int PIN_BUZZER = 8;

void setup() {
  pinMode(PIN_RED,    OUTPUT);
  pinMode(PIN_GREEN,  OUTPUT);
  pinMode(PIN_BLUE,   OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
}

void loop() {
  setLED(255, 0, 0);
  delay(1000);
  setLED(0, 255, 0);
  delay(1000);
  setLED(0, 0, 255);
  delay(1000);
  setLED(0, 0, 0);

  digitalWrite(PIN_BUZZER, HIGH);
  delay(500);
  digitalWrite(PIN_BUZZER, LOW);
  delay(500);
}

void setLED(int r, int g, int b) {
  analogWrite(PIN_RED,   r);
  analogWrite(PIN_GREEN, g);
  analogWrite(PIN_BLUE,  b);
}
