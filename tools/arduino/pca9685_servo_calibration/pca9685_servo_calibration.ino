#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN 102
#define SERVOMAX 512
#define POT_PIN A0
#define CAL_CHANNEL 0

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(50);
  Serial.println("Ready - turn the potentiometer");
}

void setAngle(int channel, int angle) {
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}

void loop() {
  int potValue = analogRead(POT_PIN);
  int angle = map(potValue, 0, 1023, 0, 180);
  setAngle(CAL_CHANNEL, angle);
  Serial.print("Angle: ");
  Serial.println(angle);
  delay(100);
}
