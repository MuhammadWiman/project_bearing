#include <AccelStepper.h>

#define STEP_PIN 3
#define DIR_PIN 2
#define ENABLE_PIN 4

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

bool motorRunning = false;
int motorSpeed = 800;
String inputString = "";

void setup() {
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  
  stepper.setMaxSpeed(2000);
  stepper.setSpeed(0);
  
  Serial.begin(9600);
  
  pinMode(13, OUTPUT);
  for(int i=0; i<3; i++) {
    digitalWrite(13, HIGH);
    delay(200);
    digitalWrite(13, LOW);
    delay(200);
  }
  
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      if (inputString.length() > 0) {
        inputString.trim();
        
        if (inputString == "START") {
          motorRunning = true;
          stepper.setSpeed(motorSpeed);
          Serial.println("MOTOR_RUNNING");
        }
        else if (inputString == "STOP") {
          motorRunning = false;
          stepper.stop();
          stepper.setSpeed(0);
          Serial.println("MOTOR_STOPPED");
        }
        else if (inputString.startsWith("SPEED:")) {
          int val = inputString.substring(6).toInt();
          motorSpeed = map(val, 0, 100, 100, 2000);
          if (motorRunning) {
            stepper.setSpeed(motorSpeed);
          }
          Serial.print("SPEED_SET:");
          Serial.println(motorSpeed);
        }
        else if (inputString == "HELP") {
          Serial.println("Commands: START, STOP, SPEED:0-100");
        }
      }
      inputString = "";
    }
    else if (inChar != '\r') {
      inputString += inChar;
    }
  }
  
  if (motorRunning) {
    stepper.runSpeed();
  }
}