/*
 * INDUSTRIAL CONVEYOR CONTROL SYSTEM
 * Menggunakan Stepper Motor dengan AccelStepper
 * Terintegrasi dengan Python Web App via Serial USB
 */

#include <AccelStepper.h>

// ========== PIN DEFINITIONS ==========
#define STEP_PIN     3    // PUL (Step) pin
#define DIR_PIN      2    // DIR (Direction) pin
#define ENABLE_PIN   4    // Enable pin (opsional)

// Indicator LEDs
#define LED_GREEN    7
#define LED_RED      8
#define LED_YELLOW   9
#define BUZZER       10

// ========== CONVEYOR PARAMETERS ==========
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// Stepper parameters
float MAX_SPEED = 2000.0;      // Maximum speed (steps/second)
float MIN_SPEED = 100.0;        // Minimum speed
float DEFAULT_SPEED = 800.0;    // Default speed

// Conveyor status
bool conveyor_running = false;
bool defect_detected = false;
bool auto_stop_enabled = true;
int target_speed_percent = 70;   // 0-100%

// Speed mapping
float speed_map[101];

// Serial command buffer
String inputString = "";
boolean stringComplete = false;

// ========== HELPER FUNCTIONS ==========

void initSpeedMap() {
  for (int i = 0; i <= 100; i++) {
    speed_map[i] = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (i / 100.0);
  }
}

void setConveyorSpeed(int percent) {
  target_speed_percent = constrain(percent, 0, 100);
  
  if (target_speed_percent == 0) {
    stepper.stop();
    stepper.setSpeed(0);
    conveyor_running = false;
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, HIGH);
  } else {
    float speed = speed_map[target_speed_percent];
    stepper.setSpeed(speed);
    if (conveyor_running) {
      // Speed akan berubah di loop berikutnya
    }
  }
  
  Serial.print("SPEED_CHANGED:");
  Serial.print(target_speed_percent);
  Serial.print(",");
  Serial.println(speed_map[target_speed_percent]);
}

void startConveyor() {
  if (defect_detected && auto_stop_enabled) {
    Serial.println("ERROR:Cannot start - defect detected!");
    return;
  }
  
  conveyor_running = true;
  
  if (target_speed_percent > 0) {
    stepper.setSpeed(speed_map[target_speed_percent]);
  } else {
    stepper.setSpeed(DEFAULT_SPEED);
    target_speed_percent = 70;
  }
  
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_RED, LOW);
  
  Serial.print("CONVEYOR_STARTED:SPEED=");
  Serial.print(target_speed_percent);
  Serial.print("% (");
  Serial.print(stepper.speed());
  Serial.println(" steps/sec)");
}

void stopConveyor(String reason = "Manual") {
  conveyor_running = false;
  stepper.stop();
  stepper.setSpeed(0);
  
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, HIGH);
  
  Serial.print("CONVEYOR_STOPPED:");
  Serial.println(reason);
}

void onDefectDetected() {
  defect_detected = true;
  digitalWrite(LED_YELLOW, HIGH);
  
  // Buzzer 3 beep
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER, HIGH);
    delay(200);
    digitalWrite(BUZZER, LOW);
    delay(100);
  }
  
  if (auto_stop_enabled && conveyor_running) {
    stopConveyor("Defect Detected (Auto-Stop)");
    Serial.println("DEFECT_ALERT:AUTO_STOP_TRIGGERED");
  } else {
    Serial.println("DEFECT_ALERT:Defect detected");
  }
}

void resetDefect() {
  defect_detected = false;
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(BUZZER, LOW);
  Serial.println("DEFECT_RESET:Status cleared");
}

void setAutoStop(bool enabled) {
  auto_stop_enabled = enabled;
  Serial.print("AUTO_STOP:");
  Serial.println(enabled ? "ENABLED" : "DISABLED");
}

void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  
  Serial.print("RECEIVED: ");
  Serial.println(cmd);
  
  if (cmd == "START") {
    startConveyor();
  }
  else if (cmd == "STOP") {
    stopConveyor("Manual");
  }
  else if (cmd.startsWith("SPEED:")) {
    int speed = cmd.substring(6).toInt();
    setConveyorSpeed(speed);
  }
  else if (cmd == "DEFECT") {
    onDefectDetected();
  }
  else if (cmd == "RESET") {
    resetDefect();
  }
  else if (cmd == "AUTO_STOP:ON") {
    setAutoStop(true);
  }
  else if (cmd == "AUTO_STOP:OFF") {
    setAutoStop(false);
  }
  else if (cmd == "STATUS") {
    Serial.print("STATUS:");
    Serial.print(conveyor_running ? "RUNNING" : "STOPPED");
    Serial.print(",");
    Serial.print("SPEED:");
    Serial.print(target_speed_percent);
    Serial.print(",");
    Serial.print("DEFECT:");
    Serial.print(defect_detected ? "YES" : "NO");
    Serial.print(",");
    Serial.print("AUTO_STOP:");
    Serial.println(auto_stop_enabled ? "ON" : "OFF");
  }
  else if (cmd == "HELP") {
    Serial.println("Commands: START, STOP, SPEED:<0-100>, DEFECT, RESET, AUTO_STOP:ON/OFF, STATUS");
  }
  else if (cmd != "") {
    Serial.print("UNKNOWN:");
    Serial.println(cmd);
  }
}

// ========== SETUP ==========
void setup() {
  // Initialize pins
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  
  // Initialize stepper
  initSpeedMap();
  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setSpeed(0);
  
  // Enable motor driver
  digitalWrite(ENABLE_PIN, LOW);
  
  // Initial LED state
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(BUZZER, LOW);
  
  // Start serial
  Serial.begin(9600);
  
  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("========================================");
  Serial.println("CONVEYOR CONTROL SYSTEM READY");
  Serial.println("Type HELP for commands");
  Serial.println("========================================");
  
  // Blink green LED to indicate ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_GREEN, HIGH);
    delay(200);
    digitalWrite(LED_GREEN, LOW);
    delay(200);
  }
}

// ========== MAIN LOOP ==========
void loop() {
  // Process serial commands
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // Run stepper if conveyor is running
  if (conveyor_running) {
    stepper.runSpeed();
  }
}

// Serial event handler
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else if (inChar != '\r') {
      inputString += inChar;
    }
  }
}