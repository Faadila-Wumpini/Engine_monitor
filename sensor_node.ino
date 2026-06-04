/*
 * EngineIQ — ESP32 Sensor Firmware
 * IoT & ML-Based Predictive Car Engine Health Monitoring System
 * Group 11 — Department of Computer Science, KNUST
 * 
 * SENSORS:
 *   MPU6050 Accelerometer → GPIO 21 (SDA), GPIO 22 (SCL)  [I2C]
 *   DS18B20 Temperature   → GPIO 4                         [1-Wire]
 * 
 * WHAT THIS CODE DOES:
 *   1. Connects to your Wi-Fi
 *   2. Connects to MQTT broker (your laptop running mosquitto)
 *   3. Reads vibration from MPU6050 at 50Hz (every 20ms)
 *   4. Reads temperature from DS18B20 at 1Hz (every 1000ms)
 *   5. Packages data as JSON and publishes to MQTT topic "engine/sensors"
 * 
 * LIBRARIES TO INSTALL (in Arduino IDE → Tools → Manage Libraries):
 *   - MPU6050 by Electronic Cats
 *   - DallasTemperature by Miles Burton
 *   - OneWire by Paul Stoffregen
 *   - PubSubClient by Nick O'Leary
 * 
 * HOW TO UPLOAD:
 *   1. Open Arduino IDE
 *   2. Go to Tools → Board → ESP32 Arduino → ESP32 Dev Module
 *   3. Go to Tools → Port → select your ESP32 COM port
 *   4. Click Upload (the → arrow button)
 */

#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <MPU6050.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ── SETTINGS — CHANGE THESE TO MATCH YOUR SETUP ───────────────────────────
const char* WIFI_SSID       = "Faady's galaxy";      // your Wi-Fi name
const char* WIFI_PASSWORD   = "XAVIER22";  // your Wi-Fi password
const char* MQTT_BROKER     = "192.168.108.249";        // your laptop's IP address
const int   MQTT_PORT       = 1883;
const char* MQTT_TOPIC      = "engine/sensors";
const char* DEVICE_ID       = "ESP32_GROUP11";

// ── PIN DEFINITIONS ────────────────────────────────────────────────────────
#define SDA_PIN         21    // MPU6050 SDA
#define SCL_PIN         22    // MPU6050 SCL
#define TEMP_PIN         4    // DS18B20 data pin

// ── SAMPLING RATES ─────────────────────────────────────────────────────────
#define VIBRATION_INTERVAL   20     // ms → 50Hz (every 20 milliseconds)
#define TEMP_INTERVAL      1000     // ms → 1Hz  (every 1 second)
#define RECONNECT_INTERVAL  5000    // ms → try reconnect every 5 seconds

// ── SENSOR OBJECTS ─────────────────────────────────────────────────────────
MPU6050 mpu;
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// ── NETWORK OBJECTS ────────────────────────────────────────────────────────
WiFiClient   espClient;
PubSubClient mqttClient(espClient);

// ── TIMING VARIABLES ───────────────────────────────────────────────────────
unsigned long lastVibrationTime = 0;
unsigned long lastTempTime      = 0;
unsigned long lastReconnectTime = 0;

// ── SENSOR READINGS (global so both loops can access them) ─────────────────
float accX = 0.0, accY = 0.0, accZ = 0.0;  // acceleration in g
float temperature = 0.0;                     // temperature in Celsius
int   readingCount = 0;                      // counter for debugging


// ── WIFI SETUP ─────────────────────────────────────────────────────────────
void setupWiFi() {
  Serial.println("\n── Connecting to Wi-Fi ──────────────────────");
  Serial.print("  SSID: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 30) {
      Serial.println("\n❌ Wi-Fi connection failed. Restarting...");
      ESP.restart();
    }
  }
  
  Serial.println("\n✓ Wi-Fi connected!");
  Serial.print("  IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("  Signal strength: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
}


// ── MPU6050 SETUP ──────────────────────────────────────────────────────────
void setupMPU6050() {
  Serial.println("\n── Initialising MPU6050 ─────────────────────");
  
  Wire.begin(SDA_PIN, SCL_PIN);
  mpu.initialize();
  
  if (mpu.testConnection()) {
    Serial.println("  ✓ MPU6050 connected successfully");
    
    // Set accelerometer range to ±2g (most sensitive — good for engine vibration)
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    Serial.println("  ✓ Accelerometer range: ±2g");
    
    // Set DLPF (Digital Low Pass Filter) to reduce noise
    mpu.setDLPFMode(MPU6050_DLPF_BW_20);
    Serial.println("  ✓ DLPF mode: 20Hz bandwidth");
    
  } else {
    Serial.println("  ❌ MPU6050 connection FAILED");
    Serial.println("  Check wiring: SDA→GPIO21, SCL→GPIO22, VCC→3.3V, GND→GND");
  }
}


// ── DS18B20 SETUP ──────────────────────────────────────────────────────────
void setupDS18B20() {
  Serial.println("\n── Initialising DS18B20 ─────────────────────");
  
  tempSensor.begin();
  int deviceCount = tempSensor.getDeviceCount();
  
  Serial.print("  Devices found on 1-Wire bus: ");
  Serial.println(deviceCount);
  
  if (deviceCount > 0) {
    // Set resolution to 12-bit (most precise, ±0.0625°C)
    tempSensor.setResolution(12);
    Serial.println("  ✓ DS18B20 found and ready");
    Serial.println("  ✓ Resolution: 12-bit (±0.0625°C)");
  } else {
    Serial.println("  ❌ No DS18B20 found!");
    Serial.println("  Check wiring: DATA→GPIO4, VCC→3.3V, GND→GND");
    Serial.println("  Make sure 4.7kΩ pull-up resistor is on DATA line");
  }
}


// ── MQTT CONNECT ───────────────────────────────────────────────────────────
bool connectMQTT() {
  Serial.print("\n── Connecting to MQTT broker ");
  Serial.print(MQTT_BROKER);
  Serial.print(":");
  Serial.println(MQTT_PORT);
  
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setBufferSize(512);  // increase buffer for larger JSON payloads
  
  // Try to connect with a client ID
  String clientId = String(DEVICE_ID) + "_" + String(random(0xffff), HEX);
  
  if (mqttClient.connect(clientId.c_str())) {
    Serial.println("  ✓ MQTT connected!");
    Serial.print("  Publishing to topic: ");
    Serial.println(MQTT_TOPIC);
    
    // Publish a startup message
    String startMsg = "{\"event\":\"startup\",\"device\":\"" + String(DEVICE_ID) + "\"}";
    mqttClient.publish("engine/status", startMsg.c_str());
    return true;
    
  } else {
    Serial.print("  ❌ MQTT connection failed. Error code: ");
    Serial.println(mqttClient.state());
    Serial.println("  Make sure mosquitto is running on your laptop:");
    Serial.println("  → Open a terminal and run: mosquitto -v");
    return false;
  }
}


// ── READ MPU6050 ───────────────────────────────────────────────────────────
void readVibration() {
  int16_t rawAccX, rawAccY, rawAccZ;
  int16_t rawGyroX, rawGyroY, rawGyroZ;  // we read but don't use gyro
  
  mpu.getMotion6(&rawAccX, &rawAccY, &rawAccZ,
                 &rawGyroX, &rawGyroY, &rawGyroZ);
  
  // Convert raw 16-bit values to g-force
  // ±2g range → divide by 16384 (from datasheet)
  accX = rawAccX / 16384.0;
  accY = rawAccY / 16384.0;
  accZ = rawAccZ / 16384.0;
}


// ── READ DS18B20 ───────────────────────────────────────────────────────────
void readTemperature() {
  tempSensor.requestTemperatures();
  float reading = tempSensor.getTempCByIndex(0);
  
  // Validate reading (-127 means sensor error)
  if (reading != DEVICE_DISCONNECTED_C && reading > -100) {
    temperature = reading;
  } else {
    Serial.println("  ⚠ Temperature sensor error — check connection");
  }
}


// ── BUILD AND PUBLISH JSON PAYLOAD ─────────────────────────────────────────
void publishData() {
  // Build JSON string manually (faster than using ArduinoJson library)
  // Format: {"accX":0.0123,"accY":-0.0045,"accZ":0.9940,"temp":74.25,"id":1}
  
  char payload[256];
  snprintf(payload, sizeof(payload),
    "{"
    "\"accX\":%.4f,"
    "\"accY\":%.4f,"
    "\"accZ\":%.4f,"
    "\"temp\":%.2f,"
    "\"id\":%d"
    "}",
    accX, accY, accZ, temperature, readingCount
  );
  
  // Publish to MQTT
  bool success = mqttClient.publish(MQTT_TOPIC, payload);
  
  // Print to Serial Monitor every 50 readings (every 1 second at 50Hz)
  if (readingCount % 50 == 0) {
    Serial.print("  [");
    Serial.print(readingCount);
    Serial.print("] AccX:");
    Serial.print(accX, 4);
    Serial.print("  AccY:");
    Serial.print(accY, 4);
    Serial.print("  AccZ:");
    Serial.print(accZ, 4);
    Serial.print("  Temp:");
    Serial.print(temperature, 2);
    Serial.print("°C");
    Serial.println(success ? "  ✓ sent" : "  ✗ FAILED");
  }
  
  readingCount++;
}


// ── SETUP ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n");
  Serial.println("╔═══════════════════════════════════════════╗");
  Serial.println("║  EngineIQ — IoT Engine Monitor            ║");
  Serial.println("║  Group 11, KNUST — CS Final Year          ║");
  Serial.println("╚═══════════════════════════════════════════╝");
  
  // Initialize all components
  setupWiFi();
  setupMPU6050();
  setupDS18B20();
  
  // Connect to MQTT
  connectMQTT();
  
  // Take first temperature reading
  readTemperature();
  
  Serial.println("\n── Starting data acquisition ────────────────");
  Serial.println("  Vibration: 50Hz (every 20ms)");
  Serial.println("  Temperature: 1Hz (every 1000ms)");
  Serial.println("  Publishing to: engine/sensors");
  Serial.println("────────────────────────────────────────────\n");
}


// ── MAIN LOOP ──────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();
  
  // ── Keep MQTT connection alive ──────────────────────────────────────────
  if (!mqttClient.connected()) {
    if (now - lastReconnectTime >= RECONNECT_INTERVAL) {
      lastReconnectTime = now;
      Serial.println("⚠ MQTT disconnected — reconnecting...");
      connectMQTT();
    }
    return;  // don't read sensors if not connected
  }
  mqttClient.loop();  // handle incoming messages and keep-alive
  
  // ── Read vibration at 50Hz (every 20ms) ────────────────────────────────
  if (now - lastVibrationTime >= VIBRATION_INTERVAL) {
    lastVibrationTime = now;
    readVibration();
    publishData();    // publish every vibration reading
  }
  
  // ── Read temperature at 1Hz (every 1000ms) ─────────────────────────────
  if (now - lastTempTime >= TEMP_INTERVAL) {
    lastTempTime = now;
    readTemperature();
  }
}
