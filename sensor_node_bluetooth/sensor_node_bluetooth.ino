/*
 * EngineIQ — ESP32 BLE Sensor Firmware (Bluetooth Version)
 * IoT & ML-Based Predictive Car Engine Health Monitoring System
 * Group 11 — Department of Computer Science, KNUST
 *
 * SENSORS:
 *   MPU6050 Accelerometer → GPIO 21 (SDA), GPIO 22 (SCL)  [I2C]
 *   DS18B20 Temperature   → GPIO 4                         [1-Wire]
 *
 * WHAT THIS DOES:
 *   1. Reads vibration from MPU6050 at 50Hz
 *   2. Reads temperature from DS18B20 at 1Hz
 *   3. Sends data as BLE notifications to a connected phone/laptop
 *   4. No Wi-Fi or MQTT needed — works anywhere
 *
 * LIBRARIES TO INSTALL:
 *   - MPU6050 by Electronic Cats
 *   - DallasTemperature by Miles Burton
 *   - OneWire by Paul Stoffregen
 *   - ESP32 BLE Arduino (comes with the ESP32 board package)
 */

#include <Wire.h>
#include <MPU6050.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── DEVICE IDENTITY ────────────────────────────────────────────────────────────
// CHANGE THIS before flashing a second/third/etc. unit — it's what lets the
// system tell multiple ESP32s apart (advertised BLE name + tagged in every
// reading sent). Two units both left as "ESP32-01" are indistinguishable to
// inference.py and to anomaly_logs in Supabase.
#define DEVICE_ID "ESP32-01"

// ── PIN DEFINITIONS ───────────────────────────────────────────────────────────
#define SDA_PIN   21
#define SCL_PIN   22
#define TEMP_PIN   4

// ── SAMPLING RATES ────────────────────────────────────────────────────────────
#define VIBRATION_INTERVAL   20     // 50Hz
#define TEMP_INTERVAL      1000     // 1Hz

// ── BLE SERVICE AND CHARACTERISTIC UUIDs ─────────────────────────────────────
// These are unique IDs that the phone uses to identify the sensor data stream
#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define SENSOR_CHAR_UUID    "abcd1234-ab12-ab12-ab12-abcdef123456"

// ── SENSOR OBJECTS ────────────────────────────────────────────────────────────
MPU6050 mpu;
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// ── BLE OBJECTS ───────────────────────────────────────────────────────────────
BLEServer*         pServer         = NULL;
BLECharacteristic* pSensorChar     = NULL;
bool               deviceConnected = false;
bool               oldConnected    = false;

// ── SENSOR READINGS ───────────────────────────────────────────────────────────
float accX = 0.0, accY = 0.0, accZ = 0.0;
float temperature = 0.0;
int   readingCount = 0;

// ── TIMING ────────────────────────────────────────────────────────────────────
unsigned long lastVibrationTime = 0;
unsigned long lastTempTime      = 0;


// ── BLE CONNECTION CALLBACKS ──────────────────────────────────────────────────
// These functions run automatically when a device connects or disconnects
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("✓ Device connected via Bluetooth!");
    Serial.println("  Sending sensor data...");
  }

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("⚠ Device disconnected.");
    Serial.println("  Waiting for new connection...");
  }
};


// ── SETUP ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n╔═══════════════════════════════════════════╗");
  Serial.println("║  EngineIQ — Bluetooth Engine Monitor      ║");
  Serial.println("║  Group 11, KNUST                          ║");
  Serial.println("╚═══════════════════════════════════════════╝\n");

  // ── Initialise MPU6050 ────────────────────────────────────────────────────
  Serial.println("── Initialising MPU6050 ─────────────────────");
  Wire.begin(SDA_PIN, SCL_PIN);
  mpu.initialize();
  if (mpu.testConnection()) {
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);  // ±2g
    mpu.setDLPFMode(MPU6050_DLPF_BW_20);             // noise filter
    Serial.println("  ✓ MPU6050 ready");
  } else {
    Serial.println("  ❌ MPU6050 not found — check wiring!");
  }

  // ── Initialise DS18B20 ────────────────────────────────────────────────────
  Serial.println("── Initialising DS18B20 ─────────────────────");
  tempSensor.begin();
  if (tempSensor.getDeviceCount() > 0) {
    tempSensor.setResolution(12);
    Serial.println("  ✓ DS18B20 ready");
  } else {
    Serial.println("  ❌ DS18B20 not found — check wiring!");
  }

  // ── Initialise BLE ────────────────────────────────────────────────────────
  Serial.println("── Initialising Bluetooth BLE ───────────────");
  // "EngineIQ_Sensor_" prefix is shared across every unit (inference.py scans
  // for that prefix, not an exact name) — DEVICE_ID after it is what makes
  // each physical ESP32 distinguishable, both here and in the readings below.
  String bleName = String("EngineIQ_Sensor_") + DEVICE_ID;
  BLEDevice::init(bleName.c_str());

  // Create BLE server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create BLE service
  BLEService* pService = pServer->createService(SERVICE_UUID);

  // Create BLE characteristic — this is the "channel" that sends sensor data
  pSensorChar = pService->createCharacteristic(
    SENSOR_CHAR_UUID,
    BLECharacteristic::PROPERTY_READ   |
    BLECharacteristic::PROPERTY_NOTIFY  // NOTIFY means it pushes data automatically
  );
  pSensorChar->addDescriptor(new BLE2902());

  // Start the service and start advertising
  pService->start();
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.print("  ✓ BLE advertising as '");
  Serial.print(bleName);
  Serial.println("'");
  Serial.print("  → Device ID tagged in every reading: ");
  Serial.println(DEVICE_ID);
  Serial.println("\n── Waiting for connection ───────────────────\n");
}


// ── MAIN LOOP ─────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // ── Read vibration at 50Hz ────────────────────────────────────────────────
  if (now - lastVibrationTime >= VIBRATION_INTERVAL) {
    lastVibrationTime = now;

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    accX = ax / 16384.0;
    accY = ay / 16384.0;
    accZ = az / 16384.0;

    // ── Send over Bluetooth if connected ──────────────────────────────────
    if (deviceConnected) {
      // Build the JSON payload
      char payload[200];
      snprintf(payload, sizeof(payload),
        "{\"accX\":%.4f,\"accY\":%.4f,\"accZ\":%.4f,\"temp\":%.2f,\"id\":%d,\"device_id\":\"%s\"}",
        accX, accY, accZ, temperature, readingCount, DEVICE_ID
      );

      // Send via BLE notification
      pSensorChar->setValue(payload);
      pSensorChar->notify();

      // Print to Serial every 50 readings (every 1 second)
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
        Serial.println("°C  ✓ sent via BLE");
      }

      readingCount++;
    }
  }

  // ── Read temperature at 1Hz ───────────────────────────────────────────────
  if (now - lastTempTime >= TEMP_INTERVAL) {
    lastTempTime = now;
    tempSensor.requestTemperatures();
    float reading = tempSensor.getTempCByIndex(0);
    if (reading != DEVICE_DISCONNECTED_C && reading > -100) {
      temperature = reading;
    }
  }

  // ── Handle reconnection after disconnect ──────────────────────────────────
  if (!deviceConnected && oldConnected) {
    delay(500);
    pServer->startAdvertising();  // restart advertising
    Serial.println("  Advertising restarted — waiting for reconnection...");
    oldConnected = false;
  }

  if (deviceConnected && !oldConnected) {
    oldConnected = true;
  }
}
