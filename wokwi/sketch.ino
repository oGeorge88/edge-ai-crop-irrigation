/*
 * Edge AI Crop Disease Detection & Automated Irrigation System
 * Wokwi Simulation — ESP32-S3
 * Author: George Obinna Oguejiofor | KMITL
 *
 * Components:
 *   Potentiometer 1 → GPIO1  Soil moisture sensor (0% dry → 100% wet)
 *   Potentiometer 2 → GPIO2  Rainfall sensor       (0 mm → 10 mm)
 *   DHT22           → GPIO4  Temperature + humidity
 *   Relay module    → GPIO5  Water pump
 *   LED (red)       → GPIO6  Pump ON indicator
 *   Push button     → GPIO7  Cycle disease mode
 *   LCD 16×2 I2C   → GPIO8/9 (SDA/SCL)  Status display
 *
 * Press the green button to cycle disease mode:
 *   NORMAL   → irrigate when soil < 40%  (healthy / viral disease)
 *   REDUCE   → irrigate when soil < 25%  (fungal / bacterial — wet conditions spread disease)
 *   INCREASE → irrigate when soil < 55%  (spider mites — prefer dry, so keep moist)
 *
 * Emergency override: always irrigate when soil < 20% regardless of mode.
 * Rainfall guard: skip irrigation when rain > 5 mm in last 24h.
 *
 * Serial monitor (115200 baud) streams full telemetry every second.
 */

#include <DHT.h>
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

// ── Pins ──────────────────────────────────────────────────────────────────
#define SOIL_PIN    1
#define RAIN_PIN    2
#define DHT_PIN     4
#define RELAY_PIN   5
#define LED_PIN     6
#define BUTTON_PIN  7
#define SDA_PIN     8
#define SCL_PIN     9

// ── Irrigation thresholds (% soil moisture) ───────────────────────────────
#define THRESH_CRITICAL  20.0f
#define THRESH_REDUCE    25.0f
#define THRESH_NORMAL    40.0f
#define THRESH_INCREASE  55.0f

// ── Disease modes ─────────────────────────────────────────────────────────
enum DiseaseMode { NORMAL = 0, REDUCE = 1, INCREASE = 2 };

const char* MODE_TAG[]     = { "NORMAL  ", "REDUCE  ", "INCREASE" };
const char* DISEASE_NAME[] = { "Healthy         ",
                                "Early Blight    ",
                                "Spider Mites    " };
const float THRESHOLD[]    = { THRESH_NORMAL, THRESH_REDUCE, THRESH_INCREASE };

DiseaseMode diseaseMode  = NORMAL;
bool        lastBtn      = HIGH;
uint32_t    lastDebounce = 0;
uint32_t    modeStartMs  = 0;

DHT               dht(DHT_PIN, DHT22);
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ── Helpers ───────────────────────────────────────────────────────────────
float readSoilMoisture() {
  return (float)analogRead(SOIL_PIN) / 4095.0f * 100.0f;
}

float readRainfall() {
  return (float)analogRead(RAIN_PIN) / 4095.0f * 10.0f;
}

bool shouldIrrigate(float soil, float rain, DiseaseMode mode) {
  if (soil < THRESH_CRITICAL) return true;   // Emergency
  if (rain > 5.0f)            return false;  // Already raining
  return soil < THRESHOLD[mode];
}

// ── Setup ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(RELAY_PIN,  OUTPUT);
  pinMode(LED_PIN,    OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_PIN,   LOW);

  dht.begin();

  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print(" Edge AI Irrig. ");
  lcd.setCursor(0, 1); lcd.print("  KMITL  2026   ");
  delay(2000);
  lcd.clear();

  Serial.println(F("\n============================================"));
  Serial.println(F("  Edge AI Crop & Irrigation — Wokwi Demo  "));
  Serial.println(F("  KMITL | George Obinna Oguejiofor         "));
  Serial.println(F("============================================"));
  Serial.println(F("  Left pot  : soil moisture  (0%=dry, 100%=wet)"));
  Serial.println(F("  Right pot : rainfall       (0mm  → 10mm)"));
  Serial.println(F("  Button    : cycle disease mode"));
  Serial.println(F("============================================\n"));
}

// ── Loop ──────────────────────────────────────────────────────────────────
void loop() {
  // Button: cycle disease mode (debounced 50 ms)
  bool btn = digitalRead(BUTTON_PIN);
  if (btn == LOW && lastBtn == HIGH && millis() - lastDebounce > 50) {
    diseaseMode  = (DiseaseMode)((int(diseaseMode) + 1) % 3);
    lastDebounce = millis();
    modeStartMs  = millis();
  }
  lastBtn = btn;

  // Read sensors
  float soil = readSoilMoisture();
  float rain = readRainfall();
  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();
  if (isnan(temp)) temp = 28.0f;
  if (isnan(hum))  hum  = 65.0f;

  // Decision
  bool pump = shouldIrrigate(soil, rain, diseaseMode);

  // Actuate
  digitalWrite(RELAY_PIN, pump ? HIGH : LOW);
  digitalWrite(LED_PIN,   pump ? HIGH : LOW);

  // ── LCD ──────────────────────────────────────────────────────────────
  // Row 0:  S: 67%  T:29C
  // Row 1:  PUMP ON  REDUCE
  char r0[17], r1[17];
  snprintf(r0, sizeof(r0), "S:%3.0f%% T:%2.0fC H:%2.0f", soil, temp, hum);
  snprintf(r1, sizeof(r1), "%s %s", pump ? "PUMP ON " : "PUMP OFF",
           MODE_TAG[diseaseMode]);
  lcd.setCursor(0, 0); lcd.print(r0);
  lcd.setCursor(0, 1); lcd.print(r1);

  // ── Serial telemetry ─────────────────────────────────────────────────
  Serial.println(F("────────────────────────────────────────────"));
  Serial.printf("  Soil moisture  : %5.1f %%\n",   soil);
  Serial.printf("  Temperature    : %5.1f C\n",    temp);
  Serial.printf("  Humidity       : %5.1f %%\n",   hum);
  Serial.printf("  Rainfall 24h   : %5.1f mm\n",   rain);
  Serial.println(F("  ----------------------------------------"));
  Serial.printf("  Disease mode   : %s\n",          MODE_TAG[diseaseMode]);
  Serial.printf("  Detected as    : %s\n",          DISEASE_NAME[diseaseMode]);
  Serial.printf("  Irrig. trigger : soil < %.0f %%\n", THRESHOLD[diseaseMode]);
  if (soil < THRESH_CRITICAL)
    Serial.println(F("  !! EMERGENCY — critically dry !!"));
  if (rain > 5.0f)
    Serial.println(F("  -- Skipping: heavy rainfall detected --"));
  Serial.println(F("  ----------------------------------------"));
  Serial.printf("  >> PUMP        : %s\n",          pump ? "ON  ✓" : "OFF");
  Serial.println(F("────────────────────────────────────────────\n"));

  delay(1000);
}
