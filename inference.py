# inference.py
# Loads the trained model and scores incoming data.
# Works in two modes:
#   MODE 1 — SIMULATE  : replays rows from ai4i2020.csv as if they were live
#   MODE 2 — LIVE      : subscribes to MQTT and scores real ESP32 sensor data
#
# For now use MODE 1 (simulate) since we are still waiting for hardware.
# Switch to MODE 2 once the ESP32 is set up.
#
# HOW TO RUN:
#   python inference.py            ← runs in simulate mode by default
#   python inference.py --live     ← runs in live MQTT mode (needs ESP32)

import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from preprocessor import extract_features, normalise

# ── SETTINGS ──────────────────────────────────────────────────────────────────
MODEL_PATH     = 'model.pkl'
SCALER_PATH    = 'scaler.pkl'
DATASET_PATH   = 'ai4i2020.csv'
THRESHOLD      = -0.520        # scores below this = anomaly
WINDOW_SIZE    = 50             # must match what was used in train.py
SIMULATE_DELAY = 0.5            # seconds between simulated readings (10Hz)

# MQTT settings (only needed for live mode)
MQTT_BROKER    = 'localhost'
MQTT_PORT      = 1883
MQTT_TOPIC     = 'engine/sensors'

# Severity levels based on anomaly score
def get_severity(score: float) -> str:
    if score > THRESHOLD:
        return 'NORMAL'
    elif score > -0.15:
        return 'LOW'
    elif score > -0.25:
        return 'MEDIUM'
    else:
        return 'HIGH'

def get_severity_colour(severity: str) -> str:
    colours = {
        'NORMAL' : '\033[92m',   # green
        'LOW'    : '\033[93m',   # yellow
        'MEDIUM' : '\033[33m',   # orange
        'HIGH'   : '\033[91m',   # red
    }
    return colours.get(severity, '\033[0m')

RESET = '\033[0m'


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    """Load the trained Isolation Forest model and scaler from disk."""
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Model loaded  ← {MODEL_PATH}")
        print(f"✓ Scaler loaded ← {SCALER_PATH}\n")
        return model, scaler
    except FileNotFoundError:
        print("❌ ERROR: model.pkl or scaler.pkl not found.")
        print("   You must run  →  python train.py  first!")
        exit(1)


# ── SCORE A WINDOW ────────────────────────────────────────────────────────────
def score_window(model, scaler, window_df: pd.DataFrame) -> dict:
    """
    Takes a window (DataFrame of WINDOW_SIZE rows), extracts features,
    scales them, and returns the anomaly score and severity.
    """
    # Extract features from the window
    features = extract_features(window_df)
    feature_df = pd.DataFrame([features]).fillna(0)
    
    # Make sure columns match what the scaler expects
    try:
        feature_scaled = scaler.transform(feature_df)
    except Exception:
        # If column mismatch, align columns to scaler's expected features
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df[expected_cols]
        feature_scaled = scaler.transform(feature_df)
    
    # Get anomaly score (lower = more anomalous)
    score = model.score_samples(feature_scaled)[0]
    severity = get_severity(score)
    
    return {
        'score'    : round(float(score), 4),
        'severity' : severity,
        'is_anomaly': score < THRESHOLD,
        'timestamp': datetime.now().isoformat()
    }


# ── MODE 1: SIMULATE ──────────────────────────────────────────────────────────
def run_simulation(model, scaler):
    """
    Replay the AI4I dataset row by row as if it were live sensor data.
    This lets you see the system working without needing the ESP32.
    """
    print("=" * 55)
    print("  MODE: SIMULATION (replaying dataset as live data)")
    print("=" * 55)
    print("  Scoring every window of 50 readings...")
    print("  Press Ctrl+C to stop.\n")
    print(f"  {'Timestamp':<25} {'Score':>8}  {'Severity':<8}  Status")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*20}")
    
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Columns the model uses (same as training — no labels)
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    buffer = []           # rolling buffer of recent readings
    anomaly_count = 0
    window_count  = 0
    
    # Also write results to a CSV for the dashboard to read
    results = []
    
    try:
        for idx, row in df.iterrows():
            # Add row to buffer
            reading = {col: row[col] for col in feature_cols if col in row.index}
            reading['actual_failure'] = int(row.get('Machine failure', 0))
            buffer.append(reading)
            
            # Only score once we have a full window
            if len(buffer) < WINDOW_SIZE:
                continue
            
            # Keep only the last WINDOW_SIZE readings
            buffer = buffer[-WINDOW_SIZE:]
            window_df = pd.DataFrame(buffer)[feature_cols]
            
            # Score this window
            result = score_window(model, scaler, window_df)
            result['actual_failure'] = buffer[-1]['actual_failure']
            results.append(result)
            window_count += 1
            
            # Print result
            colour   = get_severity_colour(result['severity'])
            ts       = result['timestamp'][11:19]  # just the time part
            score    = result['score']
            severity = result['severity']
            actual   = "⚠ ACTUAL FAULT" if result['actual_failure'] == 1 else ""
            
            if result['is_anomaly']:
                anomaly_count += 1
                print(f"  {ts:<25} {score:>8.4f}  {colour}{severity:<8}{RESET}  🔴 ANOMALY DETECTED {actual}")
            else:
                print(f"  {ts:<25} {score:>8.4f}  {colour}{severity:<8}{RESET}  ✓ {actual}")
            
            # Save results periodically
            if window_count % 20 == 0:
                pd.DataFrame(results).to_csv('inference_results.csv', index=False)
            
            time.sleep(SIMULATE_DELAY)
    
    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user.")
    
    finally:
        # Save final results
        if results:
            pd.DataFrame(results).to_csv('inference_results.csv', index=False)
            print(f"\n  Results saved → inference_results.csv")
        
        # Print summary
        print(f"\n{'='*55}")
        print(f"  SIMULATION SUMMARY")
        print(f"{'='*55}")
        print(f"  Windows scored   : {window_count}")
        print(f"  Anomalies flagged: {anomaly_count} ({anomaly_count/max(window_count,1)*100:.1f}%)")
        print(f"{'='*55}")


# ── MODE 2: LIVE MQTT ─────────────────────────────────────────────────────────
def run_live(model, scaler):
    """
    Subscribe to the MQTT broker and score real ESP32 sensor data.
    Requires: mosquitto broker running + ESP32 publishing to engine/sensors
    """
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("❌ paho-mqtt not installed. Run: pip install paho-mqtt")
        exit(1)
    
    print("=" * 55)
    print("  MODE: LIVE (reading from ESP32 via MQTT)")
    print(f"  Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Topic : {MQTT_TOPIC}")
    print("=" * 55)
    
    buffer = []
    
    # Map ESP32 JSON keys to the feature column names the model expects
    # Adjust these based on what your ESP32 actually sends
    col_map = {
        'accX' : 'Rotational speed [rpm]',   # vibration X → speed proxy
        'accY' : 'Torque [Nm]',               # vibration Y → torque proxy
        'accZ' : 'Tool wear [min]',            # vibration Z → wear proxy
        'temp' : 'Air temperature [K]',        # temperature
    }
    
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"\n✓ Connected to MQTT broker")
            client.subscribe(MQTT_TOPIC)
            print(f"✓ Subscribed to topic: {MQTT_TOPIC}")
            print(f"\n  Waiting for ESP32 data...\n")
        else:
            print(f"❌ Connection failed with code {rc}")
    
    def on_message(client, userdata, msg):
        nonlocal buffer
        try:
            payload = json.loads(msg.payload.decode())
            
            # Map ESP32 fields to feature columns
            reading = {}
            for esp_key, col_name in col_map.items():
                reading[col_name] = float(payload.get(esp_key, 0))
            
            # Process temperature is air temp + 10 (as in AI4I dataset)
            reading['Process temperature [K]'] = reading['Air temperature [K]'] + 10
            
            buffer.append(reading)
            
            # Score when buffer is full
            if len(buffer) >= WINDOW_SIZE:
                buffer = buffer[-WINDOW_SIZE:]
                window_df = pd.DataFrame(buffer)[feature_cols]
                result = score_window(model, scaler, window_df)
                
                colour   = get_severity_colour(result['severity'])
                ts       = result['timestamp'][11:19]
                
                if result['is_anomaly']:
                    print(f"  {ts}  Score: {result['score']:>7.4f}  "
                          f"{colour}{result['severity']}{RESET}  🔴 ANOMALY!")
                else:
                    print(f"  {ts}  Score: {result['score']:>7.4f}  "
                          f"{colour}{result['severity']}{RESET}")
        
        except Exception as e:
            print(f"  ⚠ Error processing message: {e}")
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\n❌ Could not connect to MQTT broker: {e}")
        print("   Make sure mosquitto is running: mosquitto -v")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Engine Monitor Inference Engine')
    parser.add_argument('--live', action='store_true',
                        help='Use live MQTT mode instead of simulation')
    args = parser.parse_args()
    
    # Load the model first (always)
    model, scaler = load_model()
    
    # Run in the selected mode
    if args.live:
        run_live(model, scaler)
    else:
        run_simulation(model, scaler)
