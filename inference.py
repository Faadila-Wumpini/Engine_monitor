# inference.py
# Loads the trained model and scores incoming data.
# Now with Supabase integration — anomalies are saved to the cloud database.
#
# HOW TO RUN:
#   python inference.py            ← simulate mode (uses dataset)
#   python inference.py --live     ← live mode (uses ESP32 via MQTT)

import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from supabase import create_client
from preprocessor import extract_features, normalise

# ── SUPABASE SETTINGS ─────────────────────────────────────────────────────────
SUPABASE_URL = "https://vpudvhanmyggzimwcgqo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZwdWR2aGFubXlnZ3ppbXdjZ3FvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA1MTA2MTQsImV4cCI6MjA5NjA4NjYxNH0.iXIR8-UE5SIaZfAY-Uc6-LaUzYkqfI4LdSY_71V9WQs"

# ── OTHER SETTINGS ────────────────────────────────────────────────────────────
MODEL_PATH     = 'model.pkl'
SCALER_PATH    = 'scaler.pkl'
DATASET_PATH   = 'ai4i2020.csv'
THRESHOLD      = -0.520
WINDOW_SIZE    = 50
SIMULATE_DELAY = 0.1

# MQTT settings (live mode only)
MQTT_BROKER = 'localhost'
MQTT_PORT   = 1883
MQTT_TOPIC  = 'engine/sensors'

# ── SEVERITY ──────────────────────────────────────────────────────────────────
def get_severity(score: float) -> str:
    if score > -0.50:    return 'NORMAL'
    elif score > -0.52:  return 'LOW'
    elif score > -0.55:  return 'MEDIUM'
    else:                return 'HIGH'
    
def get_colour(severity: str) -> str:
    return {'NORMAL':'\033[92m','LOW':'\033[93m','MEDIUM':'\033[33m','HIGH':'\033[91m'}.get(severity,'\033[0m')

RESET = '\033[0m'

# ── CONNECT TO SUPABASE ───────────────────────────────────────────────────────
def connect_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase connected!")
        return client
    except Exception as e:
        print(f"⚠ Supabase connection failed: {e}")
        return None

# ── LOG ANOMALY TO SUPABASE ───────────────────────────────────────────────────
def log_to_supabase(client, result: dict, window_df: pd.DataFrame):
    if client is None:
        return
    try:
        record = {
            "timestamp"    : result['timestamp'],
            "acc_x_mean"   : float(window_df.iloc[:,0].mean()) if len(window_df.columns)>0 else 0,
            "acc_y_mean"   : float(window_df.iloc[:,1].mean()) if len(window_df.columns)>1 else 0,
            "acc_z_mean"   : float(window_df.iloc[:,2].mean()) if len(window_df.columns)>2 else 0,
            "temp_mean"    : float(window_df.iloc[:,3].mean()) if len(window_df.columns)>3 else 0,
            "anomaly_score": result['score'],
            "severity"     : result['severity'],
            "is_confirmed" : False
        }
        client.table("anomaly_logs").insert(record).execute()
        print(f"  💾 Saved to Supabase — severity: {result['severity']}")
    except Exception as e:
        print(f"  ⚠ Supabase log failed: {e}")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Model loaded  ← {MODEL_PATH}")
        print(f"✓ Scaler loaded ← {SCALER_PATH}\n")
        return model, scaler
    except FileNotFoundError:
        print("❌ model.pkl or scaler.pkl not found. Run python train.py first!")
        exit(1)

# ── SCORE A WINDOW ────────────────────────────────────────────────────────────
def score_window(model, scaler, window_df: pd.DataFrame) -> dict:
    features   = extract_features(window_df)
    feature_df = pd.DataFrame([features]).fillna(0)
    try:
        feature_scaled = scaler.transform(feature_df)
    except Exception:
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df     = feature_df[expected_cols]
        feature_scaled = scaler.transform(feature_df)
    score    = model.score_samples(feature_scaled)[0]
    severity = get_severity(score)
    return {
        'score'     : round(float(score), 4),
        'severity'  : severity,
        'is_anomaly': score < THRESHOLD,
        'timestamp' : datetime.now().isoformat()
    }

# ── MODE 1: SIMULATE ──────────────────────────────────────────────────────────
def run_simulation(model, scaler, supabase):
    print("="*55)
    print("  MODE: SIMULATION (replaying dataset as live data)")
    print("="*55)
    print(f"  {'Timestamp':<25} {'Score':>8}  {'Severity':<8}  Status")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*20}")

    df = pd.read_csv(DATASET_PATH)
    feature_cols = [
        'Air temperature [K]','Process temperature [K]',
        'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]'
    ]
    buffer,anomaly_count,window_count,results = [],0,0,[]

    try:
        for idx, row in df.iterrows():
            reading = {col: row[col] for col in feature_cols if col in row.index}
            reading['actual_failure'] = int(row.get('Machine failure', 0))
            buffer.append(reading)
            if len(buffer) < WINDOW_SIZE:
                continue
            buffer    = buffer[-WINDOW_SIZE:]
            window_df = pd.DataFrame(buffer)[feature_cols]
            result    = score_window(model, scaler, window_df)
            result['actual_failure'] = buffer[-1]['actual_failure']
            results.append(result)
            window_count += 1
            colour = get_colour(result['severity'])
            ts     = result['timestamp'][11:19]
            actual = "⚠ ACTUAL FAULT" if result['actual_failure']==1 else ""
            if result['is_anomaly']:
                anomaly_count += 1
                print(f"  {ts:<25} {result['score']:>8.4f}  {colour}{result['severity']:<8}{RESET}  🔴 ANOMALY {actual}")
                log_to_supabase(supabase, result, window_df)
            else:
                print(f"  {ts:<25} {result['score']:>8.4f}  {colour}{result['severity']:<8}{RESET}  ✓ {actual}")
            if window_count % 20 == 0:
                pd.DataFrame(results).to_csv('inference_results.csv', index=False)
            time.sleep(SIMULATE_DELAY)
    except KeyboardInterrupt:
        print(f"\n\n  Stopped.")
    finally:
        if results:
            pd.DataFrame(results).to_csv('inference_results.csv', index=False)
        print(f"\n{'='*55}")
        print(f"  Windows scored   : {window_count}")
        print(f"  Anomalies flagged: {anomaly_count} ({anomaly_count/max(window_count,1)*100:.1f}%)")
        print(f"{'='*55}")

# ── MODE 2: LIVE MQTT ─────────────────────────────────────────────────────────
def run_live(model, scaler, supabase):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("❌ paho-mqtt not installed. Run: pip install paho-mqtt")
        exit(1)
    print("="*55)
    print("  MODE: LIVE (reading from ESP32 via MQTT)")
    print(f"  Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print("="*55)
    buffer = []
    col_map = {
        'accX':'Air temperature [K]',
        'accY':'Process temperature [K]',
        'accZ':'Rotational speed [rpm]',
        'temp':'Torque [Nm]',
    }
    feature_cols = [
        'Air temperature [K]','Process temperature [K]',
        'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]'
    ]
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"✓ Connected to MQTT broker")
            client.subscribe(MQTT_TOPIC)
            print(f"✓ Subscribed to: {MQTT_TOPIC}\n")
        else:
            print(f"❌ MQTT failed: {rc}")
    def on_message(client, userdata, msg):
        nonlocal buffer
        try:
            payload = json.loads(msg.payload.decode())
            reading = {col_map[k]: float(payload.get(k,0)) for k in col_map}
            reading['Tool wear [min]'] = 0
            buffer.append(reading)
            if len(buffer) >= WINDOW_SIZE:
                buffer    = buffer[-WINDOW_SIZE:]
                window_df = pd.DataFrame(buffer)[feature_cols]
                result    = score_window(model, scaler, window_df)
                colour    = get_colour(result['severity'])
                ts        = result['timestamp'][11:19]
                if result['is_anomaly']:
                    print(f"  {ts}  Score:{result['score']:>7.4f}  {colour}{result['severity']}{RESET}  🔴 ANOMALY!")
                    log_to_supabase(supabase, result, window_df)
                else:
                    print(f"  {ts}  Score:{result['score']:>7.4f}  {colour}{result['severity']}{RESET}")
                new_row = pd.DataFrame([result])
                try:
                    existing = pd.read_csv('inference_results.csv')
                    updated  = pd.concat([existing, new_row], ignore_index=True)
                except FileNotFoundError:
                    updated = new_row
                updated.to_csv('inference_results.csv', index=False)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"❌ Could not connect to MQTT: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    print("\n╔═══════════════════════════════════════════╗")
    print("║  EngineIQ — Inference Engine              ║")
    print("║  Group 11, KNUST                          ║")
    print("╚═══════════════════════════════════════════╝\n")
    supabase      = connect_supabase()
    model, scaler = load_model()
    if args.live:
        run_live(model, scaler, supabase)
    else:
        run_simulation(model, scaler, supabase)