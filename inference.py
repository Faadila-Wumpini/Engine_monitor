# inference.py
# Combined inference engine — works in TWO modes automatically:
#
#   MODE 1 — SIMULATION (no hardware needed)
#     Replays the AI4I 2020 dataset as if it were live sensor data.
#     Runs automatically when no ESP32 is found or when you use --simulate flag.
#
#   MODE 2 — LIVE BLUETOOTH (with ESP32)
#     Connects to the ESP32 via Bluetooth and scores real sensor data.
#     Runs automatically when ESP32 is found nearby.
#
# HOW TO RUN:
#   python inference.py              ← auto-detects (tries BLE first, falls back to simulation)
#   python inference.py --simulate   ← force simulation mode
#   python inference.py --live       ← force live Bluetooth mode
#
# REQUIREMENTS:
#   pip install bleak supabase joblib scikit-learn pandas numpy scipy

import argparse
import asyncio
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from supabase import create_client
from preprocessor import extract_features

# ── SETTINGS ──────────────────────────────────────────────────────────────────
# Anchor every file to this script's own folder — NOT to whatever directory the
# terminal happens to be in. If inference.py and dashboard.py are launched from
# different terminals/cwds, relative paths can make them read/write two
# different inference_results.csv files without any error being raised.
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH     = os.path.join(BASE_DIR, 'scaler.pkl')
DATASET_PATH    = os.path.join(BASE_DIR, 'ai4i2020.csv')
RESULTS_PATH    = os.path.join(BASE_DIR, 'inference_results.csv')
# Separate model trained on real accX/accY/accZ/temp readings (see
# train_bluetooth.py) — the AI4I-trained model/scaler above have never seen
# these feature names and cannot meaningfully score them.
BLE_MODEL_PATH  = os.path.join(BASE_DIR, 'model_ble.pkl')
BLE_SCALER_PATH = os.path.join(BASE_DIR, 'scaler_ble.pkl')
# Auto-tuned by train.py/train_bluetooth.py from CONTAMINATION (the
# percentile of normal-data scores that matches the model's own expected
# anomaly rate) — see compute_threshold() in train.py. Each model gets its
# own file since the AI4I and BLE models score on different scales; a
# single hardcoded constant shared between them would be a coincidence,
# not a calibration. DEFAULT_THRESHOLD is only a fallback for a model
# trained before this existed.
THRESHOLD_PATH     = os.path.join(BASE_DIR, 'threshold.json')
BLE_THRESHOLD_PATH = os.path.join(BASE_DIR, 'threshold_ble.json')
DEFAULT_THRESHOLD  = -0.50
WINDOW_SIZE     = 50
SIMULATE_DELAY  = 0.1       # seconds between simulated readings

# Bluetooth settings — must match sensor_node_bluetooth.ino
# Every unit advertises as "EngineIQ_Sensor_<DEVICE_ID>" (see the firmware's
# DEVICE_ID constant) — this is the shared prefix, not a full name, so
# scanning finds any of them; --device targets one specifically when more
# than one is nearby.
BLE_DEVICE_PREFIX = "EngineIQ_Sensor"
SERVICE_UUID    = "12345678-1234-1234-1234-123456789abc"
CHAR_UUID       = "abcd1234-ab12-ab12-ab12-abcdef123456"
BLE_SCAN_TIMEOUT = 8.0      # seconds to scan before giving up

# Supabase settings
SUPABASE_URL = "https://vpudvhanmyggzimwcgqo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZwdWR2aGFubXlnZ3ppbXdjZ3FvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA1MTA2MTQsImV4cCI6MjA5NjA4NjYxNH0.iXIR8-UE5SIaZfAY-Uc6-LaUzYkqfI4LdSY_71V9WQs"

# Terminal colours
COLOURS = {
    'NORMAL': '\033[92m',
    'LOW':    '\033[93m',
    'MEDIUM': '\033[33m',
    'HIGH':   '\033[91m',
}
RESET = '\033[0m'


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Model loaded  ← {MODEL_PATH}")
        print(f"✓ Scaler loaded ← {SCALER_PATH}\n")
        return model, scaler
    except FileNotFoundError:
        print("❌ model.pkl or scaler.pkl not found.")
        print("   Run  →  python train.py  first!\n")
        exit(1)


def load_threshold(path, model_label):
    try:
        with open(path) as f:
            value = float(json.load(f)['threshold'])
        print(f"✓ Threshold loaded ← {path} ({value:.4f})\n")
        return value
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        print(f"⚠ {path} not found or unreadable — using fallback "
              f"threshold {DEFAULT_THRESHOLD} for the {model_label} model.")
        print(f"  Retrain to auto-tune this from the contamination rate.\n")
        return DEFAULT_THRESHOLD


def load_ble_model():
    """
    Load the BLE-specific model (trained on real accX/accY/accZ/temp data via
    train_bluetooth.py). Returns (None, None) if it hasn't been trained yet —
    callers must handle that explicitly rather than silently falling back to
    the AI4I model, which does not understand these feature names.
    """
    try:
        model  = joblib.load(BLE_MODEL_PATH)
        scaler = joblib.load(BLE_SCALER_PATH)
        print(f"✓ BLE model loaded  ← {BLE_MODEL_PATH}")
        print(f"✓ BLE scaler loaded ← {BLE_SCALER_PATH}\n")
        return model, scaler
    except FileNotFoundError:
        return None, None


# ── CONNECT SUPABASE ──────────────────────────────────────────────────────────
def connect_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase connected!\n")
        return client
    except Exception as e:
        print(f"⚠ Supabase connection failed: {e}")
        print("  Continuing without database logging...\n")
        return None


# ── SEVERITY ──────────────────────────────────────────────────────────────────
def get_severity(score, threshold):
    if score > threshold:        return 'NORMAL'
    if score > threshold - 0.02: return 'LOW'
    if score > threshold - 0.05: return 'MEDIUM'
    return 'HIGH'


# ── LOG TO SUPABASE ───────────────────────────────────────────────────────────
# Logs EVERY scored window, not just anomalies — the full engine-health
# history needs to include what "normal" looks like, not only the spikes.
# 'source' distinguishes AI4I-replay windows from real ESP32 windows, since
# they carry different physical readings (acc_x/y/z/temp columns only make
# sense for 'live'; 'raw_window' carries the mean of every column for either
# mode so nothing is mislabeled). 'device_id' (live only — see the firmware's
# DEVICE_ID constant) is what makes readings from multiple ESP32 units
# distinguishable in this table instead of an undifferentiated pile. Requires
# the 'source', 'raw_window', and 'device_id' columns to exist on
# anomaly_logs — see migration note in README/commit message if they're
# missing; insert failures are caught and logged, not fatal, so older
# schemas just lose history logging instead of crashing.
def log_to_supabase(client, result, window_df, source, device_id=None):
    if client is None:
        return
    try:
        record = {
            "timestamp"    : result['timestamp'],
            "source"       : source,          # 'simulation' or 'live'
            "device_id"    : device_id,       # which physical ESP32, if 'live'
            "anomaly_score": result['score'],
            "severity"     : result['severity'],
            "is_anomaly"   : result['is_anomaly'],
            "is_confirmed" : False,
            "raw_window"   : {col: float(window_df[col].mean()) for col in window_df.columns},
        }
        if source == 'live':
            record.update({
                "acc_x_mean": float(window_df['accX'].mean()),
                "acc_y_mean": float(window_df['accY'].mean()),
                "acc_z_mean": float(window_df['accZ'].mean()),
                "temp_mean" : float(window_df['temp'].mean()),
            })
        client.table("anomaly_logs").insert(record).execute()
        print(f"  💾 Saved to Supabase — severity: {result['severity']}")
    except Exception as e:
        print(f"  ⚠ Supabase log failed: {e}")


# ── SCORE A WINDOW ────────────────────────────────────────────────────────────
def score_window(model, scaler, window_df, threshold, fs=1.0):
    features   = extract_features(window_df, fs=fs)
    feature_df = pd.DataFrame([features]).fillna(0)

    expected = list(scaler.feature_names_in_)
    present  = [c for c in expected if c in feature_df.columns]
    missing_ratio = 1 - (len(present) / len(expected))

    if missing_ratio > 0.5:
        # The scaler expects feature names that mostly don't exist in this
        # window — this means the wrong model/scaler pair was passed in
        # (e.g. the AI4I-trained model given live BLE features, or vice
        # versa). Previously this silently filled every missing column with
        # 0 and returned a fake score; that masked the bug instead of
        # catching it. Fail loudly instead.
        raise ValueError(
            f"Feature mismatch: scaler expects {len(expected)} features "
            f"(e.g. {expected[:2]}) but this window only produced "
            f"{len(feature_df.columns)} features "
            f"(e.g. {list(feature_df.columns)[:2]}). "
            f"Only {len(present)}/{len(expected)} expected features are present. "
            f"You're likely scoring with the wrong model/scaler pair — "
            f"use the AI4I model for run_simulation() and the BLE model "
            f"(model_ble.pkl/scaler_ble.pkl, from train_bluetooth.py) for "
            f"run_bluetooth()."
        )

    # Minor mismatches (e.g. a feature that's NaN/absent on one window) are
    # still tolerated by filling with 0, since that's a normal data gap —
    # not a sign of a wrong model.
    for col in expected:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[expected]
    scaled = scaler.transform(feature_df)

    score    = model.score_samples(scaled)[0]
    severity = get_severity(score, threshold)
    return {
        'score'     : round(float(score), 4),
        'severity'  : severity,
        # bool(), not the bare numpy.bool_ that `score < threshold` produces —
        # numpy.bool_ isn't JSON-serializable, which silently broke every
        # Supabase insert once is_anomaly was added to the logged record.
        'is_anomaly': bool(score < threshold),
        'timestamp' : datetime.now().isoformat()
    }


# ── SAVE RESULTS TO CSV ───────────────────────────────────────────────────────
def save_result(result):
    """
    Append a result row to inference_results.csv for the dashboard to read.
    Appends directly instead of reading + rewriting the whole file each call —
    the old approach re-read and rewrote the entire (ever-growing) CSV on every
    single window, which gets slower and slower as the file grows (it was
    already ~19,900 rows) and briefly truncates the file on every write, which
    a dashboard reading concurrently could catch mid-write.
    """
    new_row     = pd.DataFrame([result])
    file_exists = os.path.exists(RESULTS_PATH)
    new_row.to_csv(RESULTS_PATH, mode='a', index=False, header=not file_exists)


# ── PRINT RESULT TO TERMINAL ──────────────────────────────────────────────────
def print_result(result, extra=''):
    colour = COLOURS.get(result['severity'], RESET)
    ts     = result['timestamp'][11:19]
    icon   = '🔴 ANOMALY!' if result['is_anomaly'] else '✓'
    print(f"  {ts:<12} {result['score']:>8.4f}  "
          f"{colour}{result['severity']:<8}{RESET}  {icon} {extra}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def run_simulation(model, scaler, supabase):
    print("=" * 55)
    print("  MODE: SIMULATION")
    print("  Replaying AI4I dataset as live engine data.")
    print("  (No ESP32 hardware needed)")
    print("=" * 55)
    print(f"\n  {'Timestamp':<12} {'Score':>8}  {'Severity':<8}  Status")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*20}")

    threshold = load_threshold(THRESHOLD_PATH, 'AI4I')

    # Check dataset exists
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"\n❌ Dataset not found: {DATASET_PATH}")
        print("   Download ai4i2020.csv and place it in this folder.")
        return

    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    buffer        = []
    anomaly_count = 0
    window_count  = 0
    results       = []

    try:
        for idx, row in df.iterrows():
            reading = {col: row[col] for col in feature_cols if col in row.index}
            reading['actual_failure'] = int(row.get('Machine failure', 0))
            buffer.append(reading)

            if len(buffer) < WINDOW_SIZE:
                continue

            buffer    = buffer[-WINDOW_SIZE:]
            window_df = pd.DataFrame(buffer)[feature_cols]
            result    = score_window(model, scaler, window_df, threshold)
            result['actual_failure'] = buffer[-1]['actual_failure']
            results.append(result)
            window_count += 1

            actual = '⚠ ACTUAL FAULT' if result['actual_failure'] == 1 else ''
            print_result(result, extra=actual)

            if result['is_anomaly']:
                anomaly_count += 1

            log_to_supabase(supabase, result, window_df, source='simulation')
            save_result(result)
            time.sleep(SIMULATE_DELAY)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user.")

    finally:
        print(f"\n{'='*55}")
        print(f"  SIMULATION SUMMARY")
        print(f"{'='*55}")
        print(f"  Windows scored   : {window_count}")
        print(f"  Anomalies flagged: {anomaly_count} "
              f"({anomaly_count/max(window_count,1)*100:.1f}%)")
        print(f"{'='*55}")


async def scan_for_sensor(device_id=None):
    """
    Scan for an ESP32 advertising as "EngineIQ_Sensor_<DEVICE_ID>" (see the
    firmware's DEVICE_ID constant — change it before flashing a second unit).
    Matches by prefix, not exact name, so multiple units can coexist and be
    told apart; pass device_id to target one specifically instead of
    connecting to whichever matching device the scan happens to find first.
    """
    from bleak import BleakScanner

    def matches(device, advertisement_data):
        name = device.name or ''
        if not name.startswith(BLE_DEVICE_PREFIX):
            return False
        if device_id is None:
            return True
        return name == f"{BLE_DEVICE_PREFIX}_{device_id}"

    return await BleakScanner.find_device_by_filter(matches, timeout=BLE_SCAN_TIMEOUT)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — LIVE BLUETOOTH
# ══════════════════════════════════════════════════════════════════════════════
async def run_bluetooth(ai4i_model, ai4i_scaler, supabase, device_id=None):
    """Scan for ESP32 via BLE and score live sensor data using the
    BLE-specific model (trained on real accX/accY/accZ/temp data via
    train_bluetooth.py) — NOT the AI4I model, which has never seen this
    feature space and cannot meaningfully score it. device_id, if given,
    targets one specific unit (see scan_for_sensor) when more than one
    EngineIQ sensor might be nearby."""
    try:
        from bleak import BleakClient
    except ImportError:
        print("❌ bleak not installed. Run:  pip install bleak")
        print("   Falling back to simulation mode...\n")
        run_simulation(ai4i_model, ai4i_scaler, supabase)
        return

    ble_model, ble_scaler = load_ble_model()
    if ble_model is None:
        print("⚠ model_ble.pkl / scaler_ble.pkl not found.")
        print("  The AI4I model cannot score live BLE sensor data — the feature")
        print("  spaces don't match (see preprocessor.py / dissertation notes).")
        print("  Run:")
        print("    1. python collect_ble_baseline.py --minutes 10   (engine running normally)")
        print("    2. python train_bluetooth.py")
        print("  Falling back to SIMULATION mode for now...\n")
        run_simulation(ai4i_model, ai4i_scaler, supabase)
        return

    ble_threshold = load_threshold(BLE_THRESHOLD_PATH, 'BLE')

    buffer       = []
    feature_cols = ['accX', 'accY', 'accZ', 'temp']

    print("=" * 55)
    print("  MODE: LIVE BLUETOOTH")
    print(f"  Scanning for '{BLE_DEVICE_PREFIX}_{device_id}'..." if device_id
          else f"  Scanning for any '{BLE_DEVICE_PREFIX}_*' sensor...")
    print("=" * 55)

    device = await scan_for_sensor(device_id)

    if device is None:
        print(f"\n⚠ Could not find a matching EngineIQ sensor nearby.")
        print("  Possible reasons:")
        print("  → ESP32 is not powered on")
        print("  → Bluetooth is off on your laptop")
        print("  → Wrong firmware uploaded to ESP32")
        if device_id:
            print(f"  → No unit advertising as '{BLE_DEVICE_PREFIX}_{device_id}' specifically "
                  f"(check its DEVICE_ID, or drop --device to accept any unit)")
        print("\n  Falling back to SIMULATION mode...\n")
        run_simulation(ai4i_model, ai4i_scaler, supabase)
        return

    print(f"\n  ✓ Found: {device.name} [{device.address}]")
    print(f"  Connecting...\n")
    print(f"  {'Timestamp':<12} {'Score':>8}  {'Severity':<8}  Status")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*20}")

    async with BleakClient(device) as client:
        print(f"  ✓ Connected via Bluetooth!\n")

        # Falls back to the advertised BLE name (minus the shared prefix) if
        # an individual payload is missing device_id — e.g. older firmware
        # that hasn't been reflashed with the DEVICE_ID field yet.
        fallback_device_id = device.name.replace(f'{BLE_DEVICE_PREFIX}_', '', 1) if device.name else 'unknown'

        def handle_notification(sender, data):
            nonlocal buffer
            try:
                payload = json.loads(data.decode('utf-8'))
                reading = {
                    'accX': float(payload.get('accX', 0)),
                    'accY': float(payload.get('accY', 0)),
                    'accZ': float(payload.get('accZ', 0)),
                    'temp': float(payload.get('temp', 0)),
                }
                connected_device_id = payload.get('device_id', fallback_device_id)
                buffer.append(reading)

                if len(buffer) >= WINDOW_SIZE:
                    buffer    = buffer[-WINDOW_SIZE:]
                    window_df = pd.DataFrame(buffer)[feature_cols]
                    result    = score_window(ble_model, ble_scaler, window_df, ble_threshold, fs=50.0)

                    print_result(result)

                    log_to_supabase(supabase, result, window_df, source='live',
                                     device_id=connected_device_id)
                    save_result(result)

            except Exception as e:
                print(f"  ⚠ BLE data error: {e}")

        await client.start_notify(CHAR_UUID, handle_notification)
        print("  Receiving live data from ESP32... Press Ctrl+C to stop.\n")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            await client.stop_notify(CHAR_UUID)


# ── AUTO-DETECT MODE ──────────────────────────────────────────────────────────
async def auto_detect(model, scaler, supabase, device_id=None):
    """
    Tries to find an EngineIQ ESP32 via Bluetooth (optionally a specific
    device_id — see scan_for_sensor).
    If found → runs live Bluetooth mode.
    If not found → falls back to simulation mode.
    """
    try:
        import bleak  # noqa: F401 — just checking it's installed
    except ImportError:
        print("ℹ bleak not installed — running simulation mode.")
        print("  To enable Bluetooth: pip install bleak\n")
        run_simulation(model, scaler, supabase)
        return

    label = f"'{BLE_DEVICE_PREFIX}_{device_id}'" if device_id else f"any '{BLE_DEVICE_PREFIX}_*' sensor"
    print(f"  Scanning for {label} ({BLE_SCAN_TIMEOUT}s)...")
    device = await scan_for_sensor(device_id)

    if device:
        print(f"  ✓ ESP32 found! Switching to LIVE BLUETOOTH mode.\n")
        await run_bluetooth(model, scaler, supabase, device_id=device_id)
    else:
        print(f"  ℹ ESP32 not found. Switching to SIMULATION mode.\n")
        run_simulation(model, scaler, supabase)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EngineIQ Inference Engine — Simulation or Live Bluetooth'
    )
    parser.add_argument('--simulate', action='store_true',
                        help='Force simulation mode (use dataset, no hardware)')
    parser.add_argument('--live', action='store_true',
                        help='Force live Bluetooth mode (requires ESP32)')
    parser.add_argument('--device', type=str, default=None,
                        help="Target a specific ESP32 by its DEVICE_ID (e.g. --device ESP32-02) "
                             "when more than one EngineIQ sensor might be nearby. "
                             "Omit to connect to whichever matching unit is found first.")
    args = parser.parse_args()

    print("\n╔═══════════════════════════════════════════╗")
    print("║  EngineIQ — Inference Engine              ║")
    print("║  Group 11, KNUST                          ║")
    print("╚═══════════════════════════════════════════╝\n")

    # Load model and connect Supabase
    model, scaler = load_model()
    supabase      = connect_supabase()

    # Choose mode
    if args.simulate:
        # Force simulation
        run_simulation(model, scaler, supabase)

    elif args.live:
        # Force live Bluetooth
        asyncio.run(run_bluetooth(model, scaler, supabase, device_id=args.device))

    else:
        # Auto-detect: try BLE first, fall back to simulation
        print("  Auto-detecting mode...\n")
        asyncio.run(auto_detect(model, scaler, supabase, device_id=args.device))
