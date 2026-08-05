# train_bluetooth.py
# Trains a SECOND, BLE-specific Isolation Forest model on real accelerometer/
# temperature data collected from the ESP32 — separate from model.pkl/scaler.pkl,
# which are trained on the AI4I 2020 dataset and used for simulation mode.
#
# WHY TWO MODELS:
#   model.pkl / scaler.pkl        → trained on AI4I columns (Air temp, Torque,
#                                    Rotational speed, ...) — used by
#                                    run_simulation() in inference.py.
#   model_ble.pkl / scaler_ble.pkl → trained on real accX/accY/accZ/temp
#                                    readings — used by run_bluetooth().
#   These feature spaces share no column names and are on completely
#   different physical scales, so one scaler/model cannot serve both.
#
# HOW TO RUN:
#   1. Collect a normal-running baseline first:
#        python collect_ble_baseline.py --minutes 10
#   2. Then:
#        python train_bluetooth.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from preprocessor import process_dataframe, normalise

# ── SETTINGS ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, 'ble_baseline_normal.csv')
MODEL_PATH     = os.path.join(BASE_DIR, 'model_ble.pkl')
SCALER_PATH    = os.path.join(BASE_DIR, 'scaler_ble.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'threshold_ble.json')
FEATURE_COLS  = ['accX', 'accY', 'accZ', 'temp']
WINDOW_SIZE   = 50       # must match WINDOW_SIZE in inference.py
STEP          = 25       # 50% overlap
CONTAMINATION = 0.05
RANDOM_STATE  = 42
MIN_WINDOWS   = 30        # below this, the baseline is too short to trust


def load_baseline(path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  ENGINE MONITOR — BLE MODEL TRAINING")
    print(f"{'='*55}\n")

    if not os.path.exists(path):
        print(f"❌ ERROR: Cannot find '{path}'")
        print("   Run  →  python collect_ble_baseline.py  first!")
        exit(1)

    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ ERROR: baseline CSV is missing expected columns: {missing}")
        print(f"   Found columns: {list(df.columns)}")
        exit(1)

    print(f"✓ Baseline loaded: {df.shape[0]} raw readings")
    approx_minutes = df.shape[0] / 50 / 60   # ~50Hz sampling
    print(f"  ≈ {approx_minutes:.1f} minutes of data at 50Hz\n")
    return df[FEATURE_COLS]


def train_and_evaluate(df: pd.DataFrame):
    print("── Extracting feature windows ───────────────────────")
    X = process_dataframe(df, window_size=WINDOW_SIZE, step=STEP, fs=50.0)
    print(f"  → {len(X)} feature windows created")
    print(f"  → {len(X.columns)} features per window\n")

    if len(X) < MIN_WINDOWS:
        print(f"⚠ WARNING: only {len(X)} windows — that's a very short baseline.")
        print(f"  Recommend collecting at least {MIN_WINDOWS} windows "
              f"(~{(MIN_WINDOWS-1)*STEP + WINDOW_SIZE} readings, "
              f"≈{((MIN_WINDOWS-1)*STEP + WINDOW_SIZE)/50:.0f}s at 50Hz).")
        print("  Training will proceed, but the model may not generalise well.\n")

    print("── Normalising features ─────────────────────────────")
    X_scaled, scaler = normalise(X)
    print(f"  ✓ Scaled: {X_scaled.shape}\n")

    print("── Training Isolation Forest (BLE model) ────────────")
    print(f"  n_estimators  : 200")
    print(f"  contamination : {CONTAMINATION} ({CONTAMINATION*100:.0f}%)")
    print(f"  Training on   : {len(X_scaled)} windows of normal BLE data\n")

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled)
    print("  ✓ Model trained successfully!\n")

    scores = model.score_samples(X_scaled)
    preds  = model.predict(X_scaled)
    fp_rate = np.mean(preds == -1) * 100
    print("── Self-evaluation on training baseline ─────────────")
    print(f"  Mean score          : {np.mean(scores):.4f}")
    print(f"  Score range         : [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"  Flagged as anomaly  : {fp_rate:.1f}%  "
          f"(should be close to the {CONTAMINATION*100:.0f}% contamination setting)\n")

    return model, scaler, scores


def plot_scores(scores):
    print("── Generating score distribution plot ───────────────")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')
    ax.hist(scores, bins=30, alpha=0.8, color='#7eb8ff',
            label=f'Normal BLE baseline (n={len(scores)})', density=True)
    ax.set_xlabel('Anomaly Score', color='#888888')
    ax.set_ylabel('Density', color='#888888')
    ax.set_title('BLE Model — Anomaly Score Distribution (Normal Baseline)',
                 color='#f0f0f0', fontsize=13, pad=15)
    ax.legend(facecolor='#181818', labelcolor='#f0f0f0')
    ax.tick_params(colors='#666666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'score_distribution_ble.png'),
                dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    print("  ✓ Plot saved as score_distribution_ble.png\n")


def save_model(model, scaler, threshold):
    print("── Saving BLE model files ───────────────────────────")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': threshold, 'contamination': CONTAMINATION}, f, indent=2)
    print(f"  ✓ Model     saved → {MODEL_PATH}")
    print(f"  ✓ Scaler    saved → {SCALER_PATH}")
    print(f"  ✓ Threshold saved → {THRESHOLD_PATH} "
          f"({threshold:.4f}, auto-tuned from {CONTAMINATION*100:.0f}% contamination)\n")


if __name__ == '__main__':
    df = load_baseline(DATASET_PATH)
    model, scaler, scores = train_and_evaluate(df)
    # Same derivation as train.py's compute_threshold(): the CONTAMINATION-th
    # percentile of the normal baseline's own scores, so this model gets its
    # own correctly-calibrated cutoff instead of inheriting the AI4I model's.
    threshold = float(np.percentile(scores, CONTAMINATION * 100))
    plot_scores(scores)
    save_model(model, scaler, threshold)

    print("=" * 55)
    print("  BLE TRAINING COMPLETE")
    print("=" * 55)
    print("  model_ble.pkl and scaler_ble.pkl are ready.")
    print("  Run  →  python inference.py --live")
    print("=" * 55)