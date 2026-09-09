# train.py
# Run this script ONCE to train the Isolation Forest model on the AI4I dataset.
# It will save two files: model.pkl and scaler.pkl
# After this, inference.py loads those files for live prediction.
#
# HOW TO RUN:
#   python train.py
#
# WHAT IT DOES:
#   1. Loads ai4i2020.csv
#   2. Separates normal rows (Machine failure = 0) for training
#   3. Processes data into feature windows
#   4. Trains Isolation Forest on normal data only
#   5. Evaluates on both normal + fault data
#   6. Saves model.pkl and scaler.pkl

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from preprocessor import process_dataframe, normalise

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DATASET_PATH  = 'ai4i2020.csv'       # must be in same folder as this script
MODEL_PATH    = 'model.pkl'           # where the trained model gets saved
SCALER_PATH   = 'scaler.pkl'          # where the scaler gets saved
THRESHOLD_PATH = 'threshold.json'     # auto-tuned anomaly threshold (see save_model)
WINDOW_SIZE   = 50                    # rows per feature window
STEP          = 25                    # window slide step (50% overlap)
CONTAMINATION = 0.05                  # expected % of anomalies (5%)
RANDOM_STATE  = 42                    # for reproducibility
HOLDOUT_FRAC  = 0.20                  # normal windows reserved for out-of-sample validation


def load_dataset(path: str) -> pd.DataFrame:
    """Load the AI4I 2020 dataset and do basic checks."""
    print(f"\n{'='*55}")
    print("  ENGINE MONITOR — MODEL TRAINING")
    print(f"{'='*55}\n")
    
    if not os.path.exists(path):
        print(f"❌ ERROR: Cannot find '{path}'")
        print("   Make sure ai4i2020.csv is in the same folder as train.py")
        exit(1)
    
    df = pd.read_csv(path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")
    
    # Show class balance
    if 'Machine failure' in df.columns:
        normal_count = (df['Machine failure'] == 0).sum()
        fault_count  = (df['Machine failure'] == 1).sum()
        print(f"  Normal rows (failure=0) : {normal_count}")
        print(f"  Fault rows  (failure=1) : {fault_count}")
        print(f"  Fault rate             : {fault_count/len(df)*100:.1f}%\n")
    
    return df


def prepare_training_data(df: pd.DataFrame):
    """
    Split into normal and fault sets.
    We train ONLY on normal data — that is the whole point of unsupervised
    anomaly detection. The model learns what normal looks like, and anything
    that does not match normal gets flagged.
    """
    print("── Preparing training data ──────────────────────────")
    
    # Separate normal and fault rows
    df_normal = df[df['Machine failure'] == 0].copy().reset_index(drop=True)
    df_fault  = df[df['Machine failure'] == 1].copy().reset_index(drop=True)
    
    print(f"  Using {len(df_normal)} normal rows for training")
    print(f"  Using {len(df_fault)} fault rows for evaluation only\n")
    
    # Process normal data into feature windows
    print("  Extracting features from normal data windows...")
    X_normal = process_dataframe(df_normal, window_size=WINDOW_SIZE, step=STEP)
    print(f"  → {len(X_normal)} feature windows created")
    print(f"  → {len(X_normal.columns)} features per window\n")
    
    # Process fault data into feature windows (for evaluation)
    print("  Extracting features from fault data windows...")
    X_fault = process_dataframe(df_fault, window_size=WINDOW_SIZE, step=STEP)
    print(f"  → {len(X_fault)} fault windows created\n")
    
    return X_normal, X_fault


def train_model(X_normal: pd.DataFrame, X_fault: pd.DataFrame):
    """Train the Isolation Forest and evaluate it."""
    
    print("── Normalising features ─────────────────────────────")
    X_normal_scaled, scaler = normalise(X_normal)
    print(f"  ✓ Normal data scaled: {X_normal_scaled.shape}\n")
    
    # Scale fault data using the SAME scaler (no refit)
    if len(X_fault) > 0:
        X_fault_scaled, _ = normalise(X_fault, scaler=scaler)
    
    print("── Training Isolation Forest ────────────────────────")
    print(f"  n_estimators  : 200")
    print(f"  contamination : {CONTAMINATION} ({CONTAMINATION*100:.0f}%)")
    print(f"  Training on   : {len(X_normal_scaled)} windows of normal data")
    print()
    
    model = IsolationForest(
        n_estimators=200,       # number of trees — more = more accurate but slower
        contamination=CONTAMINATION,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1               # use all CPU cores
    )
    
    model.fit(X_normal_scaled)
    print("  ✓ Model trained successfully!\n")
    
    return model, scaler, X_normal_scaled, X_fault_scaled if len(X_fault) > 0 else None


def validate_holdout(X_normal: pd.DataFrame, X_fault: pd.DataFrame):
    """
    Out-of-sample validation, reported BEFORE the final model is fitted.

    WHY THIS EXISTS
    ---------------
    Fitting on every normal window and then measuring the false-positive rate on
    those same windows gives an IN-SAMPLE number: it says how well the forest
    memorised its own training set, not how it behaves on normal operation it has
    never seen. This function reserves HOLDOUT_FRAC of the normal windows, fits a
    throwaway model on the rest, and reports the false-positive rate on the
    reserved part — a genuine generalisation estimate.

    WHY THE SPLIT IS CONTIGUOUS, NOT RANDOM
    ---------------------------------------
    Windows overlap by WINDOW_SIZE - STEP raw rows (25 of 50 here). A random
    split would place two windows sharing 25 identical raw rows on opposite sides
    of the split, leaking training data into the holdout and flattering the
    result. So we split contiguously — first (1 - HOLDOUT_FRAC) of the windows
    train, the tail holds out — and discard the windows straddling the seam, so
    no holdout window shares a single raw row with a training window.

    The model/scaler fitted here are deliberately DISCARDED. The artefacts saved
    to disk are refitted on all normal windows afterwards (standard practice:
    validate on a holdout, then refit on everything for deployment), so
    model.pkl, scaler.pkl and threshold.json stay exactly as documented.
    """
    print("── Out-of-sample validation (holdout) ───────────────")

    n = len(X_normal)
    # windows sharing raw rows across the seam: ceil(WINDOW_SIZE / STEP) - 1
    overlap_guard = -(-WINDOW_SIZE // STEP) - 1
    n_train = int(n * (1 - HOLDOUT_FRAC))

    if n_train < 20 or n - n_train - overlap_guard < 10:
        print(f"  ⚠ Only {n} normal windows — too few to hold out meaningfully.")
        print("    Skipping out-of-sample validation.\n")
        return None

    X_tr = X_normal.iloc[:n_train]
    X_ho = X_normal.iloc[n_train + overlap_guard:]

    print(f"  Split          : contiguous, {int((1-HOLDOUT_FRAC)*100)}/{int(HOLDOUT_FRAC*100)}")
    print(f"  Train windows  : {len(X_tr)}")
    print(f"  Holdout windows: {len(X_ho)}  (never seen during fitting)")
    print(f"  Discarded seam : {overlap_guard} window(s), to prevent overlap leakage\n")

    # Scaler is fitted on the training split ONLY — fitting it on all normal
    # data first would leak the holdout's min/max into the training scale.
    X_tr_scaled, scaler_tr = normalise(X_tr)
    X_ho_scaled, _         = normalise(X_ho, scaler=scaler_tr)

    model_tr = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_tr.fit(X_tr_scaled)

    scores_tr = model_tr.score_samples(X_tr_scaled)
    scores_ho = model_tr.score_samples(X_ho_scaled)

    # Threshold calibrated on the TRAINING split only, then applied unchanged
    # to the holdout — the same derivation compute_threshold() uses.
    thr_tr = float(np.percentile(scores_tr, CONTAMINATION * 100))

    fpr_in  = np.mean(scores_tr < thr_tr) * 100
    fpr_out = np.mean(scores_ho < thr_tr) * 100

    print(f"  Threshold (from train split) : {thr_tr:.4f}")
    print(f"  FPR in-sample  (train)       : {fpr_in:.1f}%")
    print(f"  FPR OUT-OF-SAMPLE (holdout)  : {fpr_out:.1f}%   ← quote this one")

    result = {
        'n_train': len(X_tr), 'n_holdout': len(X_ho),
        'threshold': thr_tr, 'fpr_in_sample': fpr_in, 'fpr_out_of_sample': fpr_out,
        'recall_holdout_model': None,
    }

    # Fault recall from this holdout-trained model, for an honest pairing with
    # the out-of-sample FPR above (faults are never trained on either way).
    if len(X_fault) > 0:
        X_fault_scaled, _ = normalise(X_fault, scaler=scaler_tr)
        scores_fault = model_tr.score_samples(X_fault_scaled)
        recall = np.mean(scores_fault < thr_tr) * 100
        result['recall_holdout_model'] = recall
        print(f"  Fault recall (same model)    : {recall:.1f}%  on {len(X_fault)} fault windows")

    print()
    return result


def evaluate_model(model, X_normal_scaled, X_fault_scaled):
    """
    Test the model on both normal and fault data.
    Isolation Forest returns:
      +1 = normal (inlier)
      -1 = anomaly (outlier)
    We flip this so 0 = normal, 1 = fault for easier reading.
    """
    print("── Evaluating model performance ─────────────────────")
    
    # Predict on normal data
    preds_normal = model.predict(X_normal_scaled)
    scores_normal = model.score_samples(X_normal_scaled)  # anomaly scores
    
    # Count false positives (normal flagged as fault)
    false_positives = np.sum(preds_normal == -1)
    fpr = false_positives / len(preds_normal) * 100
    print(f"  Normal data  : {len(preds_normal)} windows tested")
    print(f"  False positives (normal flagged as fault): {false_positives} ({fpr:.1f}%)")
    
    if X_fault_scaled is not None and len(X_fault_scaled) > 0:
        preds_fault = model.predict(X_fault_scaled)
        scores_fault = model.score_samples(X_fault_scaled)
        
        true_positives = np.sum(preds_fault == -1)
        tpr = true_positives / len(preds_fault) * 100
        print(f"\n  Fault data   : {len(preds_fault)} windows tested")
        print(f"  True positives (faults correctly caught): {true_positives} ({tpr:.1f}%)")
        
        # Full classification report
        y_true = [0] * len(preds_normal) + [1] * len(preds_fault)
        # Convert IF output: -1 → 1 (fault), +1 → 0 (normal)
        y_pred = [1 if p == -1 else 0 for p in list(preds_normal) + list(preds_fault)]
        
        print(f"\n── Classification Report ────────────────────────────")
        print(classification_report(y_true, y_pred,
                                     target_names=['Normal', 'Fault'],
                                     zero_division=0))
        
        # Anomaly score stats
        print(f"── Anomaly Score Summary ────────────────────────────")
        print(f"  Normal data  — mean score: {np.mean(scores_normal):.4f}")
        print(f"  Fault data   — mean score: {np.mean(scores_fault):.4f}")
        print(f"  (Lower score = more anomalous)\n")
        
        return scores_normal, scores_fault
    
    return scores_normal, None


def compute_threshold(scores_normal):
    """
    Derive the anomaly-score cutoff from CONTAMINATION instead of hand-picking
    a number. score_samples() returns a continuous score per window (lower =
    more anomalous); the CONTAMINATION-th percentile of the NORMAL training
    scores is, by definition, the cutoff below which ~CONTAMINATION fraction
    of normal data itself falls — i.e. it reproduces the same "expected ~5%
    of even normal operation looks borderline" assumption already baked into
    the model's own contamination parameter, rather than a threshold chosen
    by eyeballing the score distribution plot.
    """
    return float(np.percentile(scores_normal, CONTAMINATION * 100))


def plot_results(scores_normal, scores_fault, threshold=-0.50):
    """Plot anomaly score distributions so you can visualise the model."""
    print("── Generating score distribution plot ───────────────")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')
    
    # Plot score distributions
    ax.hist(scores_normal, bins=40, alpha=0.7, color='#39ff7a',
            label=f'Normal (n={len(scores_normal)})', density=True)
    
    if scores_fault is not None:
        ax.hist(scores_fault, bins=40, alpha=0.7, color='#ff3f3f',
                label=f'Fault (n={len(scores_fault)})', density=True)
    
    # Draw threshold line
    ax.axvline(x=threshold, color='#e8ff47', linewidth=2,
               linestyle='--', label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Anomaly Score', color='#888888')
    ax.set_ylabel('Density', color='#888888')
    ax.set_title('Isolation Forest — Anomaly Score Distribution',
                 color='#f0f0f0', fontsize=13, pad=15)
    ax.legend(facecolor='#181818', labelcolor='#f0f0f0')
    ax.tick_params(colors='#666666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
    
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.show()
    print("  ✓ Plot saved as score_distribution.png\n")


def save_model(model, scaler, threshold):
    """Save the trained model, scaler, and auto-tuned threshold to disk."""
    print("── Saving model files ───────────────────────────────")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': threshold, 'contamination': CONTAMINATION}, f, indent=2)
    print(f"  ✓ Model     saved → {MODEL_PATH}")
    print(f"  ✓ Scaler    saved → {SCALER_PATH}")
    print(f"  ✓ Threshold saved → {THRESHOLD_PATH} "
          f"({threshold:.4f}, auto-tuned from {CONTAMINATION*100:.0f}% contamination)")
    print()


def print_summary(holdout=None):
    print("=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print()
    if holdout:
        print("  Generalisation (quote these, not the in-sample figures):")
        print(f"    Holdout windows        : {holdout['n_holdout']} "
              f"(trained on {holdout['n_train']})")
        print(f"    FPR out-of-sample      : {holdout['fpr_out_of_sample']:.1f}%")
        print(f"    FPR in-sample          : {holdout['fpr_in_sample']:.1f}%")
        if holdout['recall_holdout_model'] is not None:
            print(f"    Fault recall (holdout model): "
                  f"{holdout['recall_holdout_model']:.1f}%")
        print()
    print("  Files created:")
    print("    model.pkl  — the trained Isolation Forest model")
    print("    scaler.pkl — the feature scaler")
    print("    score_distribution.png — visualisation of results")
    print()
    print("  Next step:")
    print("    Run  →  python inference.py")
    print("    Then →  streamlit run dashboard.py")
    print("=" * 55)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Step 1 — Load dataset
    df = load_dataset(DATASET_PATH)

    # Step 2 — Prepare training and evaluation data
    X_normal, X_fault = prepare_training_data(df)

    # Step 3 — Out-of-sample validation on a held-out slice of normal data.
    #          Runs BEFORE the final fit and throws its own model away; this is
    #          the number to quote for generalisation, because everything from
    #          Step 4 onward is measured in-sample by construction.
    holdout = validate_holdout(X_normal, X_fault)

    # Step 4 — Train the final model on ALL normal windows (deployment refit)
    model, scaler, X_normal_scaled, X_fault_scaled = train_model(X_normal, X_fault)

    # Step 5 — Evaluate (in-sample for the normal class; faults are unseen)
    scores_normal, scores_fault = evaluate_model(model, X_normal_scaled, X_fault_scaled)

    # Step 6 — Auto-tune the anomaly threshold from the contamination rate
    threshold = compute_threshold(scores_normal)

    # Step 7 — Plot results
    plot_results(scores_normal, scores_fault, threshold=threshold)

    # Step 8 — Save model, scaler, and threshold
    save_model(model, scaler, threshold)

    # Step 9 — Summary
    print_summary(holdout)