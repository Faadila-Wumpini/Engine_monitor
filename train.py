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
WINDOW_SIZE   = 50                    # rows per feature window
STEP          = 25                    # window slide step (50% overlap)
CONTAMINATION = 0.05                  # expected % of anomalies (5%)
RANDOM_STATE  = 42                    # for reproducibility


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


def plot_results(scores_normal, scores_fault, threshold=-0.1):
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


def save_model(model, scaler):
    """Save the trained model and scaler to disk."""
    print("── Saving model files ───────────────────────────────")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Model  saved → {MODEL_PATH}")
    print(f"  ✓ Scaler saved → {SCALER_PATH}")
    print()


def print_summary():
    print("=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
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
    
    # Step 3 — Train model
    model, scaler, X_normal_scaled, X_fault_scaled = train_model(X_normal, X_fault)
    
    # Step 4 — Evaluate
    scores_normal, scores_fault = evaluate_model(model, X_normal_scaled, X_fault_scaled)
    
    # Step 5 — Plot results
    plot_results(scores_normal, scores_fault)
    
    # Step 6 — Save model and scaler
    save_model(model, scaler)
    
    # Step 7 — Summary
    print_summary()