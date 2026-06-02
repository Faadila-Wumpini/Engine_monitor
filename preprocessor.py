# preprocessor.py
# This file handles all data cleaning and feature extraction.
# It works for BOTH the dataset (training) and live ESP32 sensor data (inference).

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew


# ── STEP 1: BUTTERWORTH FILTER ────────────────────────────────────────────────
# This removes electrical noise from the raw sensor signal.
# Think of it like noise-cancelling headphones for your data.
def butterworth_filter(data, cutoff=0.3, fs=1.0, order=4):
    """
    Apply a low-pass Butterworth filter to a signal.
    - data   : list or array of raw sensor readings
    - cutoff : frequency cutoff (0.3 works well for our data)
    - fs     : sampling frequency (1.0 for dataset, 50.0 for live ESP32 vibration)
    - order  : filter strength (4 is standard)
    """
    nyq = 0.5 * fs                          # Nyquist frequency
    normal_cutoff = cutoff / nyq
    normal_cutoff = min(normal_cutoff, 0.99)  # must stay below 1.0
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Need at least 15 samples for the filter to work
    if len(data) < 15:
        return data
    
    filtered = filtfilt(b, a, data)
    return filtered


# ── STEP 2: FEATURE EXTRACTION ────────────────────────────────────────────────
# Raw numbers are too noisy for ML. We summarise each window of data
# into meaningful statistics that the model can actually learn from.
def extract_features(window: pd.DataFrame) -> dict:
    """
    Given a window (chunk) of sensor readings, extract statistical features.
    Returns a dictionary of features.
    
    For the AI4I dataset the columns are:
      - Air temperature [K]
      - Process temperature [K]
      - Rotational speed [rpm]   ← acts like vibration
      - Torque [Nm]              ← acts like vibration magnitude
      - Tool wear [min]
    """
    features = {}
    
    # We extract features from each numeric column
    numeric_cols = window.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        vals = window[col].values.astype(float)
        
        if len(vals) == 0:
            continue
        
        col_clean = col.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_')
        
        # Basic statistics
        features[f'{col_clean}_mean']    = np.mean(vals)
        features[f'{col_clean}_std']     = np.std(vals)
        features[f'{col_clean}_min']     = np.min(vals)
        features[f'{col_clean}_max']     = np.max(vals)
        features[f'{col_clean}_range']   = np.max(vals) - np.min(vals)
        
        # RMS — Root Mean Square — captures energy/intensity of the signal
        features[f'{col_clean}_rms']     = np.sqrt(np.mean(vals**2))
        
        # Kurtosis — how "spiky" the signal is (high kurtosis = fault signatures)
        features[f'{col_clean}_kurtosis'] = kurtosis(vals) if len(vals) > 3 else 0
        
        # Skewness — asymmetry of the signal distribution
        features[f'{col_clean}_skew']    = skew(vals) if len(vals) > 3 else 0
        
        # Peak-to-peak — total swing of the signal
        features[f'{col_clean}_peak2peak'] = np.max(vals) - np.min(vals)
    
    return features


# ── STEP 3: PROCESS A FULL DATAFRAME ─────────────────────────────────────────
# This takes a big dataframe and processes it in chunks (windows).
# Each window becomes one row of features for the ML model.
def process_dataframe(df: pd.DataFrame, window_size: int = 50, step: int = 25) -> pd.DataFrame:
    """
    Slide a window across the dataframe and extract features from each window.
    
    - window_size : how many rows per window (50 = 50 data points per chunk)
    - step        : how many rows to move forward each time (25 = 50% overlap)
    
    Returns a DataFrame where each row = features from one window.
    """
    feature_rows = []
    
    # Drop non-numeric / identifier columns that aren't useful for ML
    cols_to_drop = ['UDI', 'Product ID', 'Type', 'Machine failure',
                    'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_features = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Slide the window
    for start in range(0, len(df_features) - window_size + 1, step):
        window = df_features.iloc[start : start + window_size]
        features = extract_features(window)
        feature_rows.append(features)
    
    if not feature_rows:
        print("⚠ Warning: Not enough data to create any windows.")
        return pd.DataFrame()
    
    return pd.DataFrame(feature_rows).fillna(0)


# ── STEP 4: NORMALISE FEATURES ────────────────────────────────────────────────
# Scales all feature values to the same range so no single feature dominates.
def normalise(df: pd.DataFrame, scaler=None):
    """
    Min-Max normalisation. Returns (normalised_df, fitted_scaler).
    Pass in a fitted scaler during inference so training scale is reused.
    """
    from sklearn.preprocessing import MinMaxScaler
    
    if scaler is None:
        scaler = MinMaxScaler()
        normalised = scaler.fit_transform(df)
    else:
        normalised = scaler.transform(df)
    
    return pd.DataFrame(normalised, columns=df.columns), scaler


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing preprocessor with dummy data...")
    
    # Create fake sensor data to test
    np.random.seed(42)
    dummy = pd.DataFrame({
        'Air temperature [K]'     : np.random.normal(300, 2, 200),
        'Process temperature [K]' : np.random.normal(310, 1, 200),
        'Rotational speed [rpm]'  : np.random.normal(1500, 100, 200),
        'Torque [Nm]'             : np.random.normal(40, 10, 200),
        'Tool wear [min]'         : np.arange(200),
    })
    
    result = process_dataframe(dummy, window_size=50, step=25)
    print(f"✓ Processed {len(dummy)} rows → {len(result)} feature windows")
    print(f"✓ Features per window: {len(result.columns)}")
    print(f"\nFirst few features: {list(result.columns[:6])}")
    print("\npreprocessor.py is working correctly!")