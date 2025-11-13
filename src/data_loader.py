import wfdb
import numpy as np
from pathlib import Path

# Where your data is
DATA_DIR = Path("data/physionet.org/files/mitdb/1.0.0")

# The 5 classes we care about
CLASS_MAP = {
    'N': 0,  # Normal
    'V': 1,  # PVC (Premature Ventricular Contraction)
    'S': 2,  # PAC (Premature Atrial Contraction)
    'L': 3,  # LBBB (Left Bundle Branch Block)
    'R': 4,  # RBBB (Right Bundle Branch Block)
}

# Records to use (patient-level split - critical!)
TRAIN_RECORDS = ['100', '101', '102', '103', '104', '105', '106', '107', '118', '119']
TEST_RECORDS = ['208', '210', '212', '213', '214', '215', '217', '219', '221', '222']

def load_record(record_name):
    """
    Load one MITDB record: returns signal + labels
    """
    # Load signal (returns record.p_signal: [samples, channels])
    record = wfdb.rdrecord(str(DATA_DIR / record_name))
    
    # Load annotations (returns annotation.symbol: beat labels)
    annotation = wfdb.rdann(str(DATA_DIR / record_name), 'atr')
    
    print(f"📊 Record {record_name}: {record.sig_name} leads, {record.fs} Hz")
    print(f"   Channels: {record.sig_name}")  # Usually ['MLII', 'V1'] or similar
    
    return record, annotation

def create_windows(record, annotation, window_sec=3, fs=360):
    """
    Creates fixed windows (e.g., 3 seconds) with MAJORITY label
    """
    window_size = window_sec * fs  # 1080 samples for 3 sec @ 360Hz
    signals = record.p_signal  # [total_samples, 2]
    labels = annotation.symbol  # Beat-level labels
    
    X, y = [], []
    
    # Slide window across the recording
    for start in range(0, len(signals) - window_size, window_size // 2):  # 50% overlap
        end = start + window_size
        
        # Get all beat annotations in this window
        beats_in_window = [
            label for label, sample in zip(labels, annotation.sample)
            if start <= sample < end and label in CLASS_MAP
        ]
        
        if not beats_in_window:
            continue  # Skip windows with no labeled beats
        
        # Assign window label = majority vote
        majority_label = max(set(beats_in_window), key=beats_in_window.count)
        window_label = CLASS_MAP[majority_label]
        
        # Extract signal window
        window_signal = signals[start:end, :]  # [window_size, 2]
        
        # Normalize
        window_signal = (window_signal - np.mean(window_signal)) / (np.std(window_signal) + 1e-8)
        
        X.append(window_signal)
        y.append(window_label)
    
    return np.array(X), np.array(y)

 
def load_dataset():
    """
    Load all training + test records into X, y arrays
    """
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print("="*50)
    print("📥 Loading MIT-BIH Training Records...")
    print("="*50)
    
    for record_name in TRAIN_RECORDS:
        try:
            record, annotation = load_record(record_name)
            X, y = create_windows(record, annotation)
            X_train.append(X)
            y_train.append(y)
            print(f"   Added {len(X)} windows")
        except Exception as e:
            print(f"   Failed {record_name}: {e}")
    
    print("\n" + "="*50)
    print("Loading MIT-BIH Test Records...")
    print("="*50)
    
    for record_name in TEST_RECORDS:
        try:
            record, annotation = load_record(record_name)
            X, y = create_windows(record, annotation)
            X_test.append(X)
            y_test.append(y)
            print(f"   Added {len(X)} windows")
        except Exception as e:
            print(f"   Failed {record_name}: {e}")
    
    # Combine all records
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    
    return X_train, y_train, X_test, y_test


# Load everything (takes ~30 seconds)
# X_train, y_train, X_test, y_test = load_dataset()

# # Preview class distribution
# from collections import Counter
# print(f"\nClass distribution: {dict(Counter(y_train))}")
# Expected: {0: ~5000, 1: ~1000, 2: ~500, 3: ~200, 4: ~300} (imbalanced = realistic!)