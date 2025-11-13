# ECG Anomaly Detector

A machine learning system for detecting cardiac arrhythmias from ECG signals using a hybrid CNN-Transformer architecture.

## Overview

This project implements a deep learning model that analyzes 3-second windows of two-lead ECG signals to classify five types of cardiac rhythms: normal beats, premature ventricular contractions (PVC), premature atrial contractions (PAC), left bundle branch block (LBBB), and right bundle branch block (RBBB).

## Architecture

The model uses a hybrid CNN-Transformer approach:

- **CNN branch**: Extracts local morphological features from raw ECG waveforms
- **Transformer branch**: Captures long-range temporal dependencies
- **Fusion layer**: Combines features from both branches for classification

## Dataset

- **Source**: MIT-BIH Arrhythmia Database
- **Size**: 10,777 labeled 3-second windows from 20 patients
- **Leads**: MLII and V5
- **Sampling rate**: 360 Hz
- **Split**: Patient-level separation (training: 10 patients, test: 10 patients)

## Performance

| Model | Accuracy | PVC Sensitivity | PAC Sensitivity | Latency | Size |
|-------|----------|-----------------|-----------------|---------|------|
| Baseline | 81.1% | 85% | 68% | 9.2ms | 1.6MB |
| Retrained | 90.6% | 94% | 78% | 9.2ms | 1.6MB |
| Quantized | 89.8% | 93% | 77% | 2.8ms | 0.4MB |

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

Download the MIT-BIH Arrhythmia Database:

```bash
chmod +x fetch_dataset.sh
./fetch_dataset.sh
```

This script downloads the dataset from PhysioNet and organizes it in the `data/` directory.

## Usage

```
pip install -r requirements.txt 
```

### Training

```bash
python main.py --train
```

### Evaluation

```bash
python main.py --evaluate
```

### Command Line Options

```bash
python main.py --train --epochs 20 --device cuda
python main.py --evaluate --device mps
```

## Key Features

- **Dynamic Retraining**: Automatically generates synthetic data when model confidence drops below threshold
- **Quantization Support**: INT8 quantization for faster inference on edge devices
- **Clinical Validation**: Tested against cardiologist-annotated data
- **Patient-Level Split**: Prevents data leakage between training and validation

## Limitations

- Rare arrhythmia classes (PAC, LBBB) have limited performance due to insufficient training examples
- Model trained on single database; may require fine-tuning for other populations
- Not intended for clinical use without further validation

## Citation

Dataset: MIT-BIH Arrhythmia Database
Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.
