#!/usr/bin/env python3
"""
Adaptive ECG Detector - Main Entry Point
========================================

This script provides the main functionality for training and evaluating
an adaptive ECG classification model using a hybrid CNN-Transformer architecture.

The model classifies ECG signals into 5 categories:
- Normal (N)
- PVC (Premature Ventricular Contraction)
- PAC (Premature Atrial Contraction)
- LBBB (Left Bundle Branch Block)
- RBBB (Right Bundle Branch Block)

The script supports training and evaluation on both CPU and GPU.

Author: Jeffrey Asante (jeffasante)
"""

import torch
import numpy as np
from pathlib import Path
import sys
import argparse

from torch.utils.data import DataLoader, TensorDataset

# sys.path.append(str(Path(__file__).parent / 'src'))

from src.model import ECGHybridModel
from src.data_loader import load_dataset
from src.train import train_model, evaluate_model
from src.utils import evaluate_clinical, adaptive_retrain, benchmark_quantization, load_quantized_model, final_accuracy

def main():
    parser = argparse.ArgumentParser(description='Adaptive ECG Detector')
MODEL_PATH = Path('best_ecg_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def setup_data():
    print("Loading ECG dataset...")
    X_train, y_train, X_test, y_test = load_dataset()

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader

def train(epochs=10):
    print("Starting training...")

    train_loader, val_loader = setup_data()

    # Store original training data for potential retraining
    X_train, y_train, _, _ = load_dataset()

    model = ECGHybridModel(input_dim=2, seq_len=1080, num_classes=5)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    history = train_model(model, train_loader, val_loader, epochs=epochs, device=DEVICE)

    final_acc = evaluate_model(model, val_loader, DEVICE)
    print(f"Final validation accuracy: {final_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Clinical evaluation
    evaluate_clinical(model, val_loader, DEVICE)
    print("Clinical evaluation complete. Plot saved.")

    # Adaptive retraining if needed
    retrained = adaptive_retrain(model, train_loader, val_loader, X_train, y_train, device=DEVICE)
    if retrained:
        print("Model was retrained with synthetic data augmentation.")

    return model

def evaluate():
    if not MODEL_PATH.exists():
        print("No trained model found")
        return

    print("Evaluating model...")

    _, test_loader = setup_data()

    model = ECGHybridModel(input_dim=2, seq_len=1080, num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"Test accuracy: {accuracy:.2f}%")

    return model

def predict_sample(model, ecg_signal):
    model.eval()
    with torch.no_grad():
        if len(ecg_signal.shape) == 2:
            ecg_signal = torch.FloatTensor(ecg_signal).unsqueeze(0)
        elif len(ecg_signal.shape) == 1:
            ecg_signal = torch.FloatTensor(ecg_signal).unsqueeze(0).unsqueeze(-1)

        ecg_signal = ecg_signal.to(DEVICE)
        outputs = model(ecg_signal)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    class_names = ['Normal', 'PVC', 'PAC', 'LBBB', 'RBBB']

    return {
        'prediction': class_names[predicted_class],
        'class_id': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy()
    }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Adaptive ECG Detector')
        parser.add_argument('--train', action='store_true', help='Train a new model')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
        parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
        parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda/mps)')

        args = parser.parse_args()

        # Override device if specified
        device = DEVICE
        if args.device:
            device = torch.device(args.device)

        print(f"Using device: {device}")

        if args.train:
            train()
        elif args.evaluate:
            evaluate()
        else:
            print("Use --train or --evaluate")
    else:
        # Use simple mode variable if no arguments
        mode = 'train'

        print(f"Using device: {DEVICE}")

        if mode == 'train':
            train(epochs=1)
        elif mode == 'evaluate':
            evaluate()
        else:
            print("Invalid mode. Use 'train' or 'evaluate'")
