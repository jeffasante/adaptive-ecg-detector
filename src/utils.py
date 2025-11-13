from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def evaluate_clinical(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            outputs = model(batch_data.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    class_names = ['Normal', 'PVC', 'PAC', 'LBBB', 'RBBB']
    
    # This creates a list [0, 1, 2, 3, 4] that corresponds to your class_names
    expected_labels = list(range(len(class_names)))
    
    # This ensures the matrix is always 5x5, even if a class is missing from the data.
    cm = confusion_matrix(all_labels, all_preds, labels=expected_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Real ECG Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    # Classification Report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, 
                              target_names=class_names,
                              labels=expected_labels,
                              zero_division=0)) # Adde
    
    return cm


# Use this function in main.py after training and evaluation to get clinical metrics.

def should_retrain_simple(model, data_loader, threshold=0.6, device='mps'):
    """Check if average confidence < threshold"""
    model.eval()
    confidences = []

    with torch.no_grad():
        for batch_data, _ in data_loader:
            outputs = model(batch_data.to(device))
            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max(dim=1).values
            confidences.extend(confidence.cpu().numpy())

    avg_confidence = np.mean(confidences)
    print(f"   Average confidence: {avg_confidence:.3f} (threshold: {threshold})")
    return avg_confidence < threshold

def generate_synthetic_from_real(real_sample, label, n_augment=5):
    """Generate realistic synthetic ECG augmentations"""
    from scipy import signal

    synth_data = []
    for _ in range(n_augment):
        # Make a copy
        sample = real_sample.clone().cpu().numpy()

        # 1. Subtle time warping (95-105%)
        stretch = np.random.uniform(0.95, 1.05)
        warped = np.zeros_like(sample)
        for ch in range(sample.shape[1]):  # For each lead
            warped_ch = signal.resample(sample[:, ch],
                                       int(len(sample) * stretch))
            # Pad or truncate
            if len(warped_ch) > len(sample):
                warped_ch = warped_ch[:len(sample)]
            else:
                pad_len = len(sample) - len(warped_ch)
                warped_ch = np.pad(warped_ch, (0, pad_len), 'edge')
            warped[:, ch] = warped_ch

        # 2. Add EMG noise (physiological)
        noise = np.random.normal(0, 0.02, warped.shape)

        # 3. Baseline wander
        t = np.linspace(0, 1, len(warped))
        wander = np.sin(2 * np.pi * 0.5 * t) * 0.03
        warped += noise + wander.reshape(-1, 1)

        # 4. Normalize
        warped = (warped - warped.mean()) / (warped.std() + 1e-8)

        synth_data.append(warped)

    return torch.FloatTensor(np.stack(synth_data)), torch.LongTensor([label] * n_augment)

def adaptive_retrain(model, train_loader, val_loader, X_train, y_train, device='mps', threshold=0.6):
    """
    Adaptive retraining function that checks model confidence and retrains with synthetic data if needed.

    Args:
        model: The trained model to evaluate
        train_loader: Training data loader
        val_loader: Validation data loader
        X_train: Original training data (numpy array)
        y_train: Original training labels (numpy array)
        device: Device to run on
        threshold: Confidence threshold for retraining

    Returns:
        bool: True if retraining occurred, False otherwise
    """
    print("Testing Dynamic Retraining")
    print("="*50)

    if should_retrain_simple(model, val_loader, threshold=threshold, device=device):
        print("Low confidence detected! Triggering synthetic augmentation...")

        # Find uncertain samples
        model.eval()
        uncertain_samples = []
        uncertain_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = model(batch_data.to(device))
                probs = torch.softmax(outputs, dim=1)
                confidence = probs.max(dim=1).values
                low_conf_mask = confidence < threshold

                if low_conf_mask.any():
                    uncertain_samples.append(batch_data[low_conf_mask])
                    uncertain_labels.append(batch_labels[low_conf_mask])

        if uncertain_samples:
            X_uncertain = torch.cat(uncertain_samples, dim=0)
            y_uncertain = torch.cat(uncertain_labels, dim=0)

            # Generate synthetic data
            synth_X, synth_y = [], []
            for idx in range(min(10, len(X_uncertain))):  # Limit to 10 samples
                synth_x, synth_y_batch = generate_synthetic_from_real(
                    X_uncertain[idx], y_uncertain[idx], n_augment=3
                )
                synth_X.append(synth_x)
                synth_y.append(synth_y_batch)

            X_synth = torch.cat(synth_X, dim=0)
            y_synth = torch.cat(synth_y, dim=0)

            # Combine with training data
            X_combined = torch.cat([torch.FloatTensor(X_train), X_synth], dim=0)
            y_combined = torch.cat([torch.LongTensor(y_train), y_synth], dim=0)

            # Retrain
            combined_dataset = TensorDataset(X_combined, y_combined)
            combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

            print(f"   Retraining on {len(combined_dataset)} samples...")
            # Import train_model here to avoid circular imports
            from .train import train_model

            train_model(model, combined_loader, val_loader, epochs=3, device=device)
            torch.save(model.state_dict(), 'retrained_model.pth')
            print("Retraining completed and model saved as 'retrained_model.pth'")
            return True

    return False

def benchmark_quantization(model, val_loader, device):
    """
    Benchmark quantization using the modern FX Graph Mode API,
    which is compatible with complex architectures like Transformers.
    """
    import time
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    # Ensure model is on the correct device for FP32 benchmark
    model = model.to(device)
    model.eval()
    
    # FP32 baseline
    times_fp32 = []
    with torch.no_grad():
        for batch_data, _ in val_loader:
            batch_data = batch_data.to(device)
            start = time.perf_counter()
            model(batch_data)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            times_fp32.append(time.perf_counter() - start)
    
    # Batch size is the second dimension of the loader
    batch_size = val_loader.batch_size if val_loader.batch_size else 32
    fp32_latency = np.mean(times_fp32) / batch_size * 1000
    
    # Move to CPU for Quantization using FX Graph Mode
    print("   Moving model to CPU for quantization...")
    model_cpu = model.to('cpu').eval()
    
    # Ensure all parameters are in float32
    for param in model_cpu.parameters():
        param.data = param.data.float()
    
    print("   Setting quantized engine to 'qnnpack' for ARM CPU...")
    torch.backends.quantized.engine = 'qnnpack'

    # Define the mapping from modules to quantization configurations.
    # We'll apply dynamic quantization to Linear and Conv1d layers.
    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)

    # Get a representative example input for tracing the model's graph.
    example_inputs = next(iter(val_loader))[0].to('cpu')

    # Prepare the model for quantization. This traces the model and inserts observers.
    print("   Preparing model with FX Graph Mode...")
    prepared_model = prepare_fx(model_cpu, qconfig_mapping, example_inputs)

    # Convert the prepared model to a final quantized model.
    print("   Converting model to quantized version...")
    quantized_model = convert_fx(prepared_model)

    # INT8 benchmark on CPU
    print("   Benchmarking INT8 on CPU...")
    times_int8 = []
    with torch.no_grad():
        for batch_data, _ in val_loader:
            batch_data_cpu = batch_data.to('cpu')
            start = time.perf_counter()
            quantized_model(batch_data_cpu)
            times_int8.append(time.perf_counter() - start)
    
    int8_latency = np.mean(times_int8) / batch_size * 1000
    
    print(f"\n--- BENCHMARK RESULTS ---")
    print(f"   FP32 ({device}): {fp32_latency:.2f} ms/sample")
    print(f"   INT8 (CPU): {int8_latency:.2f} ms/sample")
    if int8_latency > 0:
        print(f"   Speedup: {fp32_latency / int8_latency:.1f}x")
    
    return quantized_model

def load_quantized_model(model_class, model_path, input_shape=(1, 1080, 2)):
    """
    Load a quantized model from saved state dict.
    """
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    device = torch.device('cpu')
    print(f"Using device for quantized inference: {device}")

    # Create a fresh instance of the original model architecture
    fp32_model = model_class().to(device)
    fp32_model.eval()

    # Create the quantized model skeleton
    print("Re-creating quantized model architecture...")
    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
    example_inputs = torch.randn(*input_shape).to(device)

    prepared_model = prepare_fx(fp32_model, qconfig_mapping, example_inputs)
    quantized_model_skeleton = convert_fx(prepared_model)

    # Load the saved weights
    print("Loading saved quantized weights...")
    state_dict = torch.load(model_path, map_location=device)
    quantized_model_skeleton.load_state_dict(state_dict)

    quantized_model_skeleton.eval()
    print("Quantized model successfully loaded and is ready for inference!")

    return quantized_model_skeleton

def final_accuracy(model, data_loader, device):
    """
    Calculate final accuracy on a dataset.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            outputs = model(batch_data.to(device))
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels.to(device)).sum().item()
    
    return 100 * correct / total
