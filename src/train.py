# Model Architecture (Hybrid CNN-Transformer) [w MPS compatibility]
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Training Function with Retraining Trigger Logic
def should_retrain(model, data_loader, threshold=0.6):
    """Check if model confidence is too low (trigger retraining)"""
    model.eval()
    uncertainties = []
    
    with torch.no_grad():
        for batch_data, _ in data_loader:
            outputs = model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max(dim=1).values
            uncertainty = 1 - confidence
            uncertainties.extend(uncertainty.cpu().numpy())
    
    avg_uncertainty = np.mean(uncertainties)
    print(f"   Average uncertainty: {avg_uncertainty:.3f} (threshold: {1-threshold:.3f})")
    return avg_uncertainty > (1 - threshold)



def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    """Train the hybrid model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    
    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}
    
    print(f" Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        val_acc = evaluate_model(model, val_loader, device)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_ecg_model.pth')
    
    print(f"Training complete! Best validation accuracy: {best_acc:.2f}%")
    return history

def evaluate_model(model, data_loader, device='cuda'):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    return 100 * correct / total
