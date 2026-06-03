import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from xray_dataset import ChestXRayDataset, get_transforms
import os

def train_xray_model(data_dir, epochs=10, batch_size=32):
    """Train ResNet-18 for chest X-ray classification"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_transform, val_transform = get_transforms()
    full_dataset = ChestXRayDataset(data_dir, transform=train_transform)
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Change transform for validation
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    
    # Freeze all layers except layer4 and FC
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Replace FC layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / len(train_loader.dataset)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'modules/module4_cv/xray_model.pth')
            print(f"  ✅ Saved best model with val_acc: {val_acc:.2f}%")
    
    print(f"\n🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model

if __name__ == "__main__":
    # Download dataset from Kaggle first: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    train_xray_model('data/chest_xray/train', epochs=10)