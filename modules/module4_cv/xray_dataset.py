import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class ChestXRayDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_train=True):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Assuming structure: image_dir/NORMAL/*.jpeg, image_dir/PNEUMONIA/*.jpeg
        for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = os.path.join(image_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpeg', '.png', '.jpg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform