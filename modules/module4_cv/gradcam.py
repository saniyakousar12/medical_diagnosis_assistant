import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class XRayGradCAM:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load model
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Target layer for GradCAM
        self.target_layers = [self.model.layer4[-1]]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def generate_heatmap(self, image_path):
        """Generate Grad-CAM heatmap for X-ray image"""
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Generate GradCAM
        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # Convert to RGB
            original_np = np.array(original_image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
        
        # Convert to base64
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        axes[1].imshow(visualization)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        class_names = ['Normal', 'Pneumonia']
        return {
            'prediction': class_names[predicted.item()],
            'confidence': float(confidence.item()),
            'heatmap': image_base64
        }

# Singleton
xray_cam = None

def get_xray_predictor():
    global xray_cam
    if xray_cam is None:
        xray_cam = XRayGradCAM('modules/module4_cv/xray_model.pth')
    return xray_cam