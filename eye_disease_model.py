import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import json
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for eye images
        root_dir should contain 'dry_eyes' and 'no_dry_eyes' folders
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load dry eyes (label = 1)
        dry_eyes_path = os.path.join(root_dir, 'dry_eyes')
        if os.path.exists(dry_eyes_path):
            for img_name in os.listdir(dry_eyes_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(dry_eyes_path, img_name))
                    self.labels.append(1)
        
        # Load no dry eyes (label = 0)
        no_dry_eyes_path = os.path.join(root_dir, 'no_dry_eyes')
        if os.path.exists(no_dry_eyes_path):
            for img_name in os.listdir(no_dry_eyes_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(no_dry_eyes_path, img_name))
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class EyeValidator:
    """Validate if an image contains an eye"""
    def __init__(self):
        # Load OpenCV's pre-trained eye cascade classifier
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def is_eye_image(self, image_path):
        """
        Check if image contains an eye
        Returns: (is_eye, confidence_score)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return False, 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Additional validation: check for face (eyes are usually in faces)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            # Calculate confidence based on detections
            eye_confidence = min(len(eyes) / 2.0, 1.0)  # Expect 1-2 eyes
            face_confidence = min(len(faces) / 1.0, 1.0)  # Expect 1 face
            
            # Combined confidence
            total_confidence = (eye_confidence * 0.7 + face_confidence * 0.3)
            
            # Consider it an eye image if we detect at least one eye
            is_eye = len(eyes) > 0
            
            return is_eye, total_confidence
            
        except Exception as e:
            print(f"Error in eye validation: {e}")
            return False, 0.0

class DryEyeClassifier(nn.Module):
    """Neural network for dry eye classification"""
    def __init__(self, num_classes=2):
        super(DryEyeClassifier, self).__init__()
        
        # Use pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers (unfreeze last 3 blocks)
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "layer3" not in name and "fc" not in name:
                param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EyeDiseasePredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DryEyeClassifier(num_classes=2)
        self.validator = EyeValidator()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.class_names = ['Healthy Eyes', 'Dry Eyes']
    
    def train_model(self, dataset_path, epochs=25, batch_size=32, learning_rate=0.001):
        """Train the dry eye classification model"""
        print(f"Training on device: {self.device}")
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = EyeDataset(dataset_path, transform=train_transform)
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = self.transform
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        epochs_no_improve = 0
        n_epochs_stop = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_loss = running_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_eye_model.pth')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            scheduler.step(val_acc)

            if epochs_no_improve == n_epochs_stop:
                print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {n_epochs_stop} epochs.")
                break
        
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        return train_losses, val_accuracies
    
    def predict(self, image_path):
        """
        Predict dry eye disease from image
        Returns: (prediction, confidence, is_valid_eye)
        """
        # First validate if it's an eye image
        is_eye, eye_confidence = self.validator.is_eye_image(image_path)
        
        if not is_eye:
            return None, 0.0, False
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = self.class_names[predicted.item()]
                confidence_score = confidence.item()
                
                return prediction, confidence_score, True
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0, False
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        print(f"Model loaded from {path}")

# Usage example
def main():
    # Initialize predictor
    predictor = EyeDiseasePredictor()
    
    # Train the model (uncomment to train)
    dataset_path = "BlinkWell/datasets/eyes"  # Path to your dataset
    train_losses, val_accuracies = predictor.train_model(dataset_path, epochs=25)
    
    # Make a prediction (uncomment to test)
    image_path = "path_to_test_image.jpg"
    prediction, confidence, is_valid = predictor.predict(image_path)
    
    if is_valid:
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Invalid eye image or no eye detected")

if __name__ == "__main__":
    main()

