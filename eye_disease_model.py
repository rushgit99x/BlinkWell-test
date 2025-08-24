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
    """Enhanced validator to check if an image contains an eye with multiple validation methods"""
    def __init__(self):
        # Load OpenCV's pre-trained cascade classifiers
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Additional eye cascade for better detection
        self.eye_tree_eyeglasses = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Validation thresholds - adjusted to be more reasonable
        self.min_confidence = 0.3  # Lowered from 0.6 to be less strict
        self.min_eye_size = (20, 20)  # Lowered from 40x40 to be more permissive
        self.max_eye_size = (300, 300)  # Increased from 200x200 for larger eyes
        
    def preprocess_image(self, img):
        """Preprocess image for better eye detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray
    
    def detect_eyes_haar(self, gray):
        """Detect eyes using Haar cascades with multiple classifiers"""
        eyes = []
        
        # Try different Haar cascade parameters
        scale_factors = [1.05, 1.1, 1.15]
        min_neighbors_list = [3, 5, 7]
        
        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_list:
                # Primary eye cascade
                detected = self.eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=self.min_eye_size,
                    maxSize=self.max_eye_size
                )
                eyes.extend(detected)
                
                # Secondary eye cascade (for glasses)
                detected_glasses = self.eye_tree_eyeglasses.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=self.min_eye_size,
                    maxSize=self.max_eye_size
                )
                eyes.extend(detected_glasses)
        
        return eyes
    
    def validate_eye_characteristics(self, gray, eyes):
        """Validate detected regions have eye-like characteristics"""
        valid_eyes = []
        
        for (x, y, w, h) in eyes:
            # Extract eye region
            eye_roi = gray[y:y+h, x:x+w]
            
            if eye_roi.size == 0:
                continue
            
            # Check aspect ratio (eyes are typically wider than tall) - made less strict
            aspect_ratio = w / h
            if aspect_ratio < 0.8 or aspect_ratio > 4.0:  # More permissive range
                continue
            
            # Check size constraints
            if w < self.min_eye_size[0] or h < self.min_eye_size[1]:
                continue
            if w > self.max_eye_size[0] or h > self.max_eye_size[1]:
                continue
            
            # Check for circular/elliptical shape using contour analysis
            try:
                # Apply threshold to get binary image
                _, thresh = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    # Check if contour area is reasonable (not too small, not too large)
                    roi_area = w * h
                    if area < roi_area * 0.1 or area > roi_area * 0.9:
                        continue
                    
                    # Check circularity - made less strict
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.1:  # Lowered from 0.3 to be more permissive
                            valid_eyes.append((x, y, w, h))
            except:
                continue
        
        return valid_eyes
    
    def check_face_context(self, gray):
        """Check if eyes are within a face context"""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return len(faces) > 0
    
    def analyze_image_quality(self, img):
        """Analyze image quality and characteristics"""
        # Check image size - made less strict
        height, width = img.shape[:2]
        if width < 50 or height < 50:  # Lowered from 100x100
            return False, "Image too small"
        
        # Check if image is too blurry - made less strict
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 20:  # Lowered from 50 to be more permissive
            return False, "Image too blurry"
        
        # Check brightness - made less strict
        mean_brightness = np.mean(gray)
        if mean_brightness < 20 or mean_brightness > 240:  # More permissive range
            return False, "Image too dark or too bright"
        
        return True, "Good quality"
    
    def is_eye_image(self, image_path):
        """
        Enhanced eye validation with multiple checks
        Returns: (is_eye, confidence_score, validation_details)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return False, 0.0, "Failed to read image"
            
            # Analyze image quality first
            quality_ok, quality_msg = self.analyze_image_quality(img)
            if not quality_ok:
                return False, 0.0, quality_msg
            
            # Preprocess image
            gray = self.preprocess_image(img)
            
            # Detect eyes using Haar cascades
            detected_eyes = self.detect_eyes_haar(gray)
            
            # Validate eye characteristics
            valid_eyes = self.validate_eye_characteristics(gray, detected_eyes)
            
            # Check face context
            has_face = self.check_face_context(gray)
            
            # Calculate confidence scores
            eye_count = len(valid_eyes)
            face_bonus = 0.2 if has_face else 0.0
            
            # Base confidence on number of valid eyes - adjusted to be more generous
            if eye_count == 0:
                confidence = 0.0
            elif eye_count == 1:
                confidence = 0.5 + face_bonus  # Increased from 0.4
            elif eye_count == 2:
                confidence = 0.8 + face_bonus  # Increased from 0.7
            else:
                confidence = 0.9 + face_bonus  # Increased from 0.8
            
            # Apply additional penalties for suspicious cases
            if eye_count > 4:  # Too many eyes detected
                confidence *= 0.5
            
            # Check for reasonable eye positioning
            if eye_count >= 2:
                # Check if eyes are roughly at same height and reasonable distance apart
                eyes_sorted = sorted(valid_eyes, key=lambda x: x[0])  # Sort by x coordinate
                if len(eyes_sorted) >= 2:
                    eye1, eye2 = eyes_sorted[0], eyes_sorted[1]
                    y_diff = abs(eye1[1] - eye2[1])
                    x_diff = abs(eye2[0] - eye1[0])
                    
                    # Eyes should be roughly at same height (within 20% of eye height)
                    if y_diff > max(eye1[3], eye2[3]) * 0.2:
                        confidence *= 0.7
                    
                    # Eyes should be reasonable distance apart
                    avg_eye_width = (eye1[2] + eye2[2]) / 2
                    if x_diff < avg_eye_width * 0.5 or x_diff > avg_eye_width * 4:
                        confidence *= 0.6
            
            # Final decision with fallback
            is_eye = confidence >= self.min_confidence
            
            # Fallback: if we have any eye detections and face context, be more permissive
            if not is_eye and eye_count > 0 and has_face:
                is_eye = True
                confidence = max(confidence, 0.4)  # Ensure minimum confidence
                print(f"Fallback validation: Accepted image with {eye_count} eyes and face context")
            
            validation_details = {
                'eye_count': eye_count,
                'has_face': has_face,
                'quality_check': quality_msg,
                'confidence_breakdown': {
                    'base_confidence': confidence - face_bonus,
                    'face_bonus': face_bonus,
                    'final_confidence': confidence
                },
                'fallback_used': not is_eye and eye_count > 0 and has_face
            }
            
            return is_eye, confidence, validation_details
            
        except Exception as e:
            print(f"Error in enhanced eye validation: {e}")
            return False, 0.0, f"Validation error: {str(e)}"

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
        Returns: (prediction, confidence, is_valid_eye, validation_details)
        """
        # First validate if it's an eye image
        is_eye, eye_confidence, validation_details = self.validator.is_eye_image(image_path)
        
        if not is_eye:
            return None, 0.0, False, validation_details
        
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
                
                return prediction, confidence_score, True, validation_details
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0, False, validation_details
    
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

    def test_validation_on_directory(self, test_dir, output_file=None):
        """
        Test the validation system on a directory of images
        Useful for evaluating validation performance
        """
        results = []
        
        if not os.path.exists(test_dir):
            print(f"Test directory {test_dir} does not exist")
            return results
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith(image_extensions)]
        
        print(f"Testing validation on {len(image_files)} images...")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(test_dir, img_file)
            print(f"Processing {i+1}/{len(image_files)}: {img_file}")
            
            is_eye, confidence, details = self.validator.is_eye_image(img_path)
            
            result = {
                'filename': img_file,
                'is_eye': is_eye,
                'confidence': confidence,
                'validation_details': details
            }
            results.append(result)
        
        # Print summary
        total_images = len(results)
        eye_images = sum(1 for r in results if r['is_eye'])
        avg_confidence = sum(r['confidence'] for r in results) / total_images if total_images > 0 else 0
        
        print(f"\nValidation Test Summary:")
        print(f"Total images: {total_images}")
        print(f"Detected as eyes: {eye_images} ({eye_images/total_images*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Save results if output file specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results

# Usage example
def main():
    # Initialize predictor
    predictor = EyeDiseasePredictor()
    
    # Train the model (uncomment to train)
    dataset_path = "BlinkWell/datasets/eyes"  # Path to your dataset
    train_losses, val_accuracies = predictor.train_model(dataset_path, epochs=25)
    
    # Make a prediction (uncomment to test)
    image_path = "path_to_test_image.jpg"
    prediction, confidence, is_valid, validation_details = predictor.predict(image_path)
    
    if is_valid:
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        print("Validation Details:", validation_details)
    else:
        print("Invalid eye image or no eye detected")
        print("Validation Details:", validation_details)

if __name__ == "__main__":
    main()

