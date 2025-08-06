import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class DryEyeTextDataset(Dataset):
    """Dataset class for text-based features"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class DryEyeTextClassifier(nn.Module):
    """Neural network for text-based dry eye prediction"""
    def __init__(self, input_size=25, hidden_sizes=[128, 64, 32], num_classes=2, dropout_rate=0.3):
        super(DryEyeTextClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DryEyeTextPredictor:
    """Text-based dry eye disease predictor"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = [
            'Gender', 'Age', 'Sleep_duration', 'Sleep_quality', 'Stress_level',
            'Blood_pressure', 'Heart_rate', 'Daily_steps', 'Physical_activity',
            'Height', 'Weight', 'Sleep_disorder', 'Wake_up_during_night',
            'Feel_sleepy_during_day', 'Caffeine_consumption', 'Alcohol_consumption',
            'Smoking', 'Medical_issue', 'Ongoing_medication', 'Smart_device_before_bed',
            'Average_screen_time', 'Blue_light_filter', 'Discomfort_eye_strain',
            'Redness_in_eye', 'Itchiness_irritation_in_eye'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the dataset"""
        df = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['Gender', 'Sleep_disorder', 'Wake_up_during_night',
                           'Feel_sleepy_during_day', 'Caffeine_consumption', 
                           'Alcohol_consumption', 'Smoking', 'Medical_issue',
                           'Ongoing_medication', 'Smart_device_before_bed',
                           'Blue_light_filter', 'Discomfort_eye_strain',
                           'Redness_in_eye', 'Itchiness_irritation_in_eye']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        df[col] = df[col].apply(lambda x: le.transform([x])[0] 
                                               if x in le.classes_ else 0)
        
        # Handle blood pressure (convert string to numeric)
        if 'Blood_pressure' in df.columns:
            df['Blood_pressure'] = df['Blood_pressure'].apply(self._parse_blood_pressure)
        
        # Select features
        feature_cols = [col for col in self.feature_names if col in df.columns]
        X = df[feature_cols].values
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, feature_cols
    
    def _parse_blood_pressure(self, bp_str):
        """Parse blood pressure string to numeric value (systolic)"""
        if pd.isna(bp_str):
            return 120  # default normal value
        
        bp_str = str(bp_str)
        if '/' in bp_str:
            try:
                systolic = int(bp_str.split('/')[0])
                return systolic
            except:
                return 120
        
        try:
            return int(float(bp_str))
        except:
            return 120
    
    def train_model(self, csv_path, epochs=100, batch_size=32, learning_rate=0.001, test_size=0.2):
        """Train the text-based dry eye classification model"""
        print(f"Training on device: {self.device}")
        
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Prepare features and target
        X, feature_cols = self.preprocess_data(df, is_training=True)
        
        if 'Dry_Eye_Disease' in df.columns:
            y = LabelEncoder().fit_transform(df['Dry_Eye_Disease'].astype(str))
        else:
            raise ValueError("Target column 'Dry_Eye_Disease' not found in dataset")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Create datasets and loaders
        train_dataset = DryEyeTextDataset(X_train, y_train)
        val_dataset = DryEyeTextDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = DryEyeTextClassifier(input_size=X_train.shape[1])
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_loss = running_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_acc)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/best_text_model.pth')
            
            scheduler.step(val_acc)
        
        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        
        # Final evaluation
        self._evaluate_model(X_val, y_val)
        
        return train_losses, val_accuracies
    
    def _evaluate_model(self, X_val, y_val):
        """Evaluate model performance"""
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            outputs = self.model(X_val_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        accuracy = accuracy_score(y_val, predicted)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, predicted, target_names=['No Dry Eyes', 'Dry Eyes']))
    
    def predict_from_questionnaire(self, questionnaire_data):
        """
        Predict dry eye disease from questionnaire data
        
        Args:
            questionnaire_data: dict with questionnaire responses
            
        Returns:
            tuple: (prediction_probability, confidence, risk_factors)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        try:
            # Convert questionnaire data to DataFrame
            df = pd.DataFrame([questionnaire_data])
            
            # Preprocess data
            X, _ = self.preprocess_data(df, is_training=False)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get probability of dry eyes (class 1)
                dry_eye_prob = probabilities[0, 1].item()
                
                # Calculate confidence and risk factors
                confidence = max(probabilities[0]).item()
                risk_factors = self._analyze_risk_factors(questionnaire_data, dry_eye_prob)
                
                return dry_eye_prob, confidence, risk_factors
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0, 0.0, []
    
    def _analyze_risk_factors(self, data, dry_eye_prob):
        """Analyze risk factors from questionnaire data"""
        risk_factors = []
        
        # Age risk
        age = data.get('Age', 0)
        if age > 50:
            risk_factors.append({"factor": "Advanced age", "impact": "high", "value": f"{age} years"})
        elif age > 40:
            risk_factors.append({"factor": "Middle age", "impact": "medium", "value": f"{age} years"})
        
        # Screen time risk
        screen_time = data.get('Average_screen_time', 0)
        if screen_time > 8:
            risk_factors.append({"factor": "Excessive screen time", "impact": "high", "value": f"{screen_time} hours/day"})
        elif screen_time > 6:
            risk_factors.append({"factor": "High screen time", "impact": "medium", "value": f"{screen_time} hours/day"})
        
        # Sleep quality risk
        sleep_quality = data.get('Sleep_quality', 10)
        if sleep_quality < 4:
            risk_factors.append({"factor": "Poor sleep quality", "impact": "high", "value": f"Score: {sleep_quality}/10"})
        elif sleep_quality < 6:
            risk_factors.append({"factor": "Below average sleep quality", "impact": "medium", "value": f"Score: {sleep_quality}/10"})
        
        # Stress level risk
        stress_level = data.get('Stress_level', 0)
        if stress_level > 7:
            risk_factors.append({"factor": "High stress levels", "impact": "high", "value": f"Level: {stress_level}/10"})
        elif stress_level > 5:
            risk_factors.append({"factor": "Moderate stress levels", "impact": "medium", "value": f"Level: {stress_level}/10"})
        
        # Environmental factors
        if data.get('Blue_light_filter') == 'N':
            risk_factors.append({"factor": "No blue light protection", "impact": "medium", "value": "Not using filter"})
        
        if data.get('Smart_device_before_bed') == 'Y':
            risk_factors.append({"factor": "Device usage before bed", "impact": "medium", "value": "Using devices before sleep"})
        
        # Existing symptoms
        if data.get('Discomfort_eye_strain') == 'Y':
            risk_factors.append({"factor": "Eye strain discomfort", "impact": "high", "value": "Experiencing symptoms"})
        
        if data.get('Redness_in_eye') == 'Y':
            risk_factors.append({"factor": "Eye redness", "impact": "high", "value": "Visible symptoms"})
        
        if data.get('Itchiness_irritation_in_eye') == 'Y':
            risk_factors.append({"factor": "Eye irritation", "impact": "high", "value": "Experiencing symptoms"})
        
        return risk_factors
    
    def save_model(self, path):
        """Save the trained model and preprocessors"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'input_size': self.model.network[0].in_features,
                'hidden_sizes': [layer.out_features for layer in self.model.network if isinstance(layer, nn.Linear)][:-1],
                'num_classes': self.model.network[-1].out_features
            },
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct model
        arch = checkpoint['model_architecture']
        self.model = DryEyeTextClassifier(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            num_classes=arch['num_classes']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Load preprocessors
        self.scaler = checkpoint['scaler']
        self.label_encoders = checkpoint['label_encoders']
        self.feature_names = checkpoint['feature_names']
        
        print(f"Model loaded from {path}")

def combine_predictions(image_prob, text_prob, image_weight=0.2, text_weight=0.8):
    """
    Combine image and text predictions with specified weights
    
    Args:
        image_prob: Probability from image analysis (0-1)
        text_prob: Probability from text analysis (0-1)
        image_weight: Weight for image prediction (default 0.2)
        text_weight: Weight for text prediction (default 0.8)
    
    Returns:
        Combined probability
    """
    return (image_prob * image_weight) + (text_prob * text_weight)

# Usage example
def main():
    # Initialize predictor
    predictor = DryEyeTextPredictor()
    
    # Train the model (uncomment to train)
    dataset_path = "datasets/eyes/Dry_Eye_Dataset.csv"
    if os.path.exists(dataset_path):
        train_losses, val_accuracies = predictor.train_model(dataset_path, epochs=100)
    
    # Example prediction
    sample_data = {
        'Gender': 'M', 'Age': 45, 'Sleep_duration': 6.5, 'Sleep_quality': 6,
        'Stress_level': 7, 'Blood_pressure': '130/80', 'Heart_rate': 75,
        'Daily_steps': 8000, 'Physical_activity': 5, 'Height': 175, 'Weight': 70,
        'Sleep_disorder': 'N', 'Wake_up_during_night': 'Y', 'Feel_sleepy_during_day': 'Y',
        'Caffeine_consumption': 'Y', 'Alcohol_consumption': 'N', 'Smoking': 'N',
        'Medical_issue': 'N', 'Ongoing_medication': 'N', 'Smart_device_before_bed': 'Y',
        'Average_screen_time': 9.0, 'Blue_light_filter': 'N', 'Discomfort_eye_strain': 'Y',
        'Redness_in_eye': 'Y', 'Itchiness_irritation_in_eye': 'Y'
    }
    
    if predictor.model is not None:
        prob, confidence, risk_factors = predictor.predict_from_questionnaire(sample_data)
        print(f"Dry eye probability: {prob:.3f}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Risk factors: {len(risk_factors)}")

if __name__ == "__main__":
    main()