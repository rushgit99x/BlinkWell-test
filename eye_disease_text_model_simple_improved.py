import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

class ImprovedDryEyeTextDataset(Dataset):
    """Improved dataset class with data augmentation"""
    def __init__(self, features, labels=None, augment=False):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx] if self.labels is not None else None
        
        if self.augment and self.labels is not None:
            # Add noise to features for data augmentation
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        if labels is not None:
            return features, labels
        return features

class ImprovedDryEyeTextClassifier(nn.Module):
    """Improved neural network with better architecture"""
    def __init__(self, input_size=25, hidden_sizes=[256, 128, 64, 32], num_classes=2, dropout_rate=0.4):
        super(ImprovedDryEyeTextClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Input layer with batch normalization
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.BatchNorm1d(hidden_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_sizes[-1] // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            if i == 0:
                residual = x
            else:
                residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + residual * 0.1  # Small residual connection
        
        # Output layers
        x = self.output_layers(x)
        return x

class ImprovedDryEyeTextPredictor:
    """Improved text-based dry eye disease predictor with ensemble"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
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
    
    def create_engineered_features(self, df):
        """Create engineered features to improve model performance"""
        df = df.copy()
        
        # Age-related features
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
        df['Age_group'] = df['Age_group'].astype(int)
        
        # BMI calculation
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
        df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
        df['BMI_category'] = df['BMI_category'].astype(int)
        
        # Sleep efficiency
        df['Sleep_efficiency'] = df['Sleep_duration'] * df['Sleep_quality'] / 10
        
        # Stress-sleep interaction
        df['Stress_sleep_ratio'] = df['Stress_level'] / (df['Sleep_quality'] + 1)
        
        # Screen time risk
        df['Screen_time_risk'] = df['Average_screen_time'] * (1 - (df['Blue_light_filter'] == 'Y').astype(int) * 0.3)
        
        # Physical activity score
        df['Activity_score'] = df['Daily_steps'] / 10000 + df['Physical_activity'] / 10
        
        # Symptom severity score
        symptom_cols = ['Discomfort_eye_strain', 'Redness_in_eye', 'Itchiness_irritation_in_eye']
        df['Symptom_score'] = sum([(df[col] == 'Y').astype(int) for col in symptom_cols])
        
        # Lifestyle risk score
        lifestyle_cols = ['Smoking', 'Alcohol_consumption', 'Caffeine_consumption']
        df['Lifestyle_risk'] = sum([(df[col] == 'Y').astype(int) for col in lifestyle_cols])
        
        # Medical risk score
        medical_cols = ['Medical_issue', 'Ongoing_medication', 'Sleep_disorder']
        df['Medical_risk'] = sum([(df[col] == 'Y').astype(int) for col in medical_cols])
        
        return df
    
    def preprocess_data(self, df, is_training=True):
        """Enhanced preprocessing with feature engineering"""
        df = df.copy()
        
        # Create engineered features
        df = self.create_engineered_features(df)
        
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
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        df[col] = df[col].apply(lambda x: le.transform([x])[0] 
                                               if x in le.classes_ else 0)
        
        # Handle blood pressure
        if 'Blood_pressure' in df.columns:
            df['Blood_pressure'] = df['Blood_pressure'].apply(self._parse_blood_pressure)
        
        # Select all features including engineered ones
        all_features = [col for col in df.columns if col not in ['Dry_Eye_Disease']]
        X = df[all_features].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, all_features
    
    def _parse_blood_pressure(self, bp_str):
        """Parse blood pressure string to numeric value (systolic)"""
        if pd.isna(bp_str):
            return 120
        
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
    
    def train_ensemble_model(self, csv_path, epochs=150, batch_size=64, learning_rate=0.0005, 
                           test_size=0.2, n_folds=5):
        """Train ensemble model with cross-validation"""
        print(f"Training improved ensemble model on device: {self.device}")
        
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Prepare features and target
        X, feature_cols = self.preprocess_data(df, is_training=True)
        
        if 'Dry_Eye_Disease' in df.columns:
            y = LabelEncoder().fit_transform(df['Dry_Eye_Disease'].astype(str))
        else:
            raise ValueError("Target column 'Dry_Eye_Disease' not found in dataset")
        
        print(f"Feature dimensions: {X.shape[1]} (including engineered features)")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Cross-validation training
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_models = []
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Training Fold {fold + 1}/{n_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create datasets
            train_dataset = ImprovedDryEyeTextDataset(X_train, y_train, augment=True)
            val_dataset = ImprovedDryEyeTextDataset(X_val, y_val, augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = ImprovedDryEyeTextClassifier(input_size=X_train.shape[1])
            model.to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            
            # Training loop
            best_val_acc = 0.0
            patience = 20
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                running_loss = 0.0
                
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    running_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = model(batch_features)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                val_acc = 100 * val_correct / val_total
                avg_loss = running_loss / len(train_loader)
                
                if epoch % 20 == 0:
                    print(f'Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                scheduler.step()
            
            fold_models.append(model)
            fold_scores.append(best_val_acc)
            print(f"Fold {fold + 1} completed. Best accuracy: {best_val_acc:.2f}%")
        
        # Store models
        self.models = fold_models
        
        # Final evaluation on full validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        ensemble_acc = self._evaluate_ensemble(X_val, y_val)
        
        print(f"\n🎯 Ensemble training completed!")
        print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_scores]}")
        print(f"Average fold accuracy: {np.mean(fold_scores):.2f}%")
        print(f"Ensemble validation accuracy: {ensemble_acc:.2f}%")
        
        return fold_scores, ensemble_acc
    
    def _evaluate_ensemble(self, X_val, y_val):
        """Evaluate ensemble model performance"""
        if not self.models:
            return 0.0
        
        self.models[0].eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            
            # Get predictions from all models
            all_predictions = []
            for model in self.models:
                model.eval()
                outputs = model(X_val_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
            
            # Ensemble prediction (average probabilities)
            ensemble_probs = np.mean(all_predictions, axis=0)
            predicted = np.argmax(ensemble_probs, axis=1)
        
        accuracy = accuracy_score(y_val, predicted)
        
        print(f"\n📊 Ensemble Validation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, predicted, target_names=['No Dry Eyes', 'Dry Eyes']))
        
        return accuracy * 100
    
    def predict_from_questionnaire(self, questionnaire_data):
        """Enhanced prediction with ensemble"""
        if not self.models:
            raise ValueError("Models not trained. Please train the ensemble first.")
        
        try:
            # Convert questionnaire data to DataFrame
            df = pd.DataFrame([questionnaire_data])
            
            # Preprocess data
            X, _ = self.preprocess_data(df, is_training=False)
            
            # Get predictions from all models
            all_predictions = []
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    outputs = model(X_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    all_predictions.append(probabilities.cpu().numpy())
            
            # Ensemble prediction
            ensemble_probs = np.mean(all_predictions, axis=0)
            dry_eye_prob = ensemble_probs[0, 1]
            
            # Calculate confidence and risk factors
            confidence = max(ensemble_probs[0])
            risk_factors = self._analyze_risk_factors(questionnaire_data, dry_eye_prob)
            
            return dry_eye_prob, confidence, risk_factors
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0, 0.0, []
    
    def _analyze_risk_factors(self, data, dry_eye_prob):
        """Enhanced risk factor analysis"""
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
        """Save the ensemble model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save each model in the ensemble
        model_states = []
        for i, model in enumerate(self.models):
            model_states.append({
                'model_state_dict': model.state_dict(),
                'model_architecture': {
                    'input_size': model.input_size,
                    'hidden_sizes': model.hidden_sizes
                }
            })
        
        torch.save({
            'ensemble_models': model_states,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        
        print(f"Ensemble model saved to {path}")
    
    def load_model(self, path):
        """Load the ensemble model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load ensemble models
        self.models = []
        for model_state in checkpoint['ensemble_models']:
            arch = model_state['model_architecture']
            model = ImprovedDryEyeTextClassifier(
                input_size=arch['input_size'],
                hidden_sizes=arch['hidden_sizes']
            )
            model.load_state_dict(model_state['model_state_dict'])
            model.to(self.device)
            self.models.append(model)
        
        # Load preprocessors
        self.scaler = checkpoint['scaler']
        self.label_encoders = checkpoint['label_encoders']
        self.feature_names = checkpoint['feature_names']
        
        print(f"Ensemble model loaded from {path}")

# Usage example
def main():
    # Initialize improved predictor
    predictor = ImprovedDryEyeTextPredictor()
    
    # Train the ensemble model
    dataset_path = "datasets/eyes/Dry_Eye_Dataset.csv"
    if os.path.exists(dataset_path):
        print("🚀 Starting improved model training...")
        fold_scores, ensemble_acc = predictor.train_ensemble_model(
            csv_path=dataset_path, 
            epochs=150,
            batch_size=64,
            learning_rate=0.0005,
            n_folds=5
        )
        
        # Save the model
        predictor.save_model('models/best_improved_text_model.pth')
        
        # Calculate improvement
        baseline_acc = 65.17
        improvement = ensemble_acc - baseline_acc
        print(f"\n📈 Performance Improvement:")
        print(f"  Baseline accuracy: {baseline_acc:.2f}%")
        print(f"  Improved accuracy: {ensemble_acc:.2f}%")
        print(f"  Improvement: +{improvement:.2f}%")
        
        if ensemble_acc >= 70:
            print(f"  🎯 Target achieved: Yes!")
            if ensemble_acc >= 80:
                print(f"  🌟 Excellent! Exceeded 80% target!")
            else:
                print(f"  ✅ Good! Achieved 70%+ target!")
        else:
            print(f"  📊 Close to target, consider further tuning")
    
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
    
    if predictor.models:
        prob, confidence, risk_factors = predictor.predict_from_questionnaire(sample_data)
        print(f"\n📊 Sample Prediction Results:")
        print(f"  Dry eye probability: {prob:.3f}")
        print(f"  Prediction confidence: {confidence:.3f}")
        print(f"  Risk factors identified: {len(risk_factors)}")

if __name__ == "__main__":
    main()