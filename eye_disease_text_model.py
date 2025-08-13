import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedDryEyeTextDataset(Dataset):
    """Enhanced dataset class with data augmentation"""
    def __init__(self, features, labels=None, augment=True):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        
        # Data augmentation during training
        if self.augment and self.labels is not None and torch.rand(1).item() < 0.3:
            features = self._augment_features(features)
        
        if self.labels is not None:
            return features, self.labels[idx]
        return features
    
    def _augment_features(self, features):
        """Apply data augmentation to features"""
        # Add small random noise
        noise = torch.randn_like(features) * 0.01
        features = features + noise
        
        # Random feature scaling (0.95 to 1.05)
        scale = torch.rand(features.shape) * 0.1 + 0.95
        features = features * scale
        
        return features

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(input_size)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class AttentionModule(nn.Module):
    """Attention mechanism for feature importance"""
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class AdvancedDryEyeTextClassifier(nn.Module):
    """Enhanced neural network with advanced techniques"""
    def __init__(self, input_size=25, hidden_sizes=[256, 128, 64, 32], num_classes=2, dropout_rate=0.4):
        super(AdvancedDryEyeTextClassifier, self).__init__()
        
        self.input_size = input_size
        self.attention = AttentionModule(input_size)
        
        # Main network with residual connections
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                # First layer
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
            else:
                # Add residual blocks for deeper layers
                if prev_size == hidden_size:
                    layers.append(ResidualBlock(hidden_size, hidden_size // 2, dropout_rate))
                else:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
            prev_size = hidden_size
        
        # Output layer with better initialization
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # Initialize weights for better training
        self._initialize_weights()
        
        self.network = nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply attention
        x = self.attention(x)
        
        # Main network
        x = self.network(x)
        
        # Output
        return self.output_layer(x)

class EnsembleClassifier:
    """Ensemble of multiple models for better performance"""
    def __init__(self, models, weights=None, device=None):
        self.models = models
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict_proba(self, X):
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # For neural networks
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    outputs = model(X_tensor)
                    pred = torch.softmax(outputs, dim=1).numpy()
            predictions.append(pred)
        
        # Weighted average
        final_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            final_pred += pred * weight
        
        return final_pred
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class AdvancedDryEyeTextPredictor:
    """Enhanced text-based dry eye disease predictor"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.ensemble_models = []
        self.scaler = RobustScaler()  # More robust to outliers
        self.label_encoders = {}
        self.feature_selector = None
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
        """Create engineered features for better model performance"""
        df = df.copy()
        
        # Age groups
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
        df['Age_group'] = df['Age_group'].astype(int)
        
        # BMI calculation
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
        df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
        df['BMI_category'] = df['BMI_category'].astype(int)
        
        # Sleep efficiency
        df['Sleep_efficiency'] = df['Sleep_quality'] / df['Sleep_duration']
        
        # Stress-sleep interaction
        df['Stress_sleep_interaction'] = df['Stress_level'] * (10 - df['Sleep_quality'])
        
        # Screen time risk
        df['Screen_time_risk'] = df['Average_screen_time'] * (10 - df['Sleep_quality'])
        
        # Physical activity ratio
        df['Activity_ratio'] = df['Physical_activity'] / df['Daily_steps']
        
        # Health risk score
        health_risk_cols = ['Sleep_disorder', 'Medical_issue', 'Ongoing_medication']
        df['Health_risk_score'] = df[health_risk_cols].apply(lambda x: x.map({'Y': 1, 'N': 0})).sum(axis=1)
        
        # Lifestyle risk score
        lifestyle_risk_cols = ['Smoking', 'Alcohol_consumption', 'Smart_device_before_bed']
        df['Lifestyle_risk_score'] = df[lifestyle_risk_cols].apply(lambda x: x.map({'Y': 1, 'N': 0})).sum(axis=1)
        
        # Symptom severity
        symptom_cols = ['Discomfort_eye_strain', 'Redness_in_eye', 'Itchiness_irritation_in_eye']
        df['Symptom_severity'] = df[symptom_cols].apply(lambda x: x.map({'Y': 1, 'N': 0})).sum(axis=1)
        
        return df
    
    def select_features(self, X, y, method='mutual_info', k=20):
        """Select most important features"""
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support()
        
        return X_selected, selected_features, selector
    
    def preprocess_data(self, df, is_training=True):
        """Enhanced preprocessing with feature engineering"""
        df = df.copy()

        # Normalize incoming CSV column names to expected underscore format
        column_rename_map = {
            'Sleep duration': 'Sleep_duration',
            'Sleep quality': 'Sleep_quality',
            'Stress level': 'Stress_level',
            'Blood pressure': 'Blood_pressure',
            'Heart rate': 'Heart_rate',
            'Daily steps': 'Daily_steps',
            'Physical activity': 'Physical_activity',
            'Sleep disorder': 'Sleep_disorder',
            'Wake up during night': 'Wake_up_during_night',
            'Feel sleepy during day': 'Feel_sleepy_during_day',
            'Caffeine consumption': 'Caffeine_consumption',
            'Alcohol consumption': 'Alcohol_consumption',
            'Medical issue': 'Medical_issue',
            'Ongoing medication': 'Ongoing_medication',
            'Smart device before bed': 'Smart_device_before_bed',
            'Average screen time': 'Average_screen_time',
            'Blue-light filter': 'Blue_light_filter',
            'Discomfort Eye-strain': 'Discomfort_eye_strain',
            'Redness in eye': 'Redness_in_eye',
            'Itchiness/Irritation in eye': 'Itchiness_irritation_in_eye'
        }
        # Strip whitespace from column names and then rename
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=column_rename_map)
        
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
        all_feature_cols = [col for col in df.columns if col != 'Dry_Eye_Disease']
        X = df[all_feature_cols].values
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Apply the same feature selection during inference to avoid shape mismatch
        if not is_training and self.feature_selector is not None:
            try:
                X = self.feature_selector.transform(X)
            except Exception:
                pass
        
        return X, all_feature_cols
    
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
    
    def train_ensemble(self, csv_path, epochs=150, batch_size=64, learning_rate=0.001, test_size=0.2):
        """Train ensemble of models for better performance"""
        print(f"Training ensemble on device: {self.device}")
        
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Prepare features and target
        X, feature_cols = self.preprocess_data(df, is_training=True)
        
        if 'Dry_Eye_Disease' in df.columns:
            y = LabelEncoder().fit_transform(df['Dry_Eye_Disease'].astype(str))
        else:
            raise ValueError("Target column 'Dry_Eye_Disease' not found in dataset")
        
        # Feature selection
        X_selected, selected_features, self.feature_selector = self.select_features(X, y, k='all')
        print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Train multiple models
        models = []
        
        # 1. Neural Network
        print("\nTraining Neural Network...")
        nn_model = self._train_neural_network(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)
        models.append(('Neural Network', nn_model))
        self.model = nn_model
        
        # 2. Random Forest
        print("\nTraining Random Forest...")
        rf_model = self._train_random_forest(X_train, y_train, X_val, y_val)
        models.append(('Random Forest', rf_model))
        self.rf_model = rf_model
        
        # 3. Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb_model = self._train_gradient_boosting(X_train, y_train, X_val, y_val)
        models.append(('Gradient Boosting', gb_model))
        self.gb_model = gb_model

        # 4. Extra Trees
        print("\nTraining Extra Trees...")
        et_model = self._train_extra_trees(X_train, y_train, X_val, y_val)
        models.append(('Extra Trees', et_model))
        self.et_model = et_model
        
        # Create ensemble
        self.ensemble_models = [model for _, model in models]
        
        # Optimize ensemble weights on validation set
        self.ensemble_weights = self._optimize_ensemble_weights(self.ensemble_models, X_val, y_val)
        
        # Evaluate ensemble
        ensemble_accuracy = self._evaluate_ensemble(X_val, y_val)
        print(f"\nEnsemble Validation Accuracy: {ensemble_accuracy:.4f}")
        
        return models
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
        """Train the neural network with advanced techniques"""
        # Create datasets and loaders
        train_dataset = AdvancedDryEyeTextDataset(X_train, y_train, augment=True)
        val_dataset = AdvancedDryEyeTextDataset(X_val, y_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = AdvancedDryEyeTextClassifier(input_size=X_train.shape[1])
        model.to(self.device)
        
        # Loss and optimizer
        # Class weights to address class imbalance
        class_counts = np.bincount(y_train)
        # Inverse frequency weighting
        weights = (len(y_train) / (2.0 * class_counts)).astype(np.float32)
        class_weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
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
            
            if epoch % 10 == 0:
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
        
        print(f'Neural Network training completed. Best validation accuracy: {best_val_acc:.2f}%')
        return model
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest classifier"""
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f'Random Forest validation accuracy: {val_acc:.4f}')
        
        return rf
    
    def _train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting classifier"""
        gb = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        val_pred = gb.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f'Gradient Boosting validation accuracy: {val_acc:.4f}')
        
        return gb

    def _train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Train Extra Trees classifier"""
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_train, y_train)
        val_pred = et.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f'Extra Trees validation accuracy: {val_acc:.4f}')
        return et

    def _optimize_ensemble_weights(self, models, X_val, y_val):
        """Simple weight search to maximize validation accuracy"""
        if not models:
            return None
        # Try a small grid of weights (NN given higher prior)
        candidate_weights = []
        for w_nn in [0.2, 0.3, 0.4, 0.5]:
            for w_rf in [0.1, 0.2, 0.3]:
                for w_gb in [0.1, 0.2, 0.3]:
                    for w_et in [0.1, 0.2, 0.3]:
                        weights = np.array([w_nn, w_rf, w_gb, w_et])
                        weights = weights / weights.sum()
                        candidate_weights.append(weights)
        
        best_acc = -1
        best_weights = None
        for weights in candidate_weights:
            ensemble = EnsembleClassifier(models, weights=weights.tolist(), device=self.device)
            pred = ensemble.predict(X_val)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best_weights = weights.tolist()
        
        print(f"Optimized ensemble weights: {best_weights} (val acc: {best_acc:.4f})")
        self.ensemble_weights = best_weights
        return best_weights
    
    def _evaluate_ensemble(self, X_val, y_val):
        """Evaluate ensemble performance"""
        if not self.ensemble_models:
            return 0.0
        ensemble = EnsembleClassifier(self.ensemble_models, weights=self.ensemble_weights, device=self.device)
        predictions = ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        return accuracy
    
    def train_model(self, csv_path, epochs=150, batch_size=64, learning_rate=0.001, test_size=0.2):
        """Main training method - now uses ensemble approach"""
        return self.train_ensemble(csv_path, epochs, batch_size, learning_rate, test_size)
    
    def predict_from_questionnaire(self, questionnaire_data):
        """
        Predict dry eye disease from questionnaire data
        
        Args:
            questionnaire_data: dict with questionnaire responses
            
        Returns:
            tuple: (prediction_probability, confidence, risk_factors)
        """
        if self.model is None and not self.ensemble_models:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        try:
            # Convert questionnaire data to DataFrame
            df = pd.DataFrame([questionnaire_data])
            
            # Preprocess data
            X, _ = self.preprocess_data(df, is_training=False)
            
            # Make prediction
            if self.ensemble_models:
                ensemble = EnsembleClassifier(self.ensemble_models, weights=getattr(self, 'ensemble_weights', None), device=self.device)
                proba = ensemble.predict_proba(X)
                dry_eye_prob = float(proba[0, 1])
                confidence = float(np.max(proba[0]))
            else:
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    outputs = self.model(X_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    dry_eye_prob = probabilities[0, 1].item()
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
        
        payload = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_selector': self.feature_selector,
            'ensemble_weights': getattr(self, 'ensemble_weights', None),
        }
        
        # Save neural network if available
        if self.model is not None and isinstance(self.model, nn.Module):
            payload['nn_state_dict'] = self.model.state_dict()
            payload['nn_architecture'] = {
                'input_size': self.model.input_size,
                'hidden_sizes': [layer.out_features for layer in self.model.network if isinstance(layer, nn.Linear)],
                'num_classes': self.model.output_layer.out_features
            }
        
        # Save sklearn models if available
        if hasattr(self, 'rf_model') and self.rf_model is not None:
            payload['rf_model'] = self.rf_model
        if hasattr(self, 'gb_model') and self.gb_model is not None:
            payload['gb_model'] = self.gb_model
        if hasattr(self, 'et_model') and self.et_model is not None:
            payload['et_model'] = self.et_model
        
        torch.save(payload, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Reconstruct neural network (optional if present)
        if 'nn_architecture' in checkpoint and 'nn_state_dict' in checkpoint:
            arch = checkpoint['nn_architecture']
            self.model = AdvancedDryEyeTextClassifier(
                input_size=arch['input_size'],
                hidden_sizes=arch['hidden_sizes'],
                num_classes=arch['num_classes']
            )
            self.model.load_state_dict(checkpoint['nn_state_dict'])
            self.model.to(self.device)
        else:
            self.model = None
        
        # Load sklearn models if present
        self.rf_model = checkpoint.get('rf_model', None)
        self.gb_model = checkpoint.get('gb_model', None)
        self.et_model = checkpoint.get('et_model', None)
        
        # Load preprocessors
        self.scaler = checkpoint.get('scaler', self.scaler)
        self.label_encoders = checkpoint.get('label_encoders', {})
        self.feature_names = checkpoint.get('feature_names', self.feature_names)
        self.feature_selector = checkpoint.get('feature_selector', None)
        self.ensemble_weights = checkpoint.get('ensemble_weights', None)
        
        # Recompose ensemble if possible
        self.ensemble_models = []
        if self.model is not None:
            self.ensemble_models.append(self.model)
        if self.rf_model is not None:
            self.ensemble_models.append(self.rf_model)
        if self.gb_model is not None:
            self.ensemble_models.append(self.gb_model)
        if self.et_model is not None:
            self.ensemble_models.append(self.et_model)
        
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
    predictor = AdvancedDryEyeTextPredictor()
    
    # Train the model (uncomment to train)
    dataset_path = "datasets/eyes/Dry_Eye_Dataset.csv"
    if os.path.exists(dataset_path):
        predictor.train_model(dataset_path, epochs=150)
    
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