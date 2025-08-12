#!/usr/bin/env python3
"""
Enhanced training script for improved text-based dry eye disease detection model
Usage: python train_improved_text_model.py --dataset_path "datasets/eyes/Dry_Eye_Dataset.csv"
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eye_disease_text_model_improved import EnsembleDryEyePredictor
import torch
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def plot_training_results(fold_scores, ensemble_acc, save_path='improved_text_training_results.png'):
    """Plot and save enhanced training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot individual fold accuracies
    ax1.bar(range(1, len(fold_scores) + 1), fold_scores, color='skyblue', alpha=0.7)
    ax1.axhline(y=np.mean(fold_scores), color='red', linestyle='--', label=f'Mean: {np.mean(fold_scores):.2f}%')
    ax1.set_title('Individual Fold Accuracies', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fold Number', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot fold accuracy distribution
    ax2.hist(fold_scores, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(fold_scores), color='red', linestyle='--', label=f'Mean: {np.mean(fold_scores):.2f}%')
    ax2.axvline(x=np.median(fold_scores), color='orange', linestyle='--', label=f'Median: {np.median(fold_scores):.2f}%')
    ax2.set_title('Fold Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Plot ensemble vs individual performance
    ax3.bar(['Individual Folds', 'Ensemble'], [np.mean(fold_scores), ensemble_acc], 
            color=['lightcoral', 'gold'], alpha=0.7)
    ax3.set_title('Ensemble vs Individual Performance', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for i, v in enumerate([np.mean(fold_scores), ensemble_acc]):
        ax3.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot improvement over baseline
    baseline_acc = 65.17  # Your current accuracy
    improvement = ensemble_acc - baseline_acc
    ax4.bar(['Current Model', 'Improved Model'], [baseline_acc, ensemble_acc], 
            color=['lightgray', 'lightblue'], alpha=0.7)
    ax4.set_title(f'Performance Improvement\n(+{improvement:.2f}%)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for i, v in enumerate([baseline_acc, ensemble_acc]):
        ax4.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Enhanced training results plot saved to {save_path}")

def plot_roc_curves(predictor, X_val, y_val, save_path='roc_curves.png'):
    """Plot ROC curves for individual models and ensemble"""
    if not predictor.models:
        print("No models available for ROC analysis")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Individual model ROC curves
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, model in enumerate(predictor.models):
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(predictor.device)
            outputs = model(X_val_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            y_scores = probabilities[:, 1].cpu().numpy()
            
            fpr, tpr, _ = roc_curve(y_val, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], alpha=0.7, 
                    label=f'Model {i+1} (AUC = {roc_auc:.3f})')
    
    # Ensemble ROC curve
    all_predictions = []
    for model in predictor.models:
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(predictor.device)
            outputs = model(X_val_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            all_predictions.append(probabilities.cpu().numpy())
    
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_scores = ensemble_probs[:, 1]
    
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_val, ensemble_scores)
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
    
    plt.plot(fpr_ensemble, tpr_ensemble, color='black', linewidth=3,
            label=f'Ensemble (AUC = {roc_auc_ensemble:.3f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Individual Models vs Ensemble', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curves plot saved to {save_path}")

def analyze_dataset_improvements(dataset_path):
    """Analyze the dataset and show potential improvements"""
    df = pd.read_csv(dataset_path)
    
    print("\n" + "="*70)
    print("ENHANCED DATASET ANALYSIS")
    print("="*70)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    
    # Target distribution
    if 'Dry_Eye_Disease' in df.columns:
        target_dist = df['Dry_Eye_Disease'].value_counts()
        print(f"\nTarget distribution:")
        for target, count in target_dist.items():
            print(f"  {target}: {count} ({count/len(df)*100:.1f}%)")
        
        # Check for class imbalance
        imbalance_ratio = target_dist.max() / target_dist.min()
        if imbalance_ratio > 1.5:
            print(f"⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        else:
            print(f"✅ Balanced dataset (ratio: {imbalance_ratio:.2f})")
    
    # Feature analysis
    print(f"\nFeature analysis:")
    
    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric features: {len(numeric_cols)}")
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"  Categorical features: {len(categorical_cols)}")
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    else:
        print(f"\n✅ No missing values found!")
    
    # Feature correlations with target
    if 'Dry_Eye_Disease' in df.columns:
        print(f"\nFeature correlations with target:")
        numeric_df = df.select_dtypes(include=[np.number])
        if 'Dry_Eye_Disease' in numeric_df.columns:
            correlations = numeric_df.corr()['Dry_Eye_Disease'].abs().sort_values(ascending=False)
            top_correlations = correlations.head(10)
            for feature, corr in top_correlations.items():
                if feature != 'Dry_Eye_Disease':
                    print(f"  {feature}: {corr:.3f}")
    
    print("="*70)
    
    return df

def create_hyperparameter_grid():
    """Create a grid of hyperparameters to try"""
    return {
        'learning_rates': [0.001, 0.0005, 0.0001],
        'batch_sizes': [32, 64, 128],
        'hidden_sizes': [
            [128, 64, 32],
            [256, 128, 64, 32],
            [512, 256, 128, 64, 32]
        ],
        'dropout_rates': [0.3, 0.4, 0.5],
        'epochs': [100, 150, 200]
    }

def hyperparameter_search(predictor, dataset_path, n_trials=5):
    """Perform hyperparameter search"""
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60)
    
    hyperparams = create_hyperparameter_grid()
    best_score = 0
    best_params = None
    results = []
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        # Randomly sample hyperparameters
        import random
        lr = random.choice(hyperparams['learning_rates'])
        batch_size = random.choice(hyperparams['batch_sizes'])
        hidden_size = random.choice(hyperparams['hidden_sizes'])
        dropout = random.choice(hyperparams['dropout_rates'])
        epochs = random.choice(hyperparams['epochs'])
        
        print(f"Learning rate: {lr}")
        print(f"Batch size: {batch_size}")
        print(f"Hidden sizes: {hidden_size}")
        print(f"Dropout: {dropout}")
        print(f"Epochs: {epochs}")
        
        try:
            # Train with current hyperparameters
            fold_scores, ensemble_acc = predictor.train_ensemble_model(
                csv_path=dataset_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                n_folds=3  # Use fewer folds for faster search
            )
            
            results.append({
                'trial': trial + 1,
                'lr': lr,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'dropout': dropout,
                'epochs': epochs,
                'mean_fold_acc': np.mean(fold_scores),
                'ensemble_acc': ensemble_acc
            })
            
            if ensemble_acc > best_score:
                best_score = ensemble_acc
                best_params = {
                    'lr': lr,
                    'batch_size': batch_size,
                    'hidden_size': hidden_size,
                    'dropout': dropout,
                    'epochs': epochs
                }
                print(f"🎯 New best score: {best_score:.2f}%")
            
        except Exception as e:
            print(f"❌ Trial failed: {e}")
            continue
    
    # Print results summary
    print(f"\n" + "="*60)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*60)
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\nTop 3 configurations:")
        top_results = df_results.nlargest(3, 'ensemble_acc')
        for _, row in top_results.iterrows():
            print(f"  Trial {row['trial']}: {row['ensemble_acc']:.2f}% "
                  f"(lr={row['lr']}, bs={row['batch_size']}, epochs={row['epochs']})")
        
        print(f"\n🎯 Best configuration:")
        print(f"  Learning rate: {best_params['lr']}")
        print(f"  Batch size: {best_params['batch_size']}")
        print(f"  Hidden sizes: {best_params['hidden_size']}")
        print(f"  Dropout: {best_params['dropout']}")
        print(f"  Epochs: {best_params['epochs']}")
        print(f"  Best ensemble accuracy: {best_score:.2f}%")
    
    return best_params, results

def main():
    parser = argparse.ArgumentParser(description='Train improved text-based dry eye disease detection model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to CSV dataset file')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--model_save_path', type=str, default='models/best_improved_text_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--plot_results', action='store_true',
                       help='Generate training results plots')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze dataset without training')
    parser.add_argument('--hyperparameter_search', action='store_true',
                       help='Perform hyperparameter search')
    
    args = parser.parse_args()
    
    print("Enhanced Text-based Dry Eye Disease Detection Model Training")
    print("="*65)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Cross-validation folds: {args.n_folds}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 65)
    
    try:
        # Analyze dataset
        df = analyze_dataset_improvements(args.dataset_path)
        
        if args.analyze_only:
            print("Dataset analysis completed. Exiting without training.")
            return
        
        # Create models directory
        model_dir = os.path.dirname(args.model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        # Initialize enhanced predictor
        predictor = EnsembleDryEyePredictor()
        
        if args.hyperparameter_search:
            # Perform hyperparameter search
            best_params, search_results = hyperparameter_search(predictor, args.dataset_path)
            
            # Use best parameters for final training
            if best_params:
                print(f"\n🎯 Training final model with best hyperparameters...")
                args.learning_rate = best_params['lr']
                args.batch_size = best_params['batch_size']
                args.epochs = best_params['epochs']
        
        # Train the enhanced model
        print(f"\n🚀 Starting enhanced training...")
        fold_scores, ensemble_acc = predictor.train_ensemble_model(
            csv_path=args.dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_folds=args.n_folds
        )
        
        # Save final model
        predictor.save_model(args.model_save_path)
        
        # Plot results if requested
        if args.plot_results:
            plot_training_results(fold_scores, ensemble_acc)
            
            # Create validation set for ROC analysis
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            
            X, _ = predictor.preprocess_data(df, is_training=True)
            y = LabelEncoder().fit_transform(df['Dry_Eye_Disease'].astype(str))
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            plot_roc_curves(predictor, X_val, y_val)
        
        print("\n" + "="*70)
        print("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Model saved to: {args.model_save_path}")
        print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_scores]}")
        print(f"Average fold accuracy: {np.mean(fold_scores):.2f}%")
        print(f"Ensemble validation accuracy: {ensemble_acc:.2f}%")
        
        # Calculate improvement
        baseline_acc = 65.17
        improvement = ensemble_acc - baseline_acc
        print(f"\n📈 Performance Improvement:")
        print(f"  Baseline accuracy: {baseline_acc:.2f}%")
        print(f"  Improved accuracy: {ensemble_acc:.2f}%")
        print(f"  Improvement: +{improvement:.2f}%")
        
        if improvement > 0:
            print(f"  🎯 Target achieved: {'Yes' if ensemble_acc >= 70 else 'No'}")
            if ensemble_acc >= 80:
                print(f"  🌟 Excellent! Exceeded 80% target!")
            elif ensemble_acc >= 70:
                print(f"  ✅ Good! Achieved 70%+ target!")
            else:
                print(f"  📊 Close to target, consider further tuning")
        
        print("="*70)
        
        # Test the trained model
        print(f"\n🧪 Testing enhanced model with sample data...")
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
        
        try:
            prob, confidence, risk_factors = predictor.predict_from_questionnaire(sample_data)
            
            print(f"\n📊 Sample Prediction Results:")
            print(f"  Dry eye probability: {prob:.3f}")
            print(f"  Prediction confidence: {confidence:.3f}")
            print(f"  Risk factors identified: {len(risk_factors)}")
            
            if risk_factors:
                print(f"  Top risk factors:")
                for rf in risk_factors[:3]:
                    print(f"    - {rf['factor']}: {rf['value']} ({rf['impact']} impact)")
        
        except Exception as e:
            print(f"Error during sample prediction: {e}")
        
        print(f"\n🎯 Enhanced model is ready for use in the application!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()