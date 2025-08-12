#!/usr/bin/env python3
"""
Simple training script for improved text-based dry eye disease detection model
Usage: python3 train_simple_improved.py
"""

import os
from eye_disease_text_model_simple_improved import ImprovedDryEyeTextPredictor

def main():
    print("🚀 Improved Text-based Dry Eye Disease Detection Model Training")
    print("="*65)
    
    # Check if dataset exists
    dataset_path = "datasets/eyes/Dry_Eye_Dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"Dataset path: {dataset_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 65)
    
    try:
        # Create models directory
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        # Initialize improved predictor
        predictor = ImprovedDryEyeTextPredictor()
        
        # Train the improved model
        print(f"\n🚀 Starting improved model training...")
        fold_scores, ensemble_acc = predictor.train_ensemble_model(
            csv_path=dataset_path, 
            epochs=150,
            batch_size=64,
            learning_rate=0.0005,
            n_folds=5
        )
        
        # Save final model
        model_save_path = os.path.join(model_dir, "best_improved_text_model.pth")
        predictor.save_model(model_save_path)
        
        print("\n" + "="*70)
        print("🎉 IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Model saved to: {model_save_path}")
        print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_scores]}")
        print(f"Average fold accuracy: {sum(fold_scores)/len(fold_scores):.2f}%")
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
        print(f"\n🧪 Testing improved model with sample data...")
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
        
        print(f"\n🎯 Improved model is ready for use in the application!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()