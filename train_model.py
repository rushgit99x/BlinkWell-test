#!/usr/bin/env python3
"""
Training script for dry eye disease detection model
Usage: python train_model.py --dataset_path "BlinkWell/datasets/eyes" --epochs 30
"""

import argparse
import os
import matplotlib.pyplot as plt
from eye_disease_model import EyeDiseasePredictor
import torch

def plot_training_results(train_losses, val_accuracies, save_path='training_results.png'):
    """Plot and save training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training results plot saved to {save_path}")

def validate_dataset_structure(dataset_path):
    """Validate that the dataset has the correct structure"""
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    dry_eyes_path = os.path.join(dataset_path, 'dry_eyes')
    no_dry_eyes_path = os.path.join(dataset_path, 'no_dry_eyes')
    
    if not os.path.exists(dry_eyes_path):
        raise ValueError(f"'dry_eyes' folder not found in {dataset_path}")
    
    if not os.path.exists(no_dry_eyes_path):
        raise ValueError(f"'no_dry_eyes' folder not found in {dataset_path}")
    
    # Count images in each folder
    dry_eyes_count = len([f for f in os.listdir(dry_eyes_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    no_dry_eyes_count = len([f for f in os.listdir(no_dry_eyes_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Dataset validation successful!")
    print(f"Dry eyes images: {dry_eyes_count}")
    print(f"Healthy eyes images: {no_dry_eyes_count}")
    print(f"Total images: {dry_eyes_count + no_dry_eyes_count}")
    
    if dry_eyes_count == 0 or no_dry_eyes_count == 0:
        raise ValueError("One of the categories has no images!")
    
    return dry_eyes_count, no_dry_eyes_count

def validate_model_on_test_images(predictor, test_dir, epoch=None):
    """Validate model on test images and report validation performance"""
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist, skipping validation test")
        return
    
    print(f"\n{'='*50}")
    print(f"VALIDATION TESTING (Epoch {epoch if epoch else 'N/A'})")
    print(f"{'='*50}")
    
    # Test validation system
    validation_results = predictor.test_validation_on_directory(test_dir)
    
    # Calculate validation metrics
    total_images = len(validation_results)
    eye_images = sum(1 for r in validation_results if r['is_eye'])
    avg_confidence = sum(r['confidence'] for r in validation_results) / total_images if total_images > 0 else 0
    
    print(f"\nValidation Summary:")
    print(f"Total test images: {total_images}")
    print(f"Detected as eyes: {eye_images} ({eye_images/total_images*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description='Train dry eye disease detection model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset folder containing dry_eyes and no_dry_eyes subfolders')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--model_save_path', type=str, default='models/best_eye_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--plot_results', action='store_true',
                       help='Generate training results plot')
    parser.add_argument('--test_validation_dir', type=str, default=None,
                       help='Directory with test images for validation testing during training')
    parser.add_argument('--validation_test_interval', type=int, default=5,
                       help='Test validation every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    print("Starting dry eye disease detection model training...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 50)
    
    try:
        # Validate dataset structure
        dry_count, healthy_count = validate_dataset_structure(args.dataset_path)
        
        # Create models directory if it doesn\'t exist
        model_dir = os.path.dirname(args.model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        # Initialize predictor
        predictor = EyeDiseasePredictor()
        
        # Train the model
        train_losses, val_accuracies = predictor.train_model(
            dataset_path=args.dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Save final model
        predictor.save_model(args.model_save_path)
        
        # Plot results if requested
        if args.plot_results:
            plot_training_results(train_losses, val_accuracies)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best model saved to: {args.model_save_path}")
        print(f"Final validation accuracy: {max(val_accuracies):.2f}%")
        print("="*50)
        
        # Test the trained model on a sample
        print("\nTesting model loading...")
        test_predictor = EyeDiseasePredictor(args.model_save_path)
        print("Model loaded successfully and ready for inference!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()

