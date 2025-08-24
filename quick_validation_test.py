#!/usr/bin/env python3
"""
Quick validation test script for testing the improved eye validation system
Usage: python quick_validation_test.py --image_path "path/to/image.jpg"
"""

import argparse
import os
from eye_disease_model import EyeDiseasePredictor

def test_single_image(image_path, model_path=None):
    """Test validation on a single image"""
    print(f"Testing validation on: {image_path}")
    print("-" * 50)
    
    # Initialize predictor
    predictor = EyeDiseasePredictor(model_path=model_path)
    
    # Test validation
    is_eye, confidence, validation_details = predictor.validator.is_eye_image(image_path)
    
    print(f"Validation Result: {'✅ EYE DETECTED' if is_eye else '❌ NOT AN EYE'}")
    print(f"Confidence Score: {confidence:.3f}")
    print(f"Minimum Required: 0.6")
    
    print(f"\nValidation Details:")
    print(f"  Eye Count: {validation_details['eye_count']}")
    print(f"  Face Context: {'Yes' if validation_details['has_face'] else 'No'}")
    print(f"  Quality Check: {validation_details['quality_check']}")
    
    if 'confidence_breakdown' in validation_details:
        breakdown = validation_details['confidence_breakdown']
        print(f"  Base Confidence: {breakdown['base_confidence']:.3f}")
        print(f"  Face Bonus: {breakdown['face_bonus']:.3f}")
        print(f"  Final Confidence: {breakdown['final_confidence']:.3f}")
    
    # If it's a valid eye image and we have a model, make prediction
    if is_eye and model_path and os.path.exists(model_path):
        print(f"\n" + "="*50)
        print("MAKING PREDICTION")
        print("="*50)
        
        prediction, pred_confidence, is_valid, pred_details = predictor.predict(image_path)
        
        if is_valid:
            print(f"Prediction: {prediction}")
            print(f"Prediction Confidence: {pred_confidence:.3f}")
        else:
            print("Prediction failed")
    
    return is_eye, confidence, validation_details

def main():
    parser = argparse.ArgumentParser(description='Quick validation test for eye detection')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image file to test')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (optional, for full prediction)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return
    
    try:
        test_single_image(args.image_path, args.model_path)
    except Exception as e:
        print(f"Error during validation test: {e}")

if __name__ == "__main__":
    main()