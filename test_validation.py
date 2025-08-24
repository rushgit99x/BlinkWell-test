#!/usr/bin/env python3
"""
Test script for the improved eye validation system
Usage: python test_validation.py --test_dir "path/to/test/images" --output_file "validation_results.json"
"""

import argparse
import os
import json
from eye_disease_model import EyeDiseasePredictor

def analyze_validation_results(results_file):
    """Analyze validation results and provide insights"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("VALIDATION ANALYSIS REPORT")
    print("="*60)
    
    total_images = len(results)
    eye_images = sum(1 for r in results if r['is_eye'])
    non_eye_images = total_images - eye_images
    
    print(f"Total images tested: {total_images}")
    print(f"Detected as eyes: {eye_images} ({eye_images/total_images*100:.1f}%)")
    print(f"Rejected as non-eyes: {non_eye_images} ({non_eye_images/total_images*100:.1f}%)")
    
    # Analyze confidence distribution
    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    max_confidence = max(confidences)
    min_confidence = min(confidences)
    
    print(f"\nConfidence Statistics:")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Maximum confidence: {max_confidence:.3f}")
    print(f"Minimum confidence: {min_confidence:.3f}")
    
    # Analyze validation details
    face_detections = sum(1 for r in results if r['validation_details'].get('has_face', False))
    quality_issues = [r for r in results if r['validation_details'].get('quality_check', '') != 'Good quality']
    
    print(f"\nValidation Details:")
    print(f"Images with face context: {face_detections} ({face_detections/total_images*100:.1f}%)")
    print(f"Images with quality issues: {len(quality_issues)}")
    
    if quality_issues:
        print("\nQuality issues found:")
        for issue in quality_issues[:10]:  # Show first 10
            print(f"  - {issue['filename']}: {issue['validation_details']['quality_check']}")
    
    # Show high-confidence non-eye detections (potential false positives)
    high_conf_non_eyes = [r for r in results if not r['is_eye'] and r['confidence'] > 0.3]
    if high_conf_non_eyes:
        print(f"\n⚠️  Potential false positives (high confidence non-eyes):")
        for item in high_conf_non_eyes[:5]:  # Show first 5
            print(f"  - {item['filename']}: confidence {item['confidence']:.3f}")
    
    # Show low-confidence eye detections (potential false negatives)
    low_conf_eyes = [r for r in results if r['is_eye'] and r['confidence'] < 0.7]
    if low_conf_eyes:
        print(f"\n⚠️  Low confidence eye detections:")
        for item in low_conf_eyes[:5]:  # Show first 5
            print(f"  - {item['filename']}: confidence {item['confidence']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Test the improved eye validation system')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_file', type=str, default='validation_results.json',
                       help='Output file for validation results (default: validation_results.json)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze existing results file, skip validation testing')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (optional, for full prediction testing)')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        if os.path.exists(args.output_file):
            analyze_validation_results(args.output_file)
        else:
            print(f"Results file {args.output_file} not found. Run validation test first.")
        return
    
    print("Testing improved eye validation system...")
    print(f"Test directory: {args.test_dir}")
    print(f"Output file: {args.output_file}")
    print("-" * 50)
    
    # Initialize predictor
    predictor = EyeDiseasePredictor(model_path=args.model_path)
    
    # Test validation on directory
    results = predictor.test_validation_on_directory(args.test_dir, args.output_file)
    
    # Analyze results
    if os.path.exists(args.output_file):
        analyze_validation_results(args.output_file)
    
    print("\nValidation testing completed!")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()