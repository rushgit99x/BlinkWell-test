#!/usr/bin/env python3
"""
Test script to verify validation adjustments are working better
"""

import os
import sys
from eye_disease_model import EyeDiseasePredictor

def test_validation_on_directory(test_dir):
    """Test validation on a directory of images"""
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist")
        return
    
    # Initialize predictor
    predictor = EyeDiseasePredictor()
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(image_extensions)]
    
    print(f"Testing validation on {len(image_files)} images...")
    print("="*60)
    
    results = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(test_dir, img_file)
        print(f"\n{i+1}/{len(image_files)}: {img_file}")
        
        try:
            is_eye, confidence, details = predictor.validator.is_eye_image(img_path)
            
            status = "✅ ACCEPTED" if is_eye else "❌ REJECTED"
            print(f"  Status: {status}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Eye Count: {details['eye_count']}")
            print(f"  Has Face: {details['has_face']}")
            print(f"  Quality: {details['quality_check']}")
            
            if 'fallback_used' in details and details['fallback_used']:
                print(f"  ⚠️  Fallback validation used")
            
            results.append({
                'filename': img_file,
                'is_eye': is_eye,
                'confidence': confidence,
                'details': details
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'filename': img_file,
                'is_eye': False,
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total = len(results)
    accepted = sum(1 for r in results if r['is_eye'])
    rejected = total - accepted
    
    print(f"Total images: {total}")
    print(f"Accepted: {accepted} ({accepted/total*100:.1f}%)")
    print(f"Rejected: {rejected} ({rejected/total*100:.1f}%)")
    
    if accepted > 0:
        avg_confidence = sum(r['confidence'] for r in results if r['is_eye']) / accepted
        print(f"Average confidence (accepted): {avg_confidence:.3f}")
    
    # Show rejected images with high confidence (potential false negatives)
    high_conf_rejected = [r for r in results if not r['is_eye'] and r['confidence'] > 0.2]
    if high_conf_rejected:
        print(f"\n⚠️  High confidence rejections (potential false negatives):")
        for item in high_conf_rejected[:5]:
            print(f"  - {item['filename']}: {item['confidence']:.3f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_validation_adjustments.py <test_directory>")
        print("Example: python test_validation_adjustments.py ./test_images")
        return
    
    test_dir = sys.argv[1]
    test_validation_on_directory(test_dir)

if __name__ == "__main__":
    main()