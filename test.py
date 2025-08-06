def test_text_model():
    """Test function to verify text model works"""
    try:
        from eye_disease_text_model import DryEyeTextPredictor
        predictor = DryEyeTextPredictor('best_text_model.pth')
        
        # Test data
        test_data = {
            'Gender': 'M',
            'Age': 30,
            'Average_screen_time': 8.0,
            'Sleep_quality': 6,
            'Stress_level': 7
        }
        
        result = predictor.predict_from_questionnaire(test_data)
        print(f"Test successful: {result}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

# Run this test
test_text_model()