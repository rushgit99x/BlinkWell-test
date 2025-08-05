# Add this to your existing routes/main.py or create a new file routes/eye_detection.py

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import sys

# Import your eye disease predictor (make sure the model file is in your project)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eye_disease_model import EyeDiseasePredictor

# Create blueprint
eye_detection_bp = Blueprint('eye_detection', __name__)

# Initialize the predictor (load pre-trained model)
predictor = None

def init_predictor():
    """Initialize the eye disease predictor with pre-trained model"""
    global predictor
    model_path = os.path.join('models', 'best_eye_model.pth')
    
    try:
        predictor = EyeDiseasePredictor(model_path if os.path.exists(model_path) else None)
        print("Eye disease predictor initialized successfully")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        predictor = None

# Initialize predictor when module loads
init_predictor()

@eye_detection_bp.route('/analyze-eye-image', methods=['POST'])
@login_required
def analyze_eye_image():
    """Analyze uploaded eye image for dry eye disease"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Eye disease detection model not available'
            }), 500
        
        # Check if file was uploaded
        if 'eye_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['eye_image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files only.'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        
        # Ensure temp directory exists
        temp_dir = os.path.join('temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, unique_filename)
        file.save(file_path)
        
        try:
            # Analyze the image
            prediction, confidence, is_valid_eye = predictor.predict(file_path)
            
            if not is_valid_eye:
                return jsonify({
                    'success': False,
                    'error': 'The uploaded image does not appear to contain a valid eye. Please upload a clear eye image.',
                    'is_valid_eye': False
                })
            
            # Prepare response
            result = {
                'success': True,
                'is_valid_eye': True,
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'analysis': {
                    'has_dry_eyes': prediction == 'Dry Eyes',
                    'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
                    'recommendation': get_recommendation(prediction, confidence)
                }
            }
            
            # Log the analysis (optional)
            log_analysis_result(current_user.id, unique_filename, result)
            
            return jsonify(result)
        
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error in eye image analysis: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during image analysis. Please try again.'
        }), 500

def get_recommendation(prediction, confidence):
    """Generate recommendations based on prediction results"""
    if prediction == 'Dry Eyes':
        if confidence > 0.8:
            return {
                'severity': 'High confidence detection of dry eyes',
                'recommendations': [
                    'Consult an ophthalmologist for professional evaluation',
                    'Use preservative-free artificial tears regularly',
                    'Take frequent breaks from screen time (20-20-20 rule)',
                    'Consider using a humidifier in dry environments',
                    'Avoid direct air conditioning or heating vents'
                ],
                'urgency': 'high'
            }
        else:
            return {
                'severity': 'Possible indicators of dry eyes detected',
                'recommendations': [
                    'Monitor symptoms and consider eye care consultation',
                    'Use artificial tears if experiencing discomfort',
                    'Practice good screen hygiene',
                    'Stay hydrated and maintain eye moisture'
                ],
                'urgency': 'medium'
            }
    else:
        return {
            'severity': 'Eyes appear healthy',
            'recommendations': [
                'Continue maintaining good eye health habits',
                'Regular eye check-ups as preventive care',
                'Protect eyes from UV radiation',
                'Maintain proper screen distance and lighting'
            ],
            'urgency': 'low'
        }

def log_analysis_result(user_id, filename, result):
    """Log analysis results to database (optional)"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eye_analysis_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                filename VARCHAR(255),
                prediction VARCHAR(100),
                confidence DECIMAL(5,2),
                analysis_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX(user_id)
            )
        """)
        
        # Insert log entry
        cursor.execute("""
            INSERT INTO eye_analysis_logs (user_id, filename, prediction, confidence, analysis_data)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            user_id,
            filename,
            result.get('prediction'),
            result.get('confidence'),
            json.dumps(result)
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error logging analysis result: {e}")

@eye_detection_bp.route('/eye-analysis-history')
@login_required
def get_eye_analysis_history():
    """Get user's eye analysis history"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT prediction, confidence, analysis_data, created_at 
            FROM eye_analysis_logs 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 10
        """, (current_user.id,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        history = []
        for result in results:
            history.append({
                'prediction': result[0],
                'confidence': float(result[1]),
                'analysis_data': json.loads(result[2]) if result[2] else {},
                'date': result[3].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except Exception as e:
        print(f"Error fetching analysis history: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch analysis history'
        }), 500

