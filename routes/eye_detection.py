from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import sys
import traceback

# Import your models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add debugging for imports
try:
    from eye_disease_model import EyeDiseasePredictor
    print("✓ EyeDiseasePredictor imported successfully")
except ImportError as e:
    print(f"✗ Error importing EyeDiseasePredictor: {e}")
    EyeDiseasePredictor = None

try:
    from eye_disease_text_model import DryEyeTextPredictor, combine_predictions
    print("✓ DryEyeTextPredictor imported successfully")
except ImportError as e:
    print(f"✗ Error importing DryEyeTextPredictor: {e}")
    print("Full traceback:")
    traceback.print_exc()
    DryEyeTextPredictor = None
    combine_predictions = None

# Create blueprint
eye_detection_bp = Blueprint('eye_detection', __name__)

# Initialize predictors
image_predictor = None
text_predictor = None

def init_predictors():
    """Initialize both image and text predictors with enhanced debugging"""
    global image_predictor, text_predictor
    
    print("Starting predictor initialization...")
    
    # Initialize image predictor
    image_model_path = os.path.join('models', 'best_eye_model.pth')
    print(f"Looking for image model at: {os.path.abspath(image_model_path)}")
    print(f"Image model exists: {os.path.exists(image_model_path)}")
    
    try:
        if EyeDiseasePredictor:
            image_predictor = EyeDiseasePredictor(image_model_path if os.path.exists(image_model_path) else None)
            print("✓ Image predictor initialized successfully")
        else:
            print("✗ Image predictor class not available")
    except Exception as e:
        print(f"✗ Error initializing image predictor: {e}")
        traceback.print_exc()
        image_predictor = None
    
    # Initialize text predictor
    text_model_path = os.path.join('models', 'best_text_model.pth')
    print(f"Looking for text model at: {os.path.abspath(text_model_path)}")
    print(f"Text model exists: {os.path.exists(text_model_path)}")
    
    try:
        if DryEyeTextPredictor:
            text_predictor = DryEyeTextPredictor(text_model_path if os.path.exists(text_model_path) else None)
            print("✓ Text predictor initialized successfully")
        else:
            print("✗ Text predictor class not available")
    except Exception as e:
        print(f"✗ Error initializing text predictor: {e}")
        traceback.print_exc()
        text_predictor = None

    # Summary
    print(f"Initialization complete. Image predictor: {'Available' if image_predictor else 'Not available'}")
    print(f"Text predictor: {'Available' if text_predictor else 'Not available'}")

# Initialize predictors when module loads
init_predictors()

# Create a simple fallback text predictor if the main one fails
class FallbackTextPredictor:
    """Fallback text predictor that uses simple heuristics"""
    
    def __init__(self):
        print("Using fallback text predictor")
    
    def predict_from_questionnaire(self, questionnaire_data):
        """Simple heuristic-based prediction"""
        try:
            risk_score = 0
            risk_factors = []
            
            # Screen time risk
            screen_time = float(questionnaire_data.get('Average_screen_time', 0))
            if screen_time > 8:
                risk_score += 0.3
                risk_factors.append({
                    'factor': 'High Screen Time',
                    'value': f'{screen_time} hours/day',
                    'impact': 'high'
                })
            elif screen_time > 6:
                risk_score += 0.2
                risk_factors.append({
                    'factor': 'Moderate Screen Time',
                    'value': f'{screen_time} hours/day',
                    'impact': 'medium'
                })
            
            # Sleep quality risk
            sleep_quality = int(questionnaire_data.get('Sleep_quality', 10))
            if sleep_quality < 5:
                risk_score += 0.25
                risk_factors.append({
                    'factor': 'Poor Sleep Quality',
                    'value': f'{sleep_quality}/10',
                    'impact': 'high'
                })
            elif sleep_quality < 7:
                risk_score += 0.15
                risk_factors.append({
                    'factor': 'Moderate Sleep Quality',
                    'value': f'{sleep_quality}/10',
                    'impact': 'medium'
                })
            
            # Stress level risk
            stress_level = int(questionnaire_data.get('Stress_level', 1))
            if stress_level > 7:
                risk_score += 0.2
                risk_factors.append({
                    'factor': 'High Stress Level',
                    'value': f'{stress_level}/10',
                    'impact': 'high'
                })
            elif stress_level > 5:
                risk_score += 0.1
                risk_factors.append({
                    'factor': 'Moderate Stress Level',
                    'value': f'{stress_level}/10',
                    'impact': 'medium'
                })
            
            # Age risk
            age = int(questionnaire_data.get('Age', 0))
            if age > 50:
                risk_score += 0.15
                risk_factors.append({
                    'factor': 'Age Factor',
                    'value': f'{age} years',
                    'impact': 'medium'
                })
            
            # Blue light filter
            if questionnaire_data.get('Blue_light_filter', 'Y') == 'N':
                risk_score += 0.1
                risk_factors.append({
                    'factor': 'No Blue Light Filter',
                    'value': 'Not using blue light protection',
                    'impact': 'medium'
                })
            
            # Current symptoms
            symptoms = ['Discomfort_eye_strain', 'Redness_in_eye', 'Itchiness_irritation_in_eye']
            symptom_count = sum(1 for symptom in symptoms if questionnaire_data.get(symptom) == 'Y')
            
            if symptom_count >= 2:
                risk_score += 0.4
                risk_factors.append({
                    'factor': 'Multiple Eye Symptoms',
                    'value': f'{symptom_count} symptoms present',
                    'impact': 'high'
                })
            elif symptom_count == 1:
                risk_score += 0.2
                risk_factors.append({
                    'factor': 'Eye Symptoms',
                    'value': f'{symptom_count} symptom present',
                    'impact': 'medium'
                })
            
            # Cap the risk score at 1.0
            risk_probability = min(risk_score, 1.0)
            confidence = 0.7  # Moderate confidence for heuristic approach
            
            return risk_probability, confidence, risk_factors
            
        except Exception as e:
            print(f"Error in fallback predictor: {e}")
            return 0.1, 0.5, []  # Very conservative fallback

def simple_combine_predictions(image_prob, text_prob, image_weight, text_weight):
    """Simple combination function if the original is not available"""
    return (image_prob * image_weight) + (text_prob * text_weight)

@eye_detection_bp.route('/analyze-eye-image', methods=['POST'])
@login_required
def analyze_eye_image():
    """Analyze uploaded eye image for dry eye disease (Step 1)"""
    try:
        if image_predictor is None:
            return jsonify({
                'success': False,
                'error': 'Eye disease detection model not available. Please check if the model file exists.'
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
            prediction, confidence, is_valid_eye = image_predictor.predict(file_path)
            
            if not is_valid_eye:
                return jsonify({
                    'success': False,
                    'error': 'The uploaded image does not appear to contain a valid eye. Please upload a clear eye image.',
                    'is_valid_eye': False
                })
            
            # Store image analysis result in session for later combination
            image_result = {
                'prediction': prediction,
                'confidence': confidence,
                'dry_eye_probability': 1.0 if prediction == 'Dry Eyes' else 0.0,
                'filename': unique_filename
            }
            
            # Prepare response - only basic info, no recommendations yet
            result = {
                'success': True,
                'is_valid_eye': True,
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'message': 'Image analysis completed. Please fill out the questionnaire for comprehensive analysis.',
                'next_step': 'questionnaire'
            }
            
            return jsonify(result)
        
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error in eye image analysis: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred during image analysis. Please try again.'
        }), 500

@eye_detection_bp.route('/submit-questionnaire', methods=['POST'])
@login_required
def submit_questionnaire():
    """Process questionnaire and provide combined analysis (Step 2)"""
    try:
        # Use main predictor or fallback
        active_text_predictor = text_predictor
        active_combine_function = combine_predictions
        
        if text_predictor is None:
            print("Main text predictor not available, using fallback")
            active_text_predictor = FallbackTextPredictor()
            active_combine_function = simple_combine_predictions
        
        # Get questionnaire data
        questionnaire_data = request.get_json()
        if not questionnaire_data:
            return jsonify({
                'success': False,
                'error': 'No questionnaire data provided'
            }), 400
        
        # Get image analysis data (if available)
        image_data = questionnaire_data.pop('image_analysis', {})
        image_probability = image_data.get('dry_eye_probability', 0.0)
        image_confidence = image_data.get('confidence', 0.0)
        
        # Validate required fields
        required_fields = ['Gender', 'Age', 'Average_screen_time', 'Sleep_quality', 'Stress_level']
        missing_fields = [field for field in required_fields if field not in questionnaire_data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Get text prediction
        text_prob, text_confidence, risk_factors = active_text_predictor.predict_from_questionnaire(questionnaire_data)
        
        # Combine predictions (20% image, 80% text)
        if image_probability > 0:
            combined_probability = active_combine_function(image_probability, text_prob, 0.2, 0.8)
            combined_confidence = (image_confidence * 0.2) + (text_confidence * 0.8)
            analysis_type = "Combined Image + Questionnaire Analysis"
        else:
            combined_probability = text_prob
            combined_confidence = text_confidence
            analysis_type = "Questionnaire Analysis Only"
        
        # Determine risk level
        risk_level = get_risk_level(combined_probability)
        
        # Get comprehensive recommendations
        recommendations = get_comprehensive_recommendations(
            combined_probability, risk_factors, questionnaire_data
        )
        
        # Calculate risk score
        risk_score = calculate_risk_score(combined_probability, risk_factors)
        
        # Prepare final result
        result = {
            'success': True,
            'analysis_type': analysis_type,
            'combined_analysis': {
                'dry_eye_probability': round(combined_probability, 3),
                'confidence': round(combined_confidence, 3),
                'risk_level': risk_level,
                'risk_score': risk_score,
                'has_dry_eyes': combined_probability > 0.5
            },
            'individual_predictions': {
                'image_analysis': {
                    'probability': round(image_probability, 3),
                    'confidence': round(image_confidence, 3),
                    'weight': 0.2
                } if image_probability > 0 else None,
                'text_analysis': {
                    'probability': round(text_prob, 3),
                    'confidence': round(text_confidence, 3),
                    'weight': 0.8
                }
            },
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
        
        # Save comprehensive analysis to database
        save_comprehensive_analysis(current_user.id, questionnaire_data, result)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in questionnaire processing: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred during analysis. Please try again.'
        }), 500

# Route to check model status
@eye_detection_bp.route('/model-status')
@login_required
def check_model_status():
    """Check the status of both models"""
    return jsonify({
        'success': True,
        'models': {
            'image_predictor': {
                'available': image_predictor is not None,
                'class_imported': EyeDiseasePredictor is not None
            },
            'text_predictor': {
                'available': text_predictor is not None,
                'class_imported': DryEyeTextPredictor is not None
            }
        }
    })

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Moderate"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"

def calculate_risk_score(probability, risk_factors):
    """Calculate a comprehensive risk score (0-100)"""
    base_score = probability * 70  # 70% from prediction
    
    # Add risk factor impact
    risk_factor_score = 0
    high_impact_count = sum(1 for rf in risk_factors if rf['impact'] == 'high')
    medium_impact_count = sum(1 for rf in risk_factors if rf['impact'] == 'medium')
    
    risk_factor_score = (high_impact_count * 5) + (medium_impact_count * 2)
    risk_factor_score = min(risk_factor_score, 30)  # Cap at 30%
    
    total_score = min(base_score + risk_factor_score, 100)
    return round(total_score, 1)

def get_comprehensive_recommendations(probability, risk_factors, questionnaire_data):
    """Generate comprehensive recommendations based on all analysis"""
    recommendations = {
        'immediate_actions': [],
        'lifestyle_changes': [],
        'medical_advice': [],
        'monitoring': []
    }
    
    # Risk-based recommendations
    if probability >= 0.7:
        recommendations['medical_advice'].extend([
            "Schedule an appointment with an ophthalmologist within 1-2 weeks",
            "Consider comprehensive eye examination including tear film assessment",
            "Discuss prescription eye drops or treatments with your doctor"
        ])
        recommendations['immediate_actions'].extend([
            "Use preservative-free artificial tears every 2-3 hours",
            "Apply warm compresses to eyes twice daily",
            "Avoid air conditioning and heating vents directed at face"
        ])
    elif probability >= 0.4:
        recommendations['medical_advice'].append(
            "Consider consulting an eye care professional for evaluation"
        )
        recommendations['immediate_actions'].extend([
            "Use artificial tears when experiencing discomfort",
            "Take regular breaks from screen time (20-20-20 rule)"
        ])
    
    # Lifestyle recommendations based on risk factors
    screen_time = questionnaire_data.get('Average_screen_time', 0)
    if float(screen_time) > 8:
        recommendations['lifestyle_changes'].extend([
            f"Reduce daily screen time from {screen_time} hours to under 8 hours",
            "Implement strict screen time limits, especially before bedtime",
            "Use blue light filtering glasses or software"
        ])
    
    if questionnaire_data.get('Blue_light_filter') == 'N':
        recommendations['lifestyle_changes'].append(
            "Enable blue light filters on all devices"
        )
    
    stress_level = questionnaire_data.get('Stress_level', 0)
    if int(stress_level) > 6:
        recommendations['lifestyle_changes'].extend([
            "Practice stress reduction techniques (meditation, yoga, deep breathing)",
            "Consider stress management counseling or resources"
        ])
    
    sleep_quality = questionnaire_data.get('Sleep_quality', 10)
    if int(sleep_quality) < 6:
        recommendations['lifestyle_changes'].extend([
            "Improve sleep hygiene practices",
            "Aim for 7-9 hours of quality sleep per night",
            "Avoid screens 1 hour before bedtime"
        ])
    
    # Monitoring recommendations
    recommendations['monitoring'].extend([
        "Track daily symptoms and triggers",
        "Monitor screen time and take regular breaks",
        "Schedule regular eye check-ups (annually or bi-annually)"
    ])
    
    # Environmental modifications
    recommendations['lifestyle_changes'].extend([
        "Use a humidifier in dry environments",
        "Ensure proper lighting when reading or working",
        "Stay well-hydrated throughout the day"
    ])
    
    return recommendations

def save_comprehensive_analysis(user_id, questionnaire_data, analysis_result):
    """Save comprehensive analysis results to database"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Prepare data for user_eye_health_data table
        health_data = {
            'user_id': user_id,
            'gender': questionnaire_data.get('Gender', 'Other'),
            'age': questionnaire_data.get('Age', 0),
            'sleep_duration': questionnaire_data.get('Sleep_duration', 7.0),
            'sleep_quality': questionnaire_data.get('Sleep_quality', 5),
            'stress_level': questionnaire_data.get('Stress_level', 5),
            'blood_pressure': str(questionnaire_data.get('Blood_pressure', '120/80')),
            'heart_rate': questionnaire_data.get('Heart_rate', 70),
            'daily_steps': questionnaire_data.get('Daily_steps', 5000),
            'physical_activity': questionnaire_data.get('Physical_activity', 3),
            'height': questionnaire_data.get('Height', 170),
            'weight': questionnaire_data.get('Weight', 70),
            'sleep_disorder': questionnaire_data.get('Sleep_disorder', 'N'),
            'wake_up_during_night': questionnaire_data.get('Wake_up_during_night', 'N'),
            'feel_sleepy_during_day': questionnaire_data.get('Feel_sleepy_during_day', 'N'),
            'caffeine_consumption': questionnaire_data.get('Caffeine_consumption', 'N'),
            'alcohol_consumption': questionnaire_data.get('Alcohol_consumption', 'N'),
            'smoking': questionnaire_data.get('Smoking', 'N'),
            'medical_issue': questionnaire_data.get('Medical_issue', 'N'),
            'ongoing_medication': questionnaire_data.get('Ongoing_medication', 'N'),
            'smart_device_before_bed': questionnaire_data.get('Smart_device_before_bed', 'N'),
            'average_screen_time': questionnaire_data.get('Average_screen_time', 6.0),
            'blue_light_filter': questionnaire_data.get('Blue_light_filter', 'N'),
            'discomfort_eye_strain': questionnaire_data.get('Discomfort_eye_strain', 'N'),
            'redness_in_eye': questionnaire_data.get('Redness_in_eye', 'N'),
            'itchiness_irritation_in_eye': questionnaire_data.get('Itchiness_irritation_in_eye', 'N'),
            'dry_eye_disease': 'Y' if analysis_result['combined_analysis']['has_dry_eyes'] else 'N',
            'risk_score': analysis_result['combined_analysis']['risk_score'],
            'risk_factors': json.dumps(analysis_result['risk_factors'])
        }
        
        # Insert or update user health data
        cursor.execute("""
            INSERT INTO user_eye_health_data (
                user_id, gender, age, sleep_duration, sleep_quality, stress_level,
                blood_pressure, heart_rate, daily_steps, physical_activity, height, weight,
                sleep_disorder, wake_up_during_night, feel_sleepy_during_day, caffeine_consumption,
                alcohol_consumption, smoking, medical_issue, ongoing_medication, smart_device_before_bed,
                average_screen_time, blue_light_filter, discomfort_eye_strain, redness_in_eye,
                itchiness_irritation_in_eye, dry_eye_disease, risk_score, risk_factors
            ) VALUES (
                %(user_id)s, %(gender)s, %(age)s, %(sleep_duration)s, %(sleep_quality)s, %(stress_level)s,
                %(blood_pressure)s, %(heart_rate)s, %(daily_steps)s, %(physical_activity)s, %(height)s, %(weight)s,
                %(sleep_disorder)s, %(wake_up_during_night)s, %(feel_sleepy_during_day)s, %(caffeine_consumption)s,
                %(alcohol_consumption)s, %(smoking)s, %(medical_issue)s, %(ongoing_medication)s, %(smart_device_before_bed)s,
                %(average_screen_time)s, %(blue_light_filter)s, %(discomfort_eye_strain)s, %(redness_in_eye)s,
                %(itchiness_irritation_in_eye)s, %(dry_eye_disease)s, %(risk_score)s, %(risk_factors)s
            )
            ON DUPLICATE KEY UPDATE
                gender=VALUES(gender), age=VALUES(age), sleep_duration=VALUES(sleep_duration),
                sleep_quality=VALUES(sleep_quality), stress_level=VALUES(stress_level),
                blood_pressure=VALUES(blood_pressure), heart_rate=VALUES(heart_rate),
                daily_steps=VALUES(daily_steps), physical_activity=VALUES(physical_activity),
                height=VALUES(height), weight=VALUES(weight), sleep_disorder=VALUES(sleep_disorder),
                wake_up_during_night=VALUES(wake_up_during_night), feel_sleepy_during_day=VALUES(feel_sleepy_during_day),
                caffeine_consumption=VALUES(caffeine_consumption), alcohol_consumption=VALUES(alcohol_consumption),
                smoking=VALUES(smoking), medical_issue=VALUES(medical_issue), ongoing_medication=VALUES(ongoing_medication),
                smart_device_before_bed=VALUES(smart_device_before_bed), average_screen_time=VALUES(average_screen_time),
                blue_light_filter=VALUES(blue_light_filter), discomfort_eye_strain=VALUES(discomfort_eye_strain),
                redness_in_eye=VALUES(redness_in_eye), itchiness_irritation_in_eye=VALUES(itchiness_irritation_in_eye),
                dry_eye_disease=VALUES(dry_eye_disease), risk_score=VALUES(risk_score),
                risk_factors=VALUES(risk_factors), updated_at=CURRENT_TIMESTAMP
        """, health_data)
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error saving comprehensive analysis: {e}")
        traceback.print_exc()

@eye_detection_bp.route('/eye-analysis-history')
@login_required
def get_eye_analysis_history():
    """Get user's comprehensive eye analysis history"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT risk_score, dry_eye_disease, risk_factors, created_at, updated_at
            FROM user_eye_health_data 
            WHERE user_id = %s 
            ORDER BY updated_at DESC 
            LIMIT 10
        """, (current_user.id,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        history = []
        for result in results:
            history.append({
                'risk_score': float(result[0]) if result[0] else 0.0,
                'prediction': 'Dry Eyes Detected' if result[1] == 'Y' else 'No Dry Eyes',
                'risk_factors': json.loads(result[2]) if result[2] else [],
                'created_date': result[3].strftime('%Y-%m-%d %H:%M:%S') if result[3] else '',
                'updated_date': result[4].strftime('%Y-%m-%d %H:%M:%S') if result[4] else ''
            })
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except Exception as e:
        print(f"Error fetching analysis history: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to fetch analysis history'
        }), 500