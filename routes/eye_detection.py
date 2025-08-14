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

try:
    from eye_disease_model import EyeDiseasePredictor
    print("✓ EyeDiseasePredictor imported successfully")
except ImportError as e:
    print(f"✗ Error importing EyeDiseasePredictor: {e}")
    EyeDiseasePredictor = None

try:
    from eye_disease_text_model import AdvancedDryEyeTextPredictor, combine_predictions
    print("✓ AdvancedDryEyeTextPredictor imported successfully")
except ImportError as e:
    print(f"✗ Error importing AdvancedDryEyeTextPredictor: {e}")
    print("Full traceback:")
    traceback.print_exc()
    AdvancedDryEyeTextPredictor = None
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
        if AdvancedDryEyeTextPredictor:
            text_predictor = AdvancedDryEyeTextPredictor(text_model_path if os.path.exists(text_model_path) else None)
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

def clear_all_user_data(user_id):
    """COMPREHENSIVE clearing of all user data - both recommendations and health data"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Start transaction
        conn.begin()
        
        # 1. Clear all recommendations
        cursor.execute("""
            DELETE FROM user_recommendations 
            WHERE user_id = %s
        """, (user_id,))
        recommendations_deleted = cursor.rowcount
        
        # 2. Clear health data 
        cursor.execute("""
            DELETE FROM user_eye_health_data 
            WHERE user_id = %s
        """, (user_id,))
        health_data_deleted = cursor.rowcount
        
        # Commit transaction
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ COMPLETE DATA CLEAR for user {user_id}:")
        print(f"  - Recommendations deleted: {recommendations_deleted}")
        print(f"  - Health data records deleted: {health_data_deleted}")
        
        return True
        
    except Exception as e:
        print(f"Error clearing user data: {e}")
        traceback.print_exc()
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return False

@eye_detection_bp.route('/start-new-analysis', methods=['POST'])
@login_required
def start_new_analysis():
    """Clear ALL old data and start a completely fresh analysis session"""
    try:
        print(f"Starting COMPLETE new analysis for user {current_user.id}")
        
        # Clear ALL data comprehensively
        success = clear_all_user_data(current_user.id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'All previous data cleared. Ready for completely new analysis.',
                'cleared_data': {
                    'recommendations': True,
                    'health_data': True
                },
                'next_step': 'image_upload'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to clear previous data completely'
            }), 500
            
    except Exception as e:
        print(f"Error starting new analysis: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred while starting new analysis'
        }), 500

@eye_detection_bp.route('/analyze-eye-image', methods=['POST'])
@login_required
def analyze_eye_image():
    """Analyze uploaded eye image for dry eye disease (Step 1)"""
    try:
        # ALWAYS clear data at the start of any new analysis
        print(f"Image analysis started - clearing old data for user {current_user.id}")
        clear_all_user_data(current_user.id)
        
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
                'next_step': 'questionnaire',
                'data_cleared': True  # Confirmation that data was cleared
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
        print(f"Processing questionnaire for user {current_user.id}")
        
        # Double-check data is clear before saving new analysis
        verify_data_cleared(current_user.id)
        
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
            'recommendations': recommendations,
            'fresh_analysis': True  # Flag to indicate this is completely new data
        }
        
        # Save comprehensive analysis to database (including recommendations)
        analysis_id = save_comprehensive_analysis(current_user.id, questionnaire_data, result, recommendations)
        result['analysis_id'] = analysis_id
        
        print(f"✓ New analysis saved successfully with ID: {analysis_id}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in questionnaire processing: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred during analysis. Please try again.'
        }), 500

def verify_data_cleared(user_id):
    """Verify that user data is actually cleared before proceeding"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Check recommendations
        cursor.execute("SELECT COUNT(*) FROM user_recommendations WHERE user_id = %s", (user_id,))
        rec_count = cursor.fetchone()[0]
        
        # Check health data
        cursor.execute("SELECT COUNT(*) FROM user_eye_health_data WHERE user_id = %s", (user_id,))
        health_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"Data verification for user {user_id}: Recommendations={rec_count}, Health data={health_count}")
        
        # If data still exists, clear it again
        if rec_count > 0 or health_count > 0:
            print("⚠️ Data still exists! Clearing again...")
            clear_all_user_data(user_id)
        
    except Exception as e:
        print(f"Error verifying data cleared: {e}")

# FIXED: Main recommendation endpoint with proper error handling
@eye_detection_bp.route('/my-recommendations')
@login_required
def get_user_recommendations():
    """Get user's current recommendations with enhanced debugging"""
    try:
        print(f"Fetching recommendations for user {current_user.id}")
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # FIXED: Simplified query to avoid potential JOIN issues
        # First get recommendations
        cursor.execute("""
            SELECT id, category, recommendation_text, priority, status, 
                   created_at, updated_at, completed_at
            FROM user_recommendations 
            WHERE user_id = %s 
            ORDER BY created_at DESC, id DESC
        """, (current_user.id,))
        
        recommendations_results = cursor.fetchall()
        print(f"Found {len(recommendations_results)} recommendation records for user {current_user.id}")
        
        # Then get health data separately
        cursor.execute("""
            SELECT dry_eye_disease, risk_score
            FROM user_eye_health_data 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (current_user.id,))
        
        health_result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not recommendations_results:
            print("No recommendations found - returning empty state")
            return jsonify({
                'success': True,
                'message': 'No recommendations found',
                'recommendations': {
                    'immediate_actions': [],
                    'medical_advice': [],
                    'lifestyle_changes': [],
                    'monitoring': []
                },
                'stats': {
                    'total_recommendations': 0,
                    'completed_count': 0,
                    'pending_count': 0,
                    'in_progress_count': 0,
                    'has_dry_eyes': False,
                    'risk_score': 0
                }
            })
        
        # Group recommendations by category
        recommendations_by_category = {
            'immediate_actions': [],
            'medical_advice': [],
            'lifestyle_changes': [],
            'monitoring': []
        }
        
        # Stats tracking
        user_stats = {
            'total_recommendations': len(recommendations_results),
            'completed_count': 0,
            'pending_count': 0,
            'in_progress_count': 0,
            'has_dry_eyes': False,
            'risk_score': 0
        }
        
        # Process recommendations
        for result in recommendations_results:
            rec = {
                'id': result[0],
                'category': result[1],
                'text': result[2],
                'priority': result[3],
                'status': result[4],
                'created_at': result[5].strftime('%Y-%m-%d %H:%M:%S') if result[5] else '',
                'updated_at': result[6].strftime('%Y-%m-%d %H:%M:%S') if result[6] else '',
                'completed_at': result[7].strftime('%Y-%m-%d %H:%M:%S') if result[7] else None
            }
            
            # Add to appropriate category
            category = result[1]
            if category in recommendations_by_category:
                recommendations_by_category[category].append(rec)
            
            # Update stats
            if result[4] == 'completed':
                user_stats['completed_count'] += 1
            elif result[4] == 'in_progress':
                user_stats['in_progress_count'] += 1
            else:
                user_stats['pending_count'] += 1
        
        # Add health data to stats
        if health_result:
            user_stats['has_dry_eyes'] = health_result[0] == 'Y'
            user_stats['risk_score'] = float(health_result[1]) if health_result[1] else 0.0
        
        print(f"Returning recommendations: {user_stats['total_recommendations']} total")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations_by_category,
            'stats': user_stats
        })
    
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to fetch recommendations'
        }), 500

@eye_detection_bp.route('/update-recommendation-status', methods=['POST'])
@login_required
def update_recommendation_status():
    """Update the status of a specific recommendation"""
    try:
        data = request.get_json()
        recommendation_id = data.get('recommendation_id')
        new_status = data.get('status')
        
        if not recommendation_id or not new_status:
            return jsonify({
                'success': False,
                'error': 'Missing recommendation_id or status'
            }), 400
        
        if new_status not in ['pending', 'in_progress', 'completed', 'dismissed']:
            return jsonify({
                'success': False,
                'error': 'Invalid status value'
            }), 400
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Verify the recommendation belongs to the current user
        cursor.execute("""
            SELECT id FROM user_recommendations 
            WHERE id = %s AND user_id = %s
        """, (recommendation_id, current_user.id))
        
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Recommendation not found or access denied'
            }), 404
        
        # Update the status
        update_query = """
            UPDATE user_recommendations 
            SET status = %s, updated_at = CURRENT_TIMESTAMP
        """
        params = [new_status]
        
        # Set completed_at if status is completed
        if new_status == 'completed':
            update_query += ", completed_at = CURRENT_TIMESTAMP"
        elif new_status in ['pending', 'in_progress']:
            update_query += ", completed_at = NULL"
        
        update_query += " WHERE id = %s AND user_id = %s"
        params.extend([recommendation_id, current_user.id])
        
        cursor.execute(update_query, params)
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Recommendation status updated successfully'
        })
    
    except Exception as e:
        print(f"Error updating recommendation status: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to update recommendation status'
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
                'class_imported': AdvancedDryEyeTextPredictor is not None
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

def save_recommendations_to_db(user_id, recommendations, analysis_id=None):
    """Save individual recommendations to the database with enhanced transaction handling"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Start transaction
        conn.begin()
        
        # Verify no old recommendations exist (extra safety check)
        cursor.execute("SELECT COUNT(*) FROM user_recommendations WHERE user_id = %s", (user_id,))
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️ Warning: Found {existing_count} existing recommendations. Clearing before saving new ones.")
            cursor.execute("DELETE FROM user_recommendations WHERE user_id = %s", (user_id,))
        
        # Save new recommendations with timestamp to ensure they're fresh
        saved_count = 0
        for category, rec_list in recommendations.items():
            for rec_text in rec_list:
                # Determine priority based on category and content
                priority = get_recommendation_priority(category, rec_text)
                
                cursor.execute("""
                    INSERT INTO user_recommendations 
                    (user_id, analysis_id, category, recommendation_text, priority, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """, (user_id, analysis_id, category, rec_text, priority, 'pending'))
                saved_count += 1
        
        # Commit transaction
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ Successfully saved {saved_count} new recommendations for user {user_id}")
        return True
        
    except Exception as e:
        print(f"Error saving recommendations: {e}")
        traceback.print_exc()
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return False

def get_recommendation_priority(category, text):
    """Determine recommendation priority based on category and content"""
    high_priority_keywords = [
        'ophthalmologist', 'doctor', 'emergency', 'immediately', 
        'within', 'urgent', 'severe', 'prescription'
    ]
    
    if category == 'medical_advice':
        return 'high'
    elif category == 'immediate_actions':
        return 'high' if any(keyword in text.lower() for keyword in high_priority_keywords) else 'medium'
    elif category == 'lifestyle_changes':
        return 'medium'
    else:  # monitoring
        return 'low'

def save_comprehensive_analysis(user_id, questionnaire_data, analysis_result, recommendations):
    """Save comprehensive analysis results to database with enhanced transaction handling"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Start transaction
        conn.begin()
        
        # Double-check no old data exists
        cursor.execute("SELECT COUNT(*) FROM user_eye_health_data WHERE user_id = %s", (user_id,))
        existing_health = cursor.fetchone()[0]
        
        if existing_health > 0:
            print(f"⚠️ Warning: Found {existing_health} existing health records. Clearing before saving new one.")
            cursor.execute("DELETE FROM user_eye_health_data WHERE user_id = %s", (user_id,))
        
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
            'risk_factors': json.dumps(analysis_result['risk_factors']),
            'recommendations_saved': 1
        }
        
        # Insert health data with explicit timestamp
        cursor.execute("""
            INSERT INTO user_eye_health_data (
                user_id, gender, age, sleep_duration, sleep_quality, stress_level,
                blood_pressure, heart_rate, daily_steps, physical_activity, height, weight,
                sleep_disorder, wake_up_during_night, feel_sleepy_during_day, caffeine_consumption,
                alcohol_consumption, smoking, medical_issue, ongoing_medication, smart_device_before_bed,
                average_screen_time, blue_light_filter, discomfort_eye_strain, redness_in_eye,
                itchiness_irritation_in_eye, dry_eye_disease, risk_score, risk_factors, 
                recommendations_saved, created_at, updated_at
            ) VALUES (
                %(user_id)s, %(gender)s, %(age)s, %(sleep_duration)s, %(sleep_quality)s, %(stress_level)s,
                %(blood_pressure)s, %(heart_rate)s, %(daily_steps)s, %(physical_activity)s, %(height)s, %(weight)s,
                %(sleep_disorder)s, %(wake_up_during_night)s, %(feel_sleepy_during_day)s, %(caffeine_consumption)s,
                %(alcohol_consumption)s, %(smoking)s, %(medical_issue)s, %(ongoing_medication)s, %(smart_device_before_bed)s,
                %(average_screen_time)s, %(blue_light_filter)s, %(discomfort_eye_strain)s, %(redness_in_eye)s,
                %(itchiness_irritation_in_eye)s, %(dry_eye_disease)s, %(risk_score)s, %(risk_factors)s, 
                %(recommendations_saved)s, NOW(), NOW()
            )
        """, health_data)
        
        # Get the analysis ID
        analysis_id = cursor.lastrowid
        
        # Save recommendations to separate table (using our enhanced function)
        # First commit the health data, then save recommendations
        conn.commit()
        cursor.close()
        conn.close()
        
        # Save recommendations using our enhanced function
        recommendations_saved = save_recommendations_to_db(user_id, recommendations, analysis_id)
        
        if not recommendations_saved:
            raise Exception("Failed to save recommendations")
        
        print(f"✓ Complete analysis saved successfully with ID: {analysis_id}")
        return analysis_id
        
    except Exception as e:
        print(f"Error saving comprehensive analysis: {e}")
        traceback.print_exc()
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return None

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
            ORDER BY created_at DESC 
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

# Additional endpoint for debugging user data
@eye_detection_bp.route('/debug-user-data')
@login_required
def debug_user_data():
    """Debug endpoint to check current user data state"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Check recommendations
        cursor.execute("""
            SELECT COUNT(*) as rec_count, MAX(created_at) as latest_rec 
            FROM user_recommendations WHERE user_id = %s
        """, (current_user.id,))
        rec_data = cursor.fetchone()
        
        # Check health data
        cursor.execute("""
            SELECT COUNT(*) as health_count, MAX(created_at) as latest_health 
            FROM user_eye_health_data WHERE user_id = %s
        """, (current_user.id,))
        health_data = cursor.fetchone()
        
        # Get latest recommendations summary
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM user_recommendations 
            WHERE user_id = %s 
            GROUP BY category
        """, (current_user.id,))
        category_counts = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'user_id': current_user.id,
            'recommendations': {
                'total_count': rec_data[0] if rec_data[0] else 0,
                'latest_created': rec_data[1].strftime('%Y-%m-%d %H:%M:%S') if rec_data[1] else None,
                'by_category': {cat[0]: cat[1] for cat in category_counts}
            },
            'health_data': {
                'total_count': health_data[0] if health_data[0] else 0,
                'latest_created': health_data[1].strftime('%Y-%m-%d %H:%M:%S') if health_data[1] else None
            }
        })
    
    except Exception as e:
        print(f"Error in debug endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500