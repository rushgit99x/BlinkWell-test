from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from models.user import save_eye_health_data
from werkzeug.utils import secure_filename
import os
import re
import json

main_bp = Blueprint('main', __name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact')
def contact():
    return render_template('contact.html')

@main_bp.route('/risk-assessment')
@login_required
def risk_assessment():
    return render_template('risk-assessment.html', user=current_user)

@main_bp.route('/eye-analysis')
@login_required
def eye_analysis():
    return render_template('eye-analysis.html', user=current_user)

@main_bp.route('/save-assessment', methods=['POST'])
@login_required
def save_assessment():
    # Get JSON data from form
    if not request.form.get('assessment_data'):
        return jsonify({'success': False, 'message': 'Missing assessment data'}), 400
    try:
        data = json.loads(request.form.get('assessment_data'))
    except json.JSONDecodeError:
        return jsonify({'success': False, 'message': 'Invalid assessment data format'}), 400

    required_fields = [
        'gender', 'age', 'sleep_duration', 'sleep_quality', 'stress_level',
        'blood_pressure', 'heart_rate', 'daily_steps', 'physical_activity',
        'height', 'weight', 'sleep_disorder', 'wake_up_during_night',
        'feel_sleepy_during_day', 'caffeine_consumption', 'alcohol_consumption',
        'smoking', 'medical_issue', 'ongoing_medication', 'smart_device_before_bed',
        'average_screen_time', 'blue_light_filter', 'discomfort_eye_strain',
        'redness_in_eye', 'itchiness_irritation_in_eye', 'dry_eye_disease',
        'risk_score', 'risk_factors'
    ]

    # Validate required fields
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'message': f'Missing required field: {field}'}), 400

    # Validate data types and constraints
    try:
        data['age'] = int(data['age'])
        data['sleep_duration'] = float(data['sleep_duration'])
        data['sleep_quality'] = int(data['sleep_quality'])
        data['stress_level'] = int(data['stress_level'])
        data['heart_rate'] = int(data['heart_rate'])
        data['daily_steps'] = int(data['daily_steps'])
        data['physical_activity'] = int(data['physical_activity'])
        data['height'] = int(data['height'])
        data['weight'] = int(data['weight'])
        data['average_screen_time'] = float(data['average_screen_time'])
        data['risk_score'] = float(data['risk_score'])
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Invalid data type for numeric field'}), 400

    # Validate enum fields
    enum_fields = ['gender', 'sleep_disorder', 'wake_up_during_night', 'feel_sleepy_during_day',
                   'caffeine_consumption', 'alcohol_consumption', 'smoking', 'medical_issue',
                   'ongoing_medication', 'smart_device_before_bed', 'blue_light_filter',
                   'discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye',
                   'dry_eye_disease']
    for field in enum_fields:
        if field == 'gender' and data[field] not in ['M', 'F', 'Other']:
            return jsonify({'success': False, 'message': f'Invalid value for {field}'}), 400
        elif field != 'gender' and data[field] not in ['Y', 'N']:
            return jsonify({'success': False, 'message': f'Invalid value for {field}'}), 400

    # Validate blood pressure format
    if not isinstance(data['blood_pressure'], str) or not bool(re.match(r'^\d{2,3}/\d{2,3}$', data['blood_pressure'])):
        return jsonify({'success': False, 'message': 'Invalid blood pressure format'}), 400

    # Handle file upload
    eye_image_path = None
    if 'eye_image' in request.files:
        file = request.files['eye_image']
        if file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            eye_image_path = os.path.join(UPLOAD_FOLDER, filename).replace('\\', '/')
        else:
            return jsonify({'success': False, 'message': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400
    
    data['eye_image_path'] = eye_image_path

    # Save to database
    if save_eye_health_data(current_user.id, data):
        return jsonify({'success': True, 'message': 'Assessment saved successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to save assessment'}), 500

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)