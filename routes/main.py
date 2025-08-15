from flask import Blueprint, current_app, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os

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

@main_bp.route('/eye-analysis')
@login_required
def eye_analysis():
    return render_template('eye-analysis.html', user=current_user)

@main_bp.route('/recommendations')
@login_required
def recommendations():
    """Display user's personalized recommendations page"""
    return render_template('recommendations.html', user=current_user)

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@main_bp.route('/habits')
@login_required
def habits():
    """Eye habits tracking and practice page"""
    return render_template('habits.html', user=current_user)

@main_bp.route('/chatbot')
def chatbot():
    """AI chatbot interaction page"""
    return render_template('chatbot.html', user=current_user)

# @main_bp.route('/settings')
# @login_required
# def settings():
#     return render_template('settings.html', user=current_user)

# REMOVED: Duplicate /my-recommendations route - this is now handled in eye_detection.py
# REMOVED: Duplicate /update-recommendation-status route - this is now handled in eye_detection.py

@main_bp.route('/start-new-analysis', methods=['POST'])
@login_required  
def start_new_analysis():
    """Clear all user data and start fresh analysis"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Start transaction
        conn.begin()
        
        # Clear all user's recommendations
        cursor.execute("DELETE FROM user_recommendations WHERE user_id = %s", (current_user.id,))
        rec_deleted = cursor.rowcount
        
        # Clear all user's eye health data (keep user account)
        cursor.execute("DELETE FROM user_eye_health_data WHERE user_id = %s", (current_user.id,))
        health_deleted = cursor.rowcount
        
        # Clear habit tracking data
        cursor.execute("DELETE FROM habit_tracking WHERE user_id = %s", (current_user.id,))
        habit_track_deleted = cursor.rowcount
        
        cursor.execute("DELETE FROM user_habits WHERE user_id = %s", (current_user.id,))
        user_habits_deleted = cursor.rowcount
        
        cursor.execute("DELETE FROM habit_achievements WHERE user_id = %s", (current_user.id,))
        achievements_deleted = cursor.rowcount
        
        cursor.execute("DELETE FROM habit_summaries WHERE user_id = %s", (current_user.id,))
        summaries_deleted = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ Data cleared for user {current_user.id}:")
        print(f"  - Recommendations: {rec_deleted}")
        print(f"  - Health data: {health_deleted}")
        print(f"  - Habit tracking: {habit_track_deleted}")
        print(f"  - User habits: {user_habits_deleted}")
        print(f"  - Achievements: {achievements_deleted}")
        print(f"  - Summaries: {summaries_deleted}")
        
        return jsonify({
            'success': True, 
            'message': 'All data cleared successfully. Ready for new analysis.',
            'cleared_counts': {
                'recommendations': rec_deleted,
                'health_data': health_deleted,
                'habit_tracking': habit_track_deleted,
                'user_habits': user_habits_deleted,
                'achievements': achievements_deleted,
                'summaries': summaries_deleted
            }
        })
        
    except Exception as e:
        print(f"Error clearing user data: {e}")
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500