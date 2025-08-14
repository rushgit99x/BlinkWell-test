from flask import Blueprint, request, jsonify, render_template, current_app, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import MySQLdb
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings_bp = Blueprint('settings', __name__)

# Allowed file extensions for profile pictures
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOAD_FOLDER = 'static/uploads/profile_pics'

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@settings_bp.route('/settings')
@login_required
def settings_page():
    """Display the settings page with current user preferences"""
    try:
        # Get user's current settings and preferences
        user_settings = get_user_settings(current_user.id)
        notification_preferences = get_notification_preferences(current_user.id)
        
        return render_template('settings.html', 
                             user=current_user, 
                             settings=user_settings,
                             notification_preferences=notification_preferences)
    except Exception as e:
        logger.error(f"Error loading settings page: {str(e)}")
        flash('Error loading settings. Please try again.', 'error')
        return redirect(url_for('main.dashboard'))

@settings_bp.route('/api/settings/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile information"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        
        # Validation
        if not username or len(username) < 3:
            return jsonify({'success': False, 'error': 'Username must be at least 3 characters long'}), 400
        
        if not email or '@' not in email:
            return jsonify({'success': False, 'error': 'Please enter a valid email address'}), 400
        
        # Check if username is already taken by another user
        if username != current_user.username:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s AND id != %s", (username, current_user.id))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'success': False, 'error': 'Username is already taken'}), 400
            cursor.close()
            conn.close()
        
        # Check if email is already taken by another user
        if email != current_user.email:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE email = %s AND id != %s", (email, current_user.id))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'success': False, 'error': 'Email is already taken'}), 400
            cursor.close()
            conn.close()
        
        # Update profile in database
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET username = %s, email = %s, updated_at = NOW()
            WHERE id = %s
        """, (username, email, current_user.id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Update current_user object
        current_user.username = username
        current_user.email = email
        
        logger.info(f"Profile updated for user {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': {
                'username': username,
                'email': email
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update profile. Please try again.'}), 500

@settings_bp.route('/api/settings/profile-picture', methods=['POST'])
@login_required
def update_profile_picture():
    """Update user profile picture"""
    try:
        if 'profile_pic' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['profile_pic']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Generate unique filename
            filename = secure_filename(f"user_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file.filename.rsplit('.', 1)[1].lower()}")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save file
            file.save(filepath)
            
            # Update database with new profile picture path
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE users 
                SET profile_pic = %s, updated_at = NOW()
                WHERE id = %s
            """, (f"uploads/profile_pics/{filename}", current_user.id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            # Update current_user object
            current_user.profile_pic = f"uploads/profile_pics/{filename}"
            
            logger.info(f"Profile picture updated for user {current_user.id}")
            
            return jsonify({
                'success': True,
                'message': 'Profile picture updated successfully',
                'profile_pic': f"uploads/profile_pics/{filename}"
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image file.'}), 400
            
    except Exception as e:
        logger.error(f"Error updating profile picture: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update profile picture. Please try again.'}), 500

@settings_bp.route('/api/settings/password', methods=['PUT'])
@login_required
def update_password():
    """Update user password"""
    try:
        data = request.get_json()
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not current_password or not new_password or not confirm_password:
            return jsonify({'success': False, 'error': 'All password fields are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'success': False, 'error': 'New passwords do not match'}), 400
        
        if len(new_password) < 6:
            return jsonify({'success': False, 'error': 'New password must be at least 6 characters long'}), 400
        
        # Verify current password
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id = %s", (current_user.id,))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not user_data or not check_password_hash(user_data[0], current_password):
            return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400
        
        # Hash new password and update
        new_password_hash = generate_password_hash(new_password)
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET password_hash = %s, updated_at = NOW()
            WHERE id = %s
        """, (new_password_hash, current_user.id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Password updated for user {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Password updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating password: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update password. Please try again.'}), 500

@settings_bp.route('/api/settings/notifications', methods=['PUT'])
@login_required
def update_notification_preferences():
    """Update user notification preferences"""
    try:
        data = request.get_json()
        
        # Get notification preferences from request
        eye_exercise_reminders = data.get('eye_exercise_reminders', False)
        daily_habit_tracking = data.get('daily_habit_tracking', False)
        weekly_progress_reports = data.get('weekly_progress_reports', False)
        risk_assessment_updates = data.get('risk_assessment_updates', False)
        email_frequency = data.get('email_frequency', 'weekly')
        
        # Update notification preferences in database
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Check if user has notification preferences record
        cursor.execute("SELECT id FROM user_notification_preferences WHERE user_id = %s", (current_user.id,))
        existing_prefs = cursor.fetchone()
        
        if existing_prefs:
            # Update existing preferences
            cursor.execute("""
                UPDATE user_notification_preferences 
                SET eye_exercise_reminders = %s, daily_habit_tracking = %s, 
                    weekly_progress_reports = %s, risk_assessment_updates = %s,
                    email_frequency = %s, updated_at = NOW()
                WHERE user_id = %s
            """, (eye_exercise_reminders, daily_habit_tracking, weekly_progress_reports, 
                  risk_assessment_updates, email_frequency, current_user.id))
        else:
            # Create new preferences record
            cursor.execute("""
                INSERT INTO user_notification_preferences 
                (user_id, eye_exercise_reminders, daily_habit_tracking, 
                 weekly_progress_reports, risk_assessment_updates, email_frequency, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (current_user.id, eye_exercise_reminders, daily_habit_tracking, 
                  weekly_progress_reports, risk_assessment_updates, email_frequency))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Notification preferences updated for user {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Notification preferences updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating notification preferences: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update notification preferences. Please try again.'}), 500

@settings_bp.route('/api/settings/privacy', methods=['PUT'])
@login_required
def update_privacy_settings():
    """Update user privacy and security settings"""
    try:
        data = request.get_json()
        
        two_factor_auth = data.get('two_factor_auth', False)
        share_data_research = data.get('share_data_research', False)
        
        # Update privacy settings in database
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Check if user has privacy settings record
        cursor.execute("SELECT id FROM user_privacy_settings WHERE user_id = %s", (current_user.id,))
        existing_settings = cursor.fetchone()
        
        if existing_settings:
            # Update existing settings
            cursor.execute("""
                UPDATE user_privacy_settings 
                SET two_factor_auth = %s, share_data_research = %s, updated_at = NOW()
                WHERE user_id = %s
            """, (two_factor_auth, share_data_research, current_user.id))
        else:
            # Create new settings record
            cursor.execute("""
                INSERT INTO user_privacy_settings 
                (user_id, two_factor_auth, share_data_research, created_at)
                VALUES (%s, %s, %s, NOW())
            """, (current_user.id, two_factor_auth, share_data_research))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Privacy settings updated for user {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Privacy settings updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating privacy settings: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update privacy settings. Please try again.'}), 500

@settings_bp.route('/api/settings/export-data', methods=['POST'])
@login_required
def export_user_data():
    """Export all user data as JSON"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Collect all user data
        user_data = {
            'export_date': datetime.now().isoformat(),
            'user_info': {},
            'eye_health_data': [],
            'habit_data': [],
            'recommendations': [],
            'achievements': []
        }
        
        # Get user info
        cursor.execute("SELECT username, email, created_at, updated_at FROM users WHERE id = %s", (current_user.id,))
        user_info = cursor.fetchone()
        if user_info:
            user_data['user_info'] = user_info
        
        # Get eye health data
        cursor.execute("SELECT * FROM user_eye_health_data WHERE user_id = %s", (current_user.id,))
        user_data['eye_health_data'] = cursor.fetchall()
        
        # Get habit data
        cursor.execute("""
            SELECT uh.*, h.name as habit_name, h.description, h.category
            FROM user_habits uh
            JOIN eye_habits h ON uh.habit_id = h.id
            WHERE uh.user_id = %s
        """, (current_user.id,))
        user_data['habit_data'] = cursor.fetchall()
        
        # Get habit tracking data
        cursor.execute("""
            SELECT ht.*, h.name as habit_name
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            JOIN eye_habits h ON uh.habit_id = h.id
            WHERE uh.user_id = %s
        """, (current_user.id,))
        user_data['habit_tracking'] = cursor.fetchall()
        
        # Get recommendations
        cursor.execute("SELECT * FROM user_recommendations WHERE user_id = %s", (current_user.id,))
        user_data['recommendations'] = cursor.fetchall()
        
        # Get achievements
        cursor.execute("SELECT * FROM habit_achievements WHERE user_id = %s", (current_user.id,))
        user_data['achievements'] = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Create export directory if it doesn't exist
        export_dir = 'static/exports'
        os.makedirs(export_dir, exist_ok=True)
        
        # Save data to file
        filename = f"user_data_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(user_data, f, indent=2, default=str)
        
        logger.info(f"Data exported for user {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Data export completed successfully',
            'download_url': f"/static/exports/{filename}",
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error exporting user data: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to export data. Please try again.'}), 500

@settings_bp.route('/api/settings/clear-cache', methods=['POST'])
@login_required
def clear_user_cache():
    """Clear user's cached data"""
    try:
        # This is a placeholder for cache clearing functionality
        # In a real application, you would clear Redis cache, file cache, etc.
        
        logger.info(f"Cache cleared for user {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to clear cache. Please try again.'}), 500

@settings_bp.route('/api/settings/delete-account', methods=['DELETE'])
@login_required
def delete_user_account():
    """Delete user account and all associated data"""
    try:
        data = request.get_json()
        confirmation = data.get('confirmation', '')
        
        if confirmation != 'DELETE':
            return jsonify({'success': False, 'error': 'Invalid confirmation. Please type DELETE to confirm.'}), 400
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Start transaction
        conn.begin()
        
        try:
            # Delete all user data in the correct order (respecting foreign keys)
            cursor.execute("DELETE FROM habit_tracking WHERE user_habit_id IN (SELECT id FROM user_habits WHERE user_id = %s)", (current_user.id,))
            cursor.execute("DELETE FROM user_habits WHERE user_id = %s", (current_user.id,))
            cursor.execute("DELETE FROM habit_achievements WHERE user_id = %s", (current_user.id,))
            cursor.execute("DELETE FROM habit_summaries WHERE user_id = %s", (current_user.id,))
            cursor.execute("DELETE FROM user_eye_health_data WHERE user_id = %s", (current_user.id,))
            cursor.execute("DELETE FROM user_recommendations WHERE user_id = %s", (current_user.id,))
            cursor.execute("DELETE FROM user_notification_preferences WHERE user_id = %s", (current_user.id,))
            cursor.execute("DELETE FROM user_privacy_settings WHERE user_id = %s", (current_user.id,))
            
            # Finally delete the user account
            cursor.execute("DELETE FROM users WHERE id = %s", (current_user.id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Account deleted for user {current_user.id}")
            
            return jsonify({
                'success': True,
                'message': 'Account deleted successfully'
            })
            
        except Exception as e:
            conn.rollback()
            cursor.close()
            conn.close()
            raise e
            
    except Exception as e:
        logger.error(f"Error deleting user account: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to delete account. Please try again.'}), 500

def get_user_settings(user_id):
    """Get user's current settings"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("""
            SELECT username, email, profile_pic, created_at, updated_at
            FROM users WHERE id = %s
        """, (user_id,))
        
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return user_data or {}
        
    except Exception as e:
        logger.error(f"Error getting user settings: {str(e)}")
        return {}

def get_notification_preferences(user_id):
    """Get user's notification preferences"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("""
            SELECT eye_exercise_reminders, daily_habit_tracking, 
                   weekly_progress_reports, risk_assessment_updates, email_frequency
            FROM user_notification_preferences WHERE user_id = %s
        """, (user_id,))
        
        prefs = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # Return default preferences if none exist
        if not prefs:
            return {
                'eye_exercise_reminders': True,
                'daily_habit_tracking': True,
                'weekly_progress_reports': False,
                'risk_assessment_updates': True,
                'email_frequency': 'weekly'
            }
        
        return prefs
        
    except Exception as e:
        logger.error(f"Error getting notification preferences: {str(e)}")
        return {
            'eye_exercise_reminders': True,
            'daily_habit_tracking': True,
            'weekly_progress_reports': False,
            'risk_assessment_updates': True,
            'email_frequency': 'weekly'
        }