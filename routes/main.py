from flask import Blueprint, current_app, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from datetime import datetime, date, timedelta
import MySQLdb

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
    """Dashboard with dynamic data from database"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Get user's latest health data
        cursor.execute("""
            SELECT risk_score, dry_eye_disease, risk_factors, created_at
            FROM user_eye_health_data 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (current_user.id,))
        
        latest_health = cursor.fetchone()
        
        # Get previous health data for comparison
        cursor.execute("""
            SELECT risk_score, created_at
            FROM user_eye_health_data 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 2
        """, (current_user.id,))
        
        health_history = cursor.fetchall()
        
        # Get habit statistics
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT uh.id) as total_habits,
                COUNT(DISTINCT CASE WHEN ht.is_completed = 1 THEN uh.id END) as completed_today,
                AVG(ht.completion_percentage) as avg_completion
            FROM user_habits uh
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            WHERE uh.user_id = %s AND uh.is_active = 1
        """, (today, current_user.id))
        
        habit_stats = cursor.fetchone()
        
        # Get weekly habit progress
        cursor.execute("""
            SELECT 
                DATE(ht.date) as date,
                COUNT(DISTINCT uh.id) as total_habits,
                COUNT(DISTINCT CASE WHEN ht.is_completed = 1 THEN uh.id END) as completed_habits,
                AVG(ht.completion_percentage) as avg_completion
            FROM user_habits uh
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date >= %s
            WHERE uh.user_id = %s AND uh.is_active = 1
            GROUP BY DATE(ht.date)
            ORDER BY date
        """, (week_start, current_user.id))
        
        weekly_progress = cursor.fetchall()
        
        # Get streak information
        cursor.execute("""
            SELECT 
                uh.id as user_habit_id,
                h.name as habit_name,
                COUNT(DISTINCT ht.date) as streak_days
            FROM user_habits uh
            JOIN eye_habits h ON uh.habit_id = h.id
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.is_completed = 1
            WHERE uh.user_id = %s AND uh.is_active = 1
            GROUP BY uh.id, h.name
            ORDER BY streak_days DESC
            LIMIT 3
        """, (current_user.id,))
        
        top_streaks = cursor.fetchall()
        
        # Get today's specific habits
        cursor.execute("""
            SELECT 
                h.name as habit_name,
                h.icon as habit_icon,
                COALESCE(ht.completed_count, 0) as completed_count,
                COALESCE(uh.custom_target_count, h.target_count) as target_count,
                COALESCE(ht.completion_percentage, 0) as completion_percentage,
                ht.is_completed,
                h.target_unit
            FROM user_habits uh
            JOIN eye_habits h ON uh.habit_id = h.id
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            WHERE uh.user_id = %s AND uh.is_active = 1
            ORDER BY h.category, h.name
        """, (today, current_user.id))
        
        today_habits = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Calculate dynamic values
        current_risk_score = latest_health['risk_score'] if latest_health else 0.0
        previous_risk_score = health_history[1]['risk_score'] if len(health_history) > 1 else current_risk_score
        risk_change = round(current_risk_score - previous_risk_score, 1)
        risk_change_text = f"{abs(risk_change)} points {'lower' if risk_change < 0 else 'higher'}"
        
        habits_completed = habit_stats['completed_today'] if habit_stats else 0
        total_habits = habit_stats['total_habits'] if habit_stats else 0
        habits_percentage = round((habits_completed / total_habits * 100) if total_habits > 0 else 0)
        
        max_streak = max([streak['streak_days'] for streak in top_streaks]) if top_streaks else 0
        
        # Calculate weekly progress percentage
        weekly_completion = 0
        if weekly_progress:
            total_weekly_habits = sum([day['total_habits'] for day in weekly_progress])
            total_weekly_completed = sum([day['completed_habits'] for day in weekly_progress])
            weekly_completion = round((total_weekly_completed / total_weekly_habits * 100) if total_weekly_habits > 0 else 0)
        
        dashboard_data = {
            'user': current_user,
            'current_risk_score': current_risk_score,
            'risk_change': risk_change,
            'risk_change_text': risk_change_text,
            'habits_completed': habits_completed,
            'total_habits': total_habits,
            'habits_percentage': habits_percentage,
            'max_streak': max_streak,
            'weekly_completion': weekly_completion,
            'today_habits': today_habits,
            'top_streaks': top_streaks,
            'weekly_progress': weekly_progress,
            'has_health_data': latest_health is not None
        }
        
        return render_template('dashboard.html', **dashboard_data)
        
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        # Fallback to basic dashboard if there's an error
        safe_defaults = {
            'user': current_user,
            'current_risk_score': 0.0,
            'risk_change': 0.0,
            'risk_change_text': 'No change',
            'habits_completed': 0,
            'total_habits': 0,
            'habits_percentage': 0,
            'max_streak': 0,
            'weekly_completion': 0,
            'today_habits': [],
            'top_streaks': [],
            'weekly_progress': [],
            'has_health_data': False,
            'error': "Unable to load dashboard data"
        }
        return render_template('dashboard.html', **safe_defaults)

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