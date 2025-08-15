from flask import Blueprint, current_app, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import MySQLdb.cursors
from datetime import date, timedelta

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
    """Dashboard with real-time data from user's health and habits"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Get current user's latest risk score and health data
        cursor.execute("""
            SELECT risk_score, created_at, updated_at
            FROM user_eye_health_data 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (current_user.id,))
        
        current_health = cursor.fetchone()
        
        # Get previous month's risk score for comparison
        cursor.execute("""
            SELECT risk_score, created_at
            FROM user_eye_health_data 
            WHERE user_id = %s 
            AND created_at < DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            ORDER BY created_at DESC 
            LIMIT 1
        """, (current_user.id,))
        
        previous_health = cursor.fetchone()
        
        # Get today's habit completion stats
        today = date.today()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_habits,
                SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_habits,
                AVG(CASE WHEN ht.is_completed = 1 THEN 100 ELSE ht.completion_percentage END) as avg_completion
            FROM user_habits uh
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            WHERE uh.user_id = %s AND uh.is_active = 1
        """, (today, current_user.id))
        
        today_stats = cursor.fetchone()
        
        # Get current streak (longest active streak)
        cursor.execute("""
            SELECT MAX(streak_days) as max_streak
            FROM (
                SELECT COUNT(*) as streak_days
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s AND ht.is_completed = 1
                GROUP BY ht.user_habit_id
                HAVING COUNT(*) >= 1
            ) as streaks
        """, (current_user.id,))
        
        streak_data = cursor.fetchone()
        
        # Get weekly progress
        week_start = today - timedelta(days=today.weekday())
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT ht.user_habit_id) as active_habits,
                AVG(ht.completion_percentage) as avg_completion,
                SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_today
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            WHERE uh.user_id = %s AND ht.date >= %s
        """, (current_user.id, week_start))
        
        weekly_stats = cursor.fetchone()
        
        # Get water intake progress (assuming there's a water intake habit)
        cursor.execute("""
            SELECT 
                COALESCE(ht.completed_count, 0) as water_glasses,
                COALESCE(uh.custom_target_count, 8) as target_glasses,
                uh.id as user_habit_id
            FROM user_habits uh
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            JOIN eye_habits h ON uh.habit_id = h.id
            WHERE uh.user_id = %s AND uh.is_active = 1 
            AND (h.name LIKE '%water%' OR h.name LIKE '%hydration%')
            LIMIT 1
        """, (today, current_user.id))
        
        water_data = cursor.fetchone()
        
        # Calculate water streak if water habit exists
        water_streak = 0
        if water_data and water_data['user_habit_id']:
            cursor.execute("""
                SELECT COUNT(*) as streak_days
                FROM (
                    SELECT date, is_completed
                    FROM habit_tracking 
                    WHERE user_habit_id = %s 
                    AND date <= %s
                    ORDER BY date DESC
                ) dates
                WHERE is_completed = 1
                AND date >= (
                    SELECT MAX(date) 
                    FROM habit_tracking 
                    WHERE user_habit_id = %s 
                    AND is_completed = 0 
                    AND date <= %s
                ) OR (
                    SELECT MAX(date) 
                    FROM habit_tracking 
                    WHERE user_habit_id = %s 
                    AND is_completed = 0 
                    AND date <= %s
                ) IS NULL
            """, (water_data['user_habit_id'], today, water_data['user_habit_id'], today, water_data['user_habit_id'], today))
            
            water_streak_result = cursor.fetchone()
            water_streak = water_streak_result['streak_days'] if water_streak_result else 0
        
        # Get eye exercises progress
        cursor.execute("""
            SELECT 
                COALESCE(ht.completed_count, 0) as exercise_sessions,
                COALESCE(uh.custom_target_count, 3) as target_sessions
            FROM user_habits uh
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            JOIN eye_habits h ON uh.habit_id = h.id
            WHERE uh.user_id = %s AND uh.is_active = 1 
            AND (h.name LIKE '%exercise%' OR h.name LIKE '%blink%' OR h.name LIKE '%focus%')
            LIMIT 1
        """, (today, current_user.id))
        
        exercise_data = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # Calculate dashboard metrics
        current_risk_score = float(current_health['risk_score']) if current_health and current_health['risk_score'] else 0.0
        previous_risk_score = float(previous_health['risk_score']) if previous_health and previous_health['risk_score'] else current_risk_score
        
        risk_change = previous_risk_score - current_risk_score
        risk_reduction = ((previous_risk_score - current_risk_score) / previous_risk_score * 100) if previous_risk_score > 0 else 0
        
        habits_completed_pct = (today_stats['completed_habits'] / today_stats['total_habits'] * 100) if today_stats['total_habits'] > 0 else 0
        streak_days = streak_data['max_streak'] if streak_data and streak_data['max_streak'] else 0
        weekly_progress = weekly_stats['avg_completion'] if weekly_stats and weekly_stats['avg_completion'] else 0
        
        water_intake = water_data['water_glasses'] if water_data else 0
        water_target = water_data['target_glasses'] if water_data else 8
        water_percentage = (water_intake / water_target * 100) if water_target > 0 else 0
        
        exercise_sessions = exercise_data['exercise_sessions'] if exercise_data else 0
        exercise_target = exercise_data['target_sessions'] if exercise_data else 3
        exercise_percentage = (exercise_sessions / exercise_target * 100) if exercise_target > 0 else 0
        
        dashboard_data = {
            'current_risk_score': round(current_risk_score, 1),
            'risk_change': round(risk_change, 1),
            'risk_reduction': round(risk_reduction, 1),
            'habits_completed_pct': round(habits_completed_pct, 1),
            'streak_days': streak_days,
            'weekly_progress': round(weekly_progress, 1),
            'water_intake': water_intake,
            'water_target': water_target,
            'water_percentage': round(water_percentage, 1),
            'water_streak': water_streak,
            'exercise_sessions': exercise_sessions,
            'exercise_target': exercise_target,
            'exercise_percentage': round(exercise_percentage, 1),
            'previous_risk_score': round(previous_risk_score, 1)
        }
        
        return render_template('dashboard.html', user=current_user, data=dashboard_data)
        
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        # Fallback to template without data
        return render_template('dashboard.html', user=current_user, data={})

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