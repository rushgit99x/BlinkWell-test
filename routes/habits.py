from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime, date, timedelta
import MySQLdb

habits_bp = Blueprint('habits', __name__)

@habits_bp.route('/habits')
@login_required
def habits_dashboard():
    """Main habits dashboard page"""
    return render_template('habits.html', user=current_user)

@habits_bp.route('/api/habits/available')
@login_required
def get_available_habits():
    """Get all available eye habits"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)  # Use DictCursor for dictionary results
        
        cursor.execute("""
            SELECT h.*, 
                   CASE WHEN uh.id IS NOT NULL THEN 1 ELSE 0 END as is_selected
            FROM eye_habits h
            LEFT JOIN user_habits uh ON h.id = uh.habit_id AND uh.user_id = %s AND uh.is_active = 1
            WHERE h.is_active = 1
            ORDER BY h.category, h.name
        """, (current_user.id,))
        
        habits = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert any datetime objects to strings for JSON serialization
        for habit in habits:
            if 'created_at' in habit and habit['created_at']:
                habit['created_at'] = habit['created_at'].isoformat()
        
        return jsonify({
            'success': True,
            'habits': habits
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Also fix the get_user_habits function
@habits_bp.route('/api/habits/user-habits')
@login_required
def get_user_habits():
    """Get user's selected habits with today's progress"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)  # Use DictCursor
        
        today = date.today()
        
        cursor.execute("""
            SELECT 
                uh.id as user_habit_id,
                uh.user_id,
                uh.habit_id,
                uh.custom_target_count,
                uh.custom_target_unit,
                uh.reminder_time,
                uh.reminder_enabled,
                h.name,
                h.description,
                h.category,
                h.icon,
                h.target_count,
                h.target_unit,
                h.instructions,
                h.benefits,
                h.difficulty_level,
                h.estimated_time_minutes,
                COALESCE(ht.completed_count, 0) as today_completed,
                COALESCE(uh.custom_target_count, h.target_count) as target_count,
                COALESCE(uh.custom_target_unit, h.target_unit) as target_unit,
                COALESCE(ht.completion_percentage, 0) as completion_percentage,
                ht.is_completed,
                ht.mood_before,
                ht.mood_after,
                -- Calculate streak
                (SELECT COUNT(DISTINCT ht2.date) 
                 FROM habit_tracking ht2 
                 WHERE ht2.user_habit_id = uh.id 
                 AND ht2.is_completed = 1 
                 AND ht2.date <= %s
                 AND ht2.date >= DATE_SUB(%s, INTERVAL 30 DAY)
                ) as streak_days
            FROM user_habits uh
            JOIN eye_habits h ON uh.habit_id = h.id
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            WHERE uh.user_id = %s AND uh.is_active = 1
            ORDER BY h.category, h.name
        """, (today, today, today, current_user.id))
        
        user_habits = cursor.fetchall()
        
        # Get weekly progress summary
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
        
        # Convert datetime objects to strings
        for habit in user_habits:
            if 'reminder_time' in habit and habit['reminder_time']:
                habit['reminder_time'] = str(habit['reminder_time'])
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'habits': user_habits,
            'weekly_stats': weekly_stats or {}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/add-habit', methods=['POST'])
@login_required
def add_user_habit():
    """Add a habit to user's routine"""
    try:
        data = request.get_json()
        habit_id = data.get('habit_id')
        custom_target = data.get('custom_target_count')
        reminder_time = data.get('reminder_time')
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_habits 
            (user_id, habit_id, custom_target_count, reminder_time, start_date)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            is_active = 1, 
            custom_target_count = VALUES(custom_target_count),
            reminder_time = VALUES(reminder_time)
        """, (current_user.id, habit_id, custom_target, reminder_time, date.today()))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Habit added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/track-progress', methods=['POST'])
@login_required
def track_habit_progress():
    """Update habit progress for today"""
    try:
        data = request.get_json()
        user_habit_id = data.get('user_habit_id')
        completed_count = data.get('completed_count', 0)
        target_count = data.get('target_count', 1)
        mood_before = data.get('mood_before')
        mood_after = data.get('mood_after')
        notes = data.get('notes', '')
        
        completion_percentage = min((completed_count / target_count) * 100, 100)
        is_completed = completion_percentage >= 100
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get habit_id from user_habit_id
        cursor.execute("SELECT habit_id FROM user_habits WHERE id = %s", (user_habit_id,))
        habit_result = cursor.fetchone()
        if not habit_result:
            return jsonify({'success': False, 'error': 'Habit not found'})
        
        habit_id = habit_result[0]
        today = date.today()
        
        # Calculate current streak
        cursor.execute("""
            SELECT COALESCE(MAX(streak_day), 0) + 1 as next_streak
            FROM habit_tracking 
            WHERE user_habit_id = %s AND is_completed = 1 
            AND date = DATE_SUB(%s, INTERVAL 1 DAY)
        """, (user_habit_id, today))
        
        streak_result = cursor.fetchone()
        streak_day = streak_result[0] if streak_result and is_completed else (1 if is_completed else 0)
        
        # Insert or update today's tracking
        cursor.execute("""
            INSERT INTO habit_tracking 
            (user_id, user_habit_id, habit_id, date, completed_count, target_count, 
             completion_percentage, completion_time, notes, mood_before, mood_after, 
             is_completed, streak_day)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            completed_count = VALUES(completed_count),
            completion_percentage = VALUES(completion_percentage),
            completion_time = CASE WHEN VALUES(is_completed) = 1 AND is_completed = 0 
                                  THEN CURRENT_TIME() ELSE completion_time END,
            notes = VALUES(notes),
            mood_before = VALUES(mood_before),
            mood_after = VALUES(mood_after),
            is_completed = VALUES(is_completed),
            streak_day = VALUES(streak_day),
            updated_at = CURRENT_TIMESTAMP
        """, (current_user.id, user_habit_id, habit_id, today, completed_count, 
              target_count, completion_percentage, 
              datetime.now().time() if is_completed else None,
              notes, mood_before, mood_after, is_completed, streak_day))
        
        conn.commit()
        
        # Check for achievements
        achievements = check_achievements(cursor, current_user.id, user_habit_id, streak_day, completion_percentage)
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Progress updated successfully',
            'completion_percentage': completion_percentage,
            'is_completed': is_completed,
            'streak_day': streak_day,
            'achievements': achievements
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def check_achievements(cursor, user_id, user_habit_id, streak_day, completion_percentage):
    """Check and award achievements"""
    achievements = []
    today = date.today()
    
    # Get habit info
    cursor.execute("""
        SELECT h.name, uh.habit_id 
        FROM user_habits uh 
        JOIN eye_habits h ON uh.habit_id = h.id 
        WHERE uh.id = %s
    """, (user_habit_id,))
    
    habit_info = cursor.fetchone()
    if not habit_info:
        return achievements
    
    habit_name = habit_info[0]
    habit_id = habit_info[1]
    
    # Check streak achievements
    if streak_day in [3, 7, 14, 30, 60, 100]:
        achievement_name = f"{streak_day}-Day Streak: {habit_name}"
        achievement_desc = f"Completed {habit_name} for {streak_day} consecutive days!"
        
        # Check if already earned
        cursor.execute("""
            SELECT id FROM habit_achievements 
            WHERE user_id = %s AND habit_id = %s AND achievement_type = 'streak' AND value = %s
        """, (user_id, habit_id, streak_day))
        
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO habit_achievements 
                (user_id, habit_id, achievement_type, achievement_name, achievement_description, 
                 badge_icon, value, earned_date)
                VALUES (%s, %s, 'streak', %s, %s, 'fas fa-fire', %s, %s)
            """, (user_id, habit_id, achievement_name, achievement_desc, streak_day, today))
            
            achievements.append({
                'type': 'streak',
                'name': achievement_name,
                'description': achievement_desc,
                'icon': 'fas fa-fire',
                'value': streak_day
            })
    
    return achievements

@habits_bp.route('/api/habits/weekly-summary')
@login_required
def get_weekly_summary():
    """Get weekly habit completion summary"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get last 7 days
        end_date = date.today()
        start_date = end_date - timedelta(days=6)
        
        cursor.execute("""
            SELECT 
                ht.date,
                h.name,
                h.category,
                h.icon,
                ht.completion_percentage,
                ht.is_completed,
                ht.mood_after
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            JOIN eye_habits h ON ht.habit_id = h.id
            WHERE uh.user_id = %s 
            AND ht.date BETWEEN %s AND %s
            ORDER BY ht.date DESC, h.category, h.name
        """, (current_user.id, start_date, end_date))
        
        tracking_data = cursor.fetchall()
        
        # Get overall stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT uh.id) as total_active_habits,
                AVG(ht.completion_percentage) as avg_completion,
                COUNT(CASE WHEN ht.is_completed = 1 THEN 1 END) as total_completed,
                COUNT(ht.id) as total_tracked
            FROM user_habits uh
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id 
                AND ht.date BETWEEN %s AND %s
            WHERE uh.user_id = %s AND uh.is_active = 1
        """, (start_date, end_date, current_user.id))
        
        weekly_stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'tracking_data': tracking_data,
            'weekly_stats': weekly_stats,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/remove-habit', methods=['POST'])
@login_required
def remove_user_habit():
    """Remove a habit from user's routine"""
    try:
        data = request.get_json()
        user_habit_id = data.get('user_habit_id')
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Soft delete by setting is_active = 0
        cursor.execute("""
            UPDATE user_habits 
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
        """, (user_habit_id, current_user.id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Habit removed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/achievements')
@login_required
def get_user_achievements():
    """Get user's earned achievements"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ha.*,
                h.name as habit_name
            FROM habit_achievements ha
            LEFT JOIN eye_habits h ON ha.habit_id = h.id
            WHERE ha.user_id = %s
            ORDER BY ha.earned_date DESC, ha.value DESC
        """, (current_user.id,))
        
        achievements = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'achievements': achievements
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/analytics/<int:days>')
@login_required
def get_habit_analytics(days):
    """Get habit completion analytics for specified number of days"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        start_date = date.today() - timedelta(days=days-1)
        
        # Daily completion rates
        cursor.execute("""
            SELECT 
                ht.date,
                COUNT(ht.id) as total_habits,
                COUNT(CASE WHEN ht.is_completed = 1 THEN 1 END) as completed_habits,
                AVG(ht.completion_percentage) as avg_completion,
                AVG(ht.mood_after - ht.mood_before) as mood_improvement
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            WHERE uh.user_id = %s AND ht.date >= %s
            GROUP BY ht.date
            ORDER BY ht.date
        """, (current_user.id, start_date))
        
        daily_analytics = cursor.fetchall()
        
        # Category-wise analytics
        cursor.execute("""
            SELECT 
                h.category,
                COUNT(ht.id) as total_tracking_entries,
                AVG(ht.completion_percentage) as avg_completion,
                COUNT(CASE WHEN ht.is_completed = 1 THEN 1 END) as completed_count
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            JOIN eye_habits h ON ht.habit_id = h.id
            WHERE uh.user_id = %s AND ht.date >= %s
            GROUP BY h.category
            ORDER BY avg_completion DESC
        """, (current_user.id, start_date))
        
        category_analytics = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'daily_analytics': daily_analytics,
            'category_analytics': category_analytics,
            'period_days': days
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/quick-log', methods=['POST'])
@login_required
def quick_log_habit():
    """Quick log completion of a habit"""
    try:
        data = request.get_json()
        user_habit_id = data.get('user_habit_id')
        increment = data.get('increment', 1)  # How much to increment by
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get current progress and target
        today = date.today()
        cursor.execute("""
            SELECT 
                COALESCE(ht.completed_count, 0) as current_count,
                COALESCE(uh.custom_target_count, h.target_count) as target_count,
                uh.habit_id
            FROM user_habits uh
            JOIN eye_habits h ON uh.habit_id = h.id
            LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
            WHERE uh.id = %s AND uh.user_id = %s
        """, (today, user_habit_id, current_user.id))
        
        result = cursor.fetchone()
        if not result:
            return jsonify({'success': False, 'error': 'Habit not found'})
        
        current_count = result[0]
        target_count = result[1]
        habit_id = result[2]
        
        new_count = min(current_count + increment, target_count)
        completion_percentage = (new_count / target_count) * 100
        is_completed = completion_percentage >= 100
        
        # Update tracking
        cursor.execute("""
            INSERT INTO habit_tracking 
            (user_id, user_habit_id, habit_id, date, completed_count, target_count, 
             completion_percentage, is_completed, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
            completed_count = VALUES(completed_count),
            completion_percentage = VALUES(completion_percentage),
            is_completed = VALUES(is_completed),
            completion_time = CASE WHEN VALUES(is_completed) = 1 AND is_completed = 0 
                                  THEN CURRENT_TIME() ELSE completion_time END,
            updated_at = CURRENT_TIMESTAMP
        """, (current_user.id, user_habit_id, habit_id, today, new_count, 
              target_count, completion_percentage, is_completed))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'new_count': new_count,
            'target_count': target_count,
            'completion_percentage': completion_percentage,
            'is_completed': is_completed
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# @habits_bp.route('/api/habits/habit-details/<int:habit_id>')
# @login_required
# def get_habit_details(habit_id):
#     """Get detailed information about a specific habit"""
#     try:
#         conn = current_app.config['get_db_connection']()
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             SELECT h.*, 
#                    uh.id as user_habit_id,
#                    uh.custom_target_count,
#                    uh.reminder_time,
#                    uh.start_date,
#                    -- Last 30 days analytics
#                    (SELECT AVG(ht.completion_percentage) 
#                     FROM habit_tracking ht 
#                     WHERE ht.user_habit_id = uh.id 
#                     AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as avg_completion_30d,
#                    (SELECT COUNT(*) 
#                     FROM habit_tracking ht 
#                     WHERE ht.user_habit_id = uh.id 
#                     AND ht.is_completed = 1 
#                     AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as completed_days_30d,
#                    (SELECT MAX(ht.streak_day) 
#                     FROM habit_tracking ht 
#                     WHERE ht.user_habit_id = uh.id) as best_streak
#             FROM eye_habits h
#             LEFT JOIN user_habits uh ON h.id = uh.habit_id AND uh.user_id = %s AND uh.is_active = 1
#             WHERE h.id = %s
#         """, (current_user.id, habit_id))
        
#         habit_details = cursor.fetchone()
        
#         if not habit_details:
#             return jsonify({'success': False, 'error': 'Habit not found'})
        
#         # Get last 7 days progress
#         cursor.execute("""
#             SELECT ht.date, ht.completion_percentage, ht.is_completed, ht.mood_after
#             FROM habit_tracking ht
#             JOIN user_habits uh ON ht.user_habit_id = uh.id
#             WHERE uh.user_id = %s AND uh.habit_id = %s 
#             AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
#             ORDER BY ht.date DESC
#         """, (current_user.id, habit_id))
        
#         recent_progress = cursor.fetchall()
        
#         cursor.close()
#         conn.close()
        
#         return jsonify({
#             'success': True,
#             'habit': habit_details,
#             'recent_progress': recent_progress
#         })
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @habits_bp.route('/api/habits/habit-details/<int:habit_id>')
# @login_required
# def get_habit_details(habit_id):
#     """Get detailed information about a specific habit"""
#     try:
#         conn = current_app.config['get_db_connection']()
#         cursor = conn.cursor(MySQLdb.cursors.DictCursor)  # Use DictCursor
        
#         cursor.execute("""
#             SELECT h.*, 
#                    uh.id as user_habit_id,
#                    uh.custom_target_count,
#                    uh.reminder_time,
#                    uh.start_date,
#                    -- Last 30 days analytics
#                    (SELECT AVG(ht.completion_percentage) 
#                     FROM habit_tracking ht 
#                     WHERE ht.user_habit_id = uh.id 
#                     AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as avg_completion_30d,
#                    (SELECT COUNT(*) 
#                     FROM habit_tracking ht 
#                     WHERE ht.user_habit_id = uh.id 
#                     AND ht.is_completed = 1 
#                     AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as completed_days_30d,
#                    (SELECT MAX(ht.streak_day) 
#                     FROM habit_tracking ht 
#                     WHERE ht.user_habit_id = uh.id) as best_streak
#             FROM eye_habits h
#             LEFT JOIN user_habits uh ON h.id = uh.habit_id AND uh.user_id = %s AND uh.is_active = 1
#             WHERE h.id = %s
#         """, (current_user.id, habit_id))
        
#         habit_details = cursor.fetchone()
        
#         if not habit_details:
#             cursor.close()
#             conn.close()
#             return jsonify({'success': False, 'error': 'Habit not found'})
        
#         # Get last 7 days progress
#         cursor.execute("""
#             SELECT ht.date, ht.completion_percentage, ht.is_completed, ht.mood_after
#             FROM habit_tracking ht
#             JOIN user_habits uh ON ht.user_habit_id = uh.id
#             WHERE uh.user_id = %s AND uh.habit_id = %s 
#             AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
#             ORDER BY ht.date DESC
#         """, (current_user.id, habit_id))
        
#         recent_progress = cursor.fetchall()
        
#         # Convert datetime objects to strings for JSON serialization
#         if 'created_at' in habit_details and habit_details['created_at']:
#             habit_details['created_at'] = habit_details['created_at'].isoformat()
#         if 'reminder_time' in habit_details and habit_details['reminder_time']:
#             habit_details['reminder_time'] = str(habit_details['reminder_time'])
#         if 'start_date' in habit_details and habit_details['start_date']:
#             habit_details['start_date'] = habit_details['start_date'].isoformat()
            
#         # Convert progress dates
#         for progress in recent_progress:
#             if 'date' in progress and progress['date']:
#                 progress['date'] = progress['date'].isoformat()
        
#         cursor.close()
#         conn.close()
        
#         return jsonify({
#             'success': True,
#             'habit': habit_details,
#             'recent_progress': recent_progress
#         })
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

@habits_bp.route('/api/habits/habit-details/<int:habit_id>')
@login_required
def get_habit_details(habit_id):
    """Get detailed information about a specific habit"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)  # Use DictCursor
        
        cursor.execute("""
            SELECT h.*, 
                   uh.id as user_habit_id,
                   uh.custom_target_count,
                   uh.reminder_time,
                   uh.start_date,
                   -- Last 30 days analytics
                   (SELECT AVG(ht.completion_percentage) 
                    FROM habit_tracking ht 
                    WHERE ht.user_habit_id = uh.id 
                    AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as avg_completion_30d,
                   (SELECT COUNT(*) 
                    FROM habit_tracking ht 
                    WHERE ht.user_habit_id = uh.id 
                    AND ht.is_completed = 1 
                    AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as completed_days_30d,
                   (SELECT MAX(ht.streak_day) 
                    FROM habit_tracking ht 
                    WHERE ht.user_habit_id = uh.id) as best_streak
            FROM eye_habits h
            LEFT JOIN user_habits uh ON h.id = uh.habit_id AND uh.user_id = %s AND uh.is_active = 1
            WHERE h.id = %s
        """, (current_user.id, habit_id))
        
        habit_details = cursor.fetchone()
        
        if not habit_details:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Habit not found'})
        
        # Get last 7 days progress
        cursor.execute("""
            SELECT ht.date, ht.completion_percentage, ht.is_completed, ht.mood_after
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            WHERE uh.user_id = %s AND uh.habit_id = %s 
            AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
            ORDER BY ht.date DESC
        """, (current_user.id, habit_id))
        
        recent_progress = cursor.fetchall()
        
        # Convert datetime objects to strings for JSON serialization
        if 'created_at' in habit_details and habit_details['created_at']:
            habit_details['created_at'] = habit_details['created_at'].isoformat()
        if 'reminder_time' in habit_details and habit_details['reminder_time']:
            habit_details['reminder_time'] = str(habit_details['reminder_time'])
        if 'start_date' in habit_details and habit_details['start_date']:
            habit_details['start_date'] = habit_details['start_date'].isoformat()
            
        # Convert progress dates
        for progress in recent_progress:
            if 'date' in progress and progress['date']:
                progress['date'] = progress['date'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'habit': habit_details,
            'recent_progress': recent_progress
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    

@habits_bp.route('/api/habits/update-reminder', methods=['POST'])
@login_required
def update_habit_reminder():
    """Update reminder settings for a habit"""
    try:
        data = request.get_json()
        user_habit_id = data.get('user_habit_id')
        reminder_time = data.get('reminder_time')
        reminder_enabled = data.get('reminder_enabled', True)
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE user_habits 
            SET reminder_time = %s, reminder_enabled = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
        """, (reminder_time, reminder_enabled, user_habit_id, current_user.id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Reminder updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    

    # Add this to routes/habits.py

@habits_bp.route('/api/habits/smart-suggestions')
@login_required
def get_smart_habit_suggestions():
    """Get AI-powered habit suggestions based on user's recommendations and risk factors"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get user's latest risk factors and recommendations
        cursor.execute("""
            SELECT ued.risk_factors, ued.risk_score, ued.average_screen_time, 
                   ued.blue_light_filter, ued.discomfort_eye_strain, ued.dry_eye_disease
            FROM user_eye_health_data ued
            WHERE ued.user_id = %s
            ORDER BY ued.created_at DESC
            LIMIT 1
        """, (current_user.id,))
        
        user_data = cursor.fetchone()
        
        if not user_data:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'No user data found'})
        
        # Get user's current recommendations
        cursor.execute("""
            SELECT recommendation_text, category, priority
            FROM user_recommendations
            WHERE user_id = %s AND status IN ('pending', 'in_progress')
            ORDER BY created_at DESC
        """, (current_user.id,))
        
        recommendations = cursor.fetchall()
        
        # Get available habits not yet added by user
        cursor.execute("""
            SELECT h.*
            FROM eye_habits h
            LEFT JOIN user_habits uh ON h.id = uh.habit_id AND uh.user_id = %s AND uh.is_active = 1
            WHERE h.is_active = 1 AND uh.id IS NULL
            ORDER BY h.category, h.name
        """, (current_user.id,))
        
        available_habits = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Generate smart suggestions
        suggestions = generate_habit_suggestions(user_data, recommendations, available_habits)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'user_risk_score': float(user_data['risk_score']) if user_data['risk_score'] else 0.0
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_habit_suggestions(user_data, recommendations, available_habits):
    """Generate personalized habit suggestions based on user data"""
    suggestions = []
    
    # Parse risk factors
    import json
    risk_factors = json.loads(user_data['risk_factors']) if user_data['risk_factors'] else []
    
    # Mapping of risk factors to habit suggestions
    risk_to_habits = {
        'High Screen Time': ['20-20-20 Rule', 'Blue Light Break', 'Blinking Exercises'],
        'No Blue Light Filter': ['Blue Light Break', '20-20-20 Rule'],
        'Moderate Sleep Quality': ['Sleep Quality', 'Humidifier Use'],
        'Digital Eye Strain': ['20-20-20 Rule', 'Blinking Exercises', 'Eye Massage'],
        'Dry Eye': ['Water Intake', 'Humidifier Use', 'Omega-3 Intake', 'Blinking Exercises']
    }
    
    # Recommendation text to habit mapping
    rec_to_habits = {
        'screen time': ['20-20-20 Rule', 'Blue Light Break'],
        'blue light': ['Blue Light Break', '20-20-20 Rule'],
        'sleep': ['Sleep Quality'],
        'hydrat': ['Water Intake'],
        'humid': ['Humidifier Use'],
        'break': ['20-20-20 Rule', 'Blue Light Break'],
        'exercise': ['Eye Massage', 'Blinking Exercises'],
        'omega': ['Omega-3 Intake']
    }
    
    suggested_habit_names = set()
    
    # Suggest based on risk factors
    for risk_factor in risk_factors:
        factor_name = risk_factor.get('factor', '')
        for risk_key, habits in risk_to_habits.items():
            if risk_key.lower() in factor_name.lower():
                suggested_habit_names.update(habits)
    
    # Suggest based on recommendations
    for rec in recommendations:
        rec_text = rec['recommendation_text'].lower()
        for keyword, habits in rec_to_habits.items():
            if keyword in rec_text:
                suggested_habit_names.update(habits)
    
    # High priority suggestions based on specific conditions
    if user_data['average_screen_time'] and float(user_data['average_screen_time']) > 8:
        suggested_habit_names.update(['20-20-20 Rule', 'Blue Light Break', 'Blinking Exercises'])
    
    if user_data['blue_light_filter'] == 'N':
        suggested_habit_names.update(['Blue Light Break', '20-20-20 Rule'])
    
    if user_data['discomfort_eye_strain'] == 'Y':
        suggested_habit_names.update(['20-20-20 Rule', 'Eye Massage', 'Blinking Exercises'])
    
    if user_data['dry_eye_disease'] == 'Y':
        suggested_habit_names.update(['Water Intake', 'Humidifier Use', 'Omega-3 Intake', 'Blinking Exercises'])
    
    # Match suggested names with available habits
    for habit in available_habits:
        if habit['name'] in suggested_habit_names:
            # Calculate recommendation score
            score = calculate_habit_score(habit, user_data, risk_factors)
            
            suggestions.append({
                'habit': habit,
                'reason': generate_suggestion_reason(habit, user_data, risk_factors),
                'score': score,
                'urgency': determine_urgency(habit, user_data)
            })
    
    # Sort by score (highest first)
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    
    return suggestions[:6]  # Return top 6 suggestions

def calculate_habit_score(habit, user_data, risk_factors):
    """Calculate a recommendation score for a habit (0-100)"""
    score = 50  # Base score
    
    # Increase score based on relevant risk factors
    for risk_factor in risk_factors:
        factor_name = risk_factor.get('factor', '').lower()
        impact = risk_factor.get('impact', 'medium')
        
        impact_multiplier = {'high': 20, 'medium': 10, 'low': 5}.get(impact, 10)
        
        if 'screen time' in factor_name and habit['category'] == 'screen_health':
            score += impact_multiplier
        elif 'sleep' in factor_name and habit['category'] == 'sleep':
            score += impact_multiplier
        elif 'blue light' in factor_name and 'blue light' in habit['name'].lower():
            score += impact_multiplier
    
    # Boost score for easy habits (better adoption)
    if habit['difficulty_level'] == 'easy':
        score += 10
    elif habit['difficulty_level'] == 'hard':
        score -= 5
    
    # Boost score for quick habits
    if habit['estimated_time_minutes'] <= 5:
        score += 10
    
    # Specific condition boosts
    if user_data.get('dry_eye_disease') == 'Y' and 'hydration' in habit['category']:
        score += 15
    
    if user_data.get('average_screen_time') and float(user_data['average_screen_time']) > 8:
        if '20-20-20' in habit['name'] or 'break' in habit['name'].lower():
            score += 20
    
    return min(score, 100)

def generate_suggestion_reason(habit, user_data, risk_factors):
    """Generate a personalized reason for suggesting this habit"""
    reasons = []
    
    # Check specific conditions
    if user_data.get('average_screen_time') and float(user_data['average_screen_time']) > 8:
        if 'screen' in habit['category'] or '20-20-20' in habit['name']:
            reasons.append(f"Your {user_data['average_screen_time']} hours of daily screen time increases eye strain risk")
    
    if user_data.get('blue_light_filter') == 'N' and 'blue light' in habit['name'].lower():
        reasons.append("You're not currently using blue light protection")
    
    if user_data.get('dry_eye_disease') == 'Y' and habit['category'] in ['hydration', 'exercise']:
        reasons.append("This habit specifically helps with dry eye symptoms")
    
    # Check risk factors
    for risk_factor in risk_factors:
        factor_name = risk_factor.get('factor', '')
        if 'Screen Time' in factor_name and habit['category'] == 'screen_health':
            reasons.append("Addresses your high screen time risk factor")
        elif 'Sleep Quality' in factor_name and habit['category'] == 'sleep':
            reasons.append("Helps improve your sleep quality concerns")
    
    if not reasons:
        reasons.append("Recommended based on your overall eye health profile")
    
    return reasons[0]  # Return the first/most relevant reason

def determine_urgency(habit, user_data):
    """Determine urgency level for a habit"""
    if user_data.get('dry_eye_disease') == 'Y' and habit['category'] in ['hydration', 'exercise']:
        return 'high'
    
    if user_data.get('average_screen_time') and float(user_data['average_screen_time']) > 10:
        if 'screen' in habit['category']:
            return 'high'
    
    if habit['difficulty_level'] == 'easy' and habit['estimated_time_minutes'] <= 3:
        return 'low'  # Easy to start
    
    return 'medium'

# Add this route for habit analytics
@habits_bp.route('/api/habits/dashboard-stats')
@login_required
def get_habit_dashboard_stats():
    """Get comprehensive stats for the habits dashboard"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        today = date.today()
        week_start = today - timedelta(days=6)
        
        # Today's completion stats
        cursor.execute("""
            SELECT 
                COUNT(ht.id) as total_habits_today,
                COUNT(CASE WHEN ht.is_completed = 1 THEN 1 END) as completed_today,
                AVG(ht.completion_percentage) as avg_completion_today
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            WHERE uh.user_id = %s AND ht.date = %s
        """, (current_user.id, today))
        
        today_stats = cursor.fetchone()
        
        # Weekly average
        cursor.execute("""
            SELECT 
                AVG(daily_completion) as weekly_avg
            FROM (
                SELECT AVG(ht.completion_percentage) as daily_completion
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s AND ht.date >= %s
                GROUP BY ht.date
            ) daily_averages
        """, (current_user.id, week_start))
        
        weekly_avg = cursor.fetchone()
        
        # Longest streak
        cursor.execute("""
            SELECT MAX(ht.streak_day) as longest_streak
            FROM habit_tracking ht
            JOIN user_habits uh ON ht.user_habit_id = uh.id
            WHERE uh.user_id = %s
        """, (current_user.id,))
        
        streak_stats = cursor.fetchone()
        
        # Active habits count
        cursor.execute("""
            SELECT COUNT(*) as active_habits
            FROM user_habits
            WHERE user_id = %s AND is_active = 1
        """, (current_user.id,))
        
        habits_count = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'today_completed': today_stats['completed_today'] if today_stats else 0,
                'today_total': today_stats['total_habits_today'] if today_stats else 0,
                'weekly_average': round(weekly_avg['weekly_avg'] if weekly_avg and weekly_avg['weekly_avg'] else 0),
                'longest_streak': streak_stats['longest_streak'] if streak_stats else 0,
                'total_active_habits': habits_count['active_habits'] if habits_count else 0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
