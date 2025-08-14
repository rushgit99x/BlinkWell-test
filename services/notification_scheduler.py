import schedule
import time
import threading
from datetime import datetime, timedelta
from flask import current_app
import MySQLdb
import logging
from .email_service import email_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationScheduler:
    def __init__(self, app=None):
        self.app = app
        self.running = False
        self.scheduler_thread = None
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the notification scheduler with Flask app"""
        self.app = app
        self.email_service = email_service
        self.email_service.init_app(app)
    
    def start(self):
        """Start the notification scheduler in a background thread"""
        if self.running:
            logger.info("Notification scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Notification scheduler started")
    
    def stop(self):
        """Stop the notification scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Notification scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        # Schedule daily habit reminders
        schedule.every().day.at("09:00").do(self.send_daily_habit_reminders)
        schedule.every().day.at("14:00").do(self.send_afternoon_reminders)
        schedule.every().day.at("18:00").do(self.send_evening_reminders)
        
        # Schedule weekly progress reports (every Sunday at 10:00 AM)
        schedule.every().sunday.at("10:00").do(self.send_weekly_progress_reports)
        
        # Schedule monthly recommendation emails (first day of month at 9:00 AM)
        schedule.every().month.at("09:00").do(self.send_monthly_recommendations)
        
        logger.info("Scheduled notifications:")
        logger.info("- Daily habit reminders: 9:00 AM, 2:00 PM, 6:00 PM")
        logger.info("- Weekly progress reports: Sundays at 10:00 AM")
        logger.info("- Monthly recommendations: First day of month at 9:00 AM")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def send_daily_habit_reminders(self):
        """Send morning habit reminders to users"""
        logger.info("Sending daily habit reminders...")
        try:
            with self.app.app_context():
                conn = current_app.config['get_db_connection']()
                cursor = conn.cursor(MySQLdb.cursors.DictCursor)
                
                # Get users with active habits and morning reminders enabled
                cursor.execute("""
                    SELECT DISTINCT 
                        u.id, u.username, u.email,
                        uh.reminder_time, uh.reminder_enabled,
                        h.name as habit_name
                    FROM users u
                    JOIN user_habits uh ON u.id = uh.user_id
                    JOIN eye_habits h ON uh.habit_id = h.id
                    WHERE uh.is_active = 1 
                    AND uh.reminder_enabled = 1
                    AND TIME(uh.reminder_time) <= '10:00:00'
                    AND TIME(uh.reminder_time) >= '08:00:00'
                """)
                
                users = cursor.fetchall()
                cursor.close()
                conn.close()
                
                for user in users:
                    try:
                        # Get user's current streak for this habit
                        streak = self._get_user_habit_streak(user['id'], user['habit_name'])
                        
                        self.email_service.send_habit_reminder_email(
                            user_email=user['email'],
                            username=user['username'],
                            habit_name=user['habit_name'],
                            reminder_time=user['reminder_time'].strftime('%I:%M %p') if user['reminder_time'] else '9:00 AM',
                            streak_days=streak
                        )
                        logger.info(f"Sent morning reminder to {user['email']} for {user['habit_name']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send reminder to {user['email']}: {str(e)}")
                
                logger.info(f"Sent {len(users)} daily habit reminders")
                
        except Exception as e:
            logger.error(f"Error sending daily habit reminders: {str(e)}")
    
    def send_afternoon_reminders(self):
        """Send afternoon habit reminders to users"""
        logger.info("Sending afternoon habit reminders...")
        try:
            with self.app.app_context():
                conn = current_app.config['get_db_connection']()
                cursor = conn.cursor(MySQLdb.cursors.DictCursor)
                
                # Get users with active habits and afternoon reminders
                cursor.execute("""
                    SELECT DISTINCT 
                        u.id, u.username, u.email,
                        uh.reminder_time, uh.reminder_enabled,
                        h.name as habit_name
                    FROM users u
                    JOIN user_habits uh ON u.id = uh.user_id
                    JOIN eye_habits h ON uh.habit_id = h.id
                    WHERE uh.is_active = 1 
                    AND uh.reminder_enabled = 1
                    AND TIME(uh.reminder_time) <= '16:00:00'
                    AND TIME(uh.reminder_time) >= '13:00:00'
                """)
                
                users = cursor.fetchall()
                cursor.close()
                conn.close()
                
                for user in users:
                    try:
                        streak = self._get_user_habit_streak(user['id'], user['habit_name'])
                        
                        self.email_service.send_habit_reminder_email(
                            user_email=user['email'],
                            username=user['username'],
                            habit_name=user['habit_name'],
                            reminder_time=user['reminder_time'].strftime('%I:%M %p') if user['reminder_time'] else '2:00 PM',
                            streak_days=streak
                        )
                        logger.info(f"Sent afternoon reminder to {user['email']} for {user['habit_name']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send reminder to {user['email']}: {str(e)}")
                
                logger.info(f"Sent {len(users)} afternoon habit reminders")
                
        except Exception as e:
            logger.error(f"Error sending afternoon habit reminders: {str(e)}")
    
    def send_evening_reminders(self):
        """Send evening habit reminders to users"""
        logger.info("Sending evening habit reminders...")
        try:
            with self.app.app_context():
                conn = current_app.config['get_db_connection']()
                cursor = conn.cursor(MySQLdb.cursors.DictCursor)
                
                # Get users with active habits and evening reminders
                cursor.execute("""
                    SELECT DISTINCT 
                        u.id, u.username, u.email,
                        uh.reminder_time, uh.reminder_enabled,
                        h.name as habit_name
                    FROM users u
                    JOIN user_habits uh ON u.id = uh.user_id
                    JOIN eye_habits h ON uh.habit_id = h.id
                    WHERE uh.is_active = 1 
                    AND uh.reminder_enabled = 1
                    AND TIME(uh.reminder_time) <= '20:00:00'
                    AND TIME(uh.reminder_time) >= '17:00:00'
                """)
                
                users = cursor.fetchall()
                cursor.close()
                conn.close()
                
                for user in users:
                    try:
                        streak = self._get_user_habit_streak(user['id'], user['habit_name'])
                        
                        self.email_service.send_habit_reminder_email(
                            user_email=user['email'],
                            username=user['username'],
                            habit_name=user['habit_name'],
                            reminder_time=user['reminder_time'].strftime('%I:%M %p') if user['reminder_time'] else '6:00 PM',
                            streak_days=streak
                        )
                        logger.info(f"Sent evening reminder to {user['email']} for {user['habit_name']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send reminder to {user['email']}: {str(e)}")
                
                logger.info(f"Sent {len(users)} evening habit reminders")
                
        except Exception as e:
            logger.error(f"Error sending evening habit reminders: {str(e)}")
    
    def send_weekly_progress_reports(self):
        """Send weekly progress reports to all active users"""
        logger.info("Sending weekly progress reports...")
        try:
            with self.app.app_context():
                conn = current_app.config['get_db_connection']()
                cursor = conn.cursor(MySQLdb.cursors.DictCursor)
                
                # Get all active users
                cursor.execute("SELECT id, username, email FROM users WHERE id > 0")
                users = cursor.fetchall()
                cursor.close()
                conn.close()
                
                for user in users:
                    try:
                        # Get user's weekly stats
                        weekly_stats = self._get_user_weekly_stats(user['id'])
                        
                        if weekly_stats.get('habits_completed', 0) > 0:
                            self.email_service.send_weekly_progress_email(
                                user_email=user['email'],
                                username=user['username'],
                                weekly_stats=weekly_stats
                            )
                            logger.info(f"Sent weekly progress report to {user['email']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send weekly report to {user['email']}: {str(e)}")
                
                logger.info(f"Sent {len(users)} weekly progress reports")
                
        except Exception as e:
            logger.error(f"Error sending weekly progress reports: {str(e)}")
    
    def send_monthly_recommendations(self):
        """Send monthly personalized recommendations to users"""
        logger.info("Sending monthly recommendations...")
        try:
            with self.app.app_context():
                conn = current_app.config['get_db_connection']()
                cursor = conn.cursor(MySQLdb.cursors.DictCursor)
                
                # Get all active users
                cursor.execute("SELECT id, username, email FROM users WHERE id > 0")
                users = cursor.fetchall()
                cursor.close()
                conn.close()
                
                for user in users:
                    try:
                        # Get personalized recommendations
                        recommendations = self._get_user_recommendations(user['id'])
                        
                        if recommendations:
                            # Get user's current habits for context
                            user_habits = self._get_user_active_habits(user['id'])
                            
                            self.email_service.send_recommendation_email(
                                user_email=user['email'],
                                username=user['username'],
                                recommendations=recommendations,
                                user_habits=user_habits
                            )
                            logger.info(f"Sent monthly recommendations to {user['email']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send recommendations to {user['email']}: {str(e)}")
                
                logger.info(f"Sent {len(users)} monthly recommendation emails")
                
        except Exception as e:
            logger.error(f"Error sending monthly recommendations: {str(e)}")
    
    def _get_user_habit_streak(self, user_id, habit_name):
        """Get user's current streak for a specific habit"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(DISTINCT ht.date) as streak
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                JOIN eye_habits h ON uh.habit_id = h.id
                WHERE uh.user_id = %s AND h.name = %s 
                AND ht.is_completed = 1
                AND ht.date <= CURDATE()
                AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                ORDER BY ht.date DESC
            """, (user_id, habit_name))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting habit streak: {str(e)}")
            return 0
    
    def _get_user_weekly_stats(self, user_id):
        """Get user's weekly habit completion statistics"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get this week's stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as habits_completed,
                    COUNT(DISTINCT ht.date) as days_with_habits,
                    MAX(ht.date) as last_completion_date
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s 
                AND ht.is_completed = 1
                AND ht.date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            """, (user_id,))
            
            weekly_stats = cursor.fetchone()
            
            # Get streak information
            cursor.execute("""
                SELECT 
                    MAX(streak_length) as longest_streak,
                    SUM(streak_length) as total_streak_days
                FROM (
                    SELECT 
                        COUNT(*) as streak_length
                    FROM habit_tracking ht
                    JOIN user_habits uh ON ht.user_habit_id = uh.id
                    WHERE uh.user_id = %s 
                    AND ht.is_completed = 1
                    AND ht.date <= CURDATE()
                    GROUP BY DATE_SUB(ht.date, INTERVAL ROW_NUMBER() OVER (ORDER BY ht.date) DAY)
                ) as streaks
            """, (user_id,))
            
            streak_info = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if weekly_stats:
                weekly_stats['completion_rate'] = (weekly_stats['habits_completed'] / 7) * 100 if weekly_stats['days_with_habits'] > 0 else 0
                weekly_stats['best_day'] = weekly_stats['last_completion_date'].strftime('%A') if weekly_stats['last_completion_date'] else 'N/A'
                weekly_stats['streak_info'] = streak_info or {}
            
            return weekly_stats or {}
            
        except Exception as e:
            logger.error(f"Error getting weekly stats: {str(e)}")
            return {}
    
    def _get_user_recommendations(self, user_id):
        """Get personalized recommendations for a user"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get user's habit completion patterns
            cursor.execute("""
                SELECT 
                    h.category,
                    COUNT(*) as completion_count,
                    AVG(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completion_rate
                FROM user_habits uh
                JOIN eye_habits h ON uh.habit_id = h.id
                LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id
                WHERE uh.user_id = %s AND uh.is_active = 1
                GROUP BY h.category
            """, (user_id,))
            
            habit_patterns = cursor.fetchall()
            cursor.close()
            conn.close()
            
            recommendations = []
            
            # Generate recommendations based on patterns
            for pattern in habit_patterns:
                if pattern['completion_rate'] < 0.5:
                    recommendations.append({
                        'title': f"Improve your {pattern['category']} habits",
                        'description': f"Your {pattern['category']} habits have a {pattern['completion_rate']*100:.1f}% completion rate. Try setting smaller goals or adjusting reminder times to improve consistency.",
                        'based_on': f"{pattern['category']} habit completion rate"
                    })
                elif pattern['completion_rate'] >= 0.8:
                    recommendations.append({
                        'title': f"Great job with {pattern['category']} habits!",
                        'description': f"You're doing excellent with {pattern['category']} habits! Consider adding a new habit in this category to challenge yourself further.",
                        'based_on': f"high {pattern['category']} habit completion rate"
                    })
            
            # Add general recommendations
            if len(habit_patterns) < 3:
                recommendations.append({
                    'title': "Expand your eye health routine",
                    'description': "You have room to add more eye health habits. Consider exploring different categories to create a comprehensive eye care routine.",
                    'based_on': "limited habit variety"
                })
            
            recommendations.append({
                'title': "Stay consistent with your routine",
                'description': "Consistency is key to building healthy eye habits. Keep up the great work and remember that small daily actions lead to long-term benefits.",
                'based_on': "general eye health best practices"
            })
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return []
    
    def _get_user_active_habits(self, user_id):
        """Get user's currently active habits"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            cursor.execute("""
                SELECT 
                    h.name,
                    h.category,
                    (SELECT COUNT(DISTINCT ht2.date) 
                     FROM habit_tracking ht2 
                     WHERE ht2.user_habit_id = uh.id 
                     AND ht2.is_completed = 1 
                     AND ht2.date <= CURDATE()
                     AND ht2.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                    ) as streak_days
                FROM user_habits uh
                JOIN eye_habits h ON uh.habit_id = h.id
                WHERE uh.user_id = %s AND uh.is_active = 1
                ORDER BY h.category, h.name
            """, (user_id,))
            
            habits = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return habits
            
        except Exception as e:
            logger.error(f"Error getting user active habits: {str(e)}")
            return []
    
    def send_immediate_reminder(self, user_id, habit_name, reminder_time):
        """Send an immediate reminder for a specific habit"""
        try:
            with self.app.app_context():
                conn = current_app.config['get_db_connection']()
                cursor = conn.cursor(MySQLdb.cursors.DictCursor)
                
                cursor.execute("""
                    SELECT u.username, u.email
                    FROM users u
                    JOIN user_habits uh ON u.id = uh.user_id
                    JOIN eye_habits h ON uh.habit_id = h.id
                    WHERE u.id = %s AND h.name = %s
                """, (user_id, habit_name))
                
                user = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if user:
                    streak = self._get_user_habit_streak(user_id, habit_name)
                    
                    success = self.email_service.send_habit_reminder_email(
                        user_email=user['email'],
                        username=user['username'],
                        habit_name=habit_name,
                        reminder_time=reminder_time,
                        streak_days=streak
                    )
                    
                    if success:
                        logger.info(f"Sent immediate reminder to {user['email']} for {habit_name}")
                        return True
                    else:
                        logger.error(f"Failed to send immediate reminder to {user['email']}")
                        return False
                else:
                    logger.error(f"User {user_id} or habit {habit_name} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending immediate reminder: {str(e)}")
            return False

# Create a global instance
notification_scheduler = NotificationScheduler()