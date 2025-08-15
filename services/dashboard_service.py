import MySQLdb
from datetime import datetime, timedelta, date
from flask import current_app
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardService:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the dashboard service with Flask app"""
        self.app = app
    
    def get_dashboard_data(self, user_id):
        """Get comprehensive dashboard data for a user"""
        try:
            dashboard_data = {
                'risk_metrics': self._get_risk_metrics(user_id),
                'habit_metrics': self._get_habit_metrics(user_id),
                'streak_metrics': self._get_streak_metrics(user_id),
                'progress_metrics': self._get_progress_metrics(user_id),
                'today_habits': self._get_today_habits(user_id),
                'weekly_progress': self._get_weekly_progress(user_id),
                'risk_trends': self._get_risk_trends(user_id),
                'water_intake': self._get_water_intake(user_id),
                'eye_exercises': self._get_eye_exercises(user_id)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data for user {user_id}: {str(e)}")
            return self._get_default_dashboard_data()
    
    def _get_risk_metrics(self, user_id):
        """Get current risk score and risk reduction metrics"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get current risk score from latest assessment
            cursor.execute("""
                SELECT risk_score, assessment_date
                FROM user_eye_health_data 
                WHERE user_id = %s 
                ORDER BY assessment_date DESC 
                LIMIT 1
            """, (user_id,))
            
            current_risk = cursor.fetchone()
            
            # Get previous month's risk score for comparison
            cursor.execute("""
                SELECT risk_score, assessment_date
                FROM user_eye_health_data 
                WHERE user_id = %s 
                AND assessment_date < DATE_SUB(CURDATE(), INTERVAL 15 DAY)
                ORDER BY assessment_date DESC 
                LIMIT 1
            """, (user_id,))
            
            previous_risk = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            current_score = current_risk['risk_score'] if current_risk else 8.0
            previous_score = previous_risk['risk_score'] if previous_risk else 8.0
            
            # Calculate risk reduction
            risk_reduction = max(0, ((previous_score - current_score) / previous_score) * 100)
            score_change = previous_score - current_score
            
            return {
                'current_risk_score': round(current_score, 1),
                'previous_risk_score': round(previous_score, 1),
                'risk_reduction': round(risk_reduction, 1),
                'score_change': round(score_change, 1),
                'trend': 'improving' if score_change > 0 else 'stable' if score_change == 0 else 'increasing',
                'last_assessment': current_risk['assessment_date'] if current_risk else None
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {
                'current_risk_score': 8.0,
                'previous_risk_score': 8.0,
                'risk_reduction': 0.0,
                'score_change': 0.0,
                'trend': 'stable',
                'last_assessment': None
            }
    
    def _get_habit_metrics(self, user_id):
        """Get habit completion metrics"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get today's habit completion
            today = date.today()
            cursor.execute("""
                SELECT COUNT(*) as completed_today
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s 
                AND ht.date = %s 
                AND ht.is_completed = 1
            """, (user_id, today))
            
            today_completed = cursor.fetchone()['completed_today']
            
            # Get total active habits
            cursor.execute("""
                SELECT COUNT(*) as total_habits
                FROM user_habits 
                WHERE user_id = %s AND is_active = 1
            """, (user_id,))
            
            total_habits = cursor.fetchone()['total_habits']
            
            # Get weekly habit completion
            week_start = today - timedelta(days=today.weekday())
            cursor.execute("""
                SELECT COUNT(*) as completed_this_week
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s 
                AND ht.date >= %s 
                AND ht.is_completed = 1
            """, (user_id, week_start))
            
            weekly_completed = cursor.fetchone()['completed_this_week']
            
            cursor.close()
            conn.close()
            
            # Calculate completion percentages
            daily_completion = (today_completed / max(total_habits, 1)) * 100 if total_habits > 0 else 0
            weekly_completion = (weekly_completed / max(total_habits * 7, 1)) * 100 if total_habits > 0 else 0
            
            return {
                'habits_completed_today': today_completed,
                'total_active_habits': total_habits,
                'daily_completion_percentage': round(daily_completion, 1),
                'weekly_completion_percentage': round(weekly_completion, 1),
                'weekly_completed': weekly_completed
            }
            
        except Exception as e:
            logger.error(f"Error getting habit metrics: {str(e)}")
            return {
                'habits_completed_today': 0,
                'total_active_habits': 0,
                'daily_completion_percentage': 0.0,
                'weekly_completion_percentage': 0.0,
                'weekly_completed': 0
            }
    
    def _get_streak_metrics(self, user_id):
        """Get current streak information"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get current streak (consecutive days with at least one habit completed)
            cursor.execute("""
                WITH RECURSIVE dates AS (
                    SELECT CURDATE() as date
                    UNION ALL
                    SELECT DATE_SUB(date, INTERVAL 1 DAY)
                    FROM dates
                    WHERE date > DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                ),
                completed_dates AS (
                    SELECT DISTINCT ht.date
                    FROM habit_tracking ht
                    JOIN user_habits uh ON ht.user_habit_id = uh.id
                    WHERE uh.user_id = %s AND ht.is_completed = 1
                )
                SELECT COUNT(*) as current_streak
                FROM (
                    SELECT d.date,
                           CASE WHEN cd.date IS NOT NULL THEN 1 ELSE 0 END as completed
                    FROM dates d
                    LEFT JOIN completed_dates cd ON d.date = cd.date
                    ORDER BY d.date DESC
                ) ranked_dates
                WHERE completed = 1
                AND (
                    SELECT completed 
                    FROM (
                        SELECT d.date,
                               CASE WHEN cd.date IS NOT NULL THEN 1 ELSE 0 END as completed
                        FROM dates d
                        LEFT JOIN completed_dates cd ON d.date = cd.date
                        ORDER BY d.date DESC
                    ) ranked_dates2
                    WHERE ranked_dates2.date = DATE_SUB(ranked_dates.date, INTERVAL 1 DAY)
                    LIMIT 1
                ) = 1
            """, (user_id,))
            
            current_streak = cursor.fetchone()['current_streak']
            
            # Get longest streak
            cursor.execute("""
                SELECT MAX(streak_length) as longest_streak
                FROM (
                    SELECT COUNT(*) as streak_length
                    FROM (
                        SELECT date,
                               CASE WHEN completed = 1 THEN @streak := @streak + 1
                                    ELSE @streak := 0 END as streak
                        FROM (
                            SELECT d.date,
                                   CASE WHEN cd.date IS NOT NULL THEN 1 ELSE 0 END as completed
                            FROM (
                                SELECT DATE_SUB(CURDATE(), INTERVAL n DAY) as date
                                FROM (
                                    SELECT 0 as n UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                                    UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9
                                    UNION SELECT 10 UNION SELECT 11 UNION SELECT 12 UNION SELECT 13 UNION SELECT 14
                                    UNION SELECT 15 UNION SELECT 16 UNION SELECT 17 UNION SELECT 18 UNION SELECT 19
                                    UNION SELECT 20 UNION SELECT 21 UNION SELECT 22 UNION SELECT 23 UNION SELECT 24
                                    UNION SELECT 25 UNION SELECT 26 UNION SELECT 27 UNION SELECT 28 UNION SELECT 29
                                ) numbers
                            ) d
                            LEFT JOIN (
                                SELECT DISTINCT ht.date
                                FROM habit_tracking ht
                                JOIN user_habits uh ON ht.user_habit_id = uh.id
                                WHERE uh.user_id = %s AND ht.is_completed = 1
                            ) cd ON d.date = cd.date
                            ORDER BY d.date DESC
                        ) dates_with_completion,
                        (SELECT @streak := 0) vars
                    ) streak_calculation
                    WHERE completed = 1
                    GROUP BY streak
                ) streaks
            """, (user_id,))
            
            longest_streak = cursor.fetchone()['longest_streak'] or 0
            
            cursor.close()
            conn.close()
            
            return {
                'current_streak': current_streak,
                'longest_streak': longest_streak,
                'streak_status': 'active' if current_streak > 0 else 'broken'
            }
            
        except Exception as e:
            logger.error(f"Error getting streak metrics: {str(e)}")
            return {
                'current_streak': 0,
                'longest_streak': 0,
                'streak_status': 'broken'
            }
    
    def _get_progress_metrics(self, user_id):
        """Get overall progress metrics"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get this week's progress
            week_start = date.today() - timedelta(days=date.today().weekday())
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_habits_this_week,
                    SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_habits_this_week
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s 
                AND ht.date >= %s
            """, (user_id, week_start))
            
            week_data = cursor.fetchone()
            
            # Get this month's progress
            month_start = date.today().replace(day=1)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_habits_this_month,
                    SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_habits_this_month
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s 
                AND ht.date >= %s
            """, (user_id, month_start))
            
            month_data = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            weekly_completion = (week_data['completed_habits_this_week'] / max(week_data['total_habits_this_week'], 1)) * 100 if week_data['total_habits_this_week'] > 0 else 0
            monthly_completion = (month_data['completed_habits_this_month'] / max(month_data['total_habits_this_month'], 1)) * 100 if month_data['total_habits_this_month'] > 0 else 0
            
            return {
                'weekly_completion': round(weekly_completion, 1),
                'monthly_completion': round(monthly_completion, 1),
                'weekly_goal': 85.0,  # Default goal
                'monthly_goal': 80.0   # Default goal
            }
            
        except Exception as e:
            logger.error(f"Error getting progress metrics: {str(e)}")
            return {
                'weekly_completion': 0.0,
                'monthly_completion': 0.0,
                'weekly_goal': 85.0,
                'monthly_goal': 80.0
            }
    
    def _get_today_habits(self, user_id):
        """Get today's specific habits and their status"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            today = date.today()
            cursor.execute("""
                SELECT 
                    h.name,
                    h.description,
                    h.icon,
                    ht.is_completed,
                    ht.completed_count,
                    h.target_count,
                    h.target_unit
                FROM user_habits uh
                JOIN eye_habits h ON uh.habit_id = h.id
                LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
                WHERE uh.user_id = %s AND uh.is_active = 1
                ORDER BY h.category, h.name
            """, (today, user_id))
            
            habits = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Process habits for display
            today_habits = []
            for habit in habits:
                status = 'completed' if habit['is_completed'] else 'pending'
                progress = (habit['completed_count'] or 0) / max(habit['target_count'], 1) * 100
                
                today_habits.append({
                    'name': habit['name'],
                    'description': habit['description'],
                    'icon': habit['icon'],
                    'status': status,
                    'progress': round(progress, 1),
                    'completed_count': habit['completed_count'] or 0,
                    'target_count': habit['target_count'],
                    'target_unit': habit['target_unit']
                })
            
            return today_habits
            
        except Exception as e:
            logger.error(f"Error getting today's habits: {str(e)}")
            return []
    
    def _get_weekly_progress(self, user_id):
        """Get weekly progress breakdown"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get last 7 days progress
            week_start = date.today() - timedelta(days=6)
            cursor.execute("""
                SELECT 
                    ht.date,
                    COUNT(*) as total_habits,
                    SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_habits
                FROM habit_tracking ht
                JOIN user_habits uh ON ht.user_habit_id = uh.id
                WHERE uh.user_id = %s 
                AND ht.date >= %s
                GROUP BY ht.date
                ORDER BY ht.date
            """, (user_id, week_start))
            
            week_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Create weekly progress data
            weekly_progress = []
            for i in range(7):
                current_date = week_start + timedelta(days=i)
                day_data = next((day for day in week_data if day['date'] == current_date), None)
                
                if day_data:
                    completion_rate = (day_data['completed_habits'] / max(day_data['total_habits'], 1)) * 100
                else:
                    completion_rate = 0
                
                weekly_progress.append({
                    'date': current_date,
                    'day_name': current_date.strftime('%a'),
                    'completion_rate': round(completion_rate, 1),
                    'is_today': current_date == date.today()
                })
            
            return weekly_progress
            
        except Exception as e:
            logger.error(f"Error getting weekly progress: {str(e)}")
            return []
    
    def _get_risk_trends(self, user_id):
        """Get risk score trends over time"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Get risk scores for last 30 days
            cursor.execute("""
                SELECT risk_score, assessment_date
                FROM user_eye_health_data 
                WHERE user_id = %s 
                AND assessment_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                ORDER BY assessment_date
            """, (user_id,))
            
            risk_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not risk_data:
                return {
                    'trend': 'stable',
                    'change_amount': 0.0,
                    'data_points': []
                }
            
            # Calculate trend
            first_score = risk_data[0]['risk_score']
            last_score = risk_data[-1]['risk_score']
            change_amount = last_score - first_score
            
            if change_amount < -0.5:
                trend = 'improving'
            elif change_amount > 0.5:
                trend = 'increasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'change_amount': round(change_amount, 1),
                'data_points': [
                    {
                        'date': data['assessment_date'].strftime('%Y-%m-%d'),
                        'score': data['risk_score']
                    } for data in risk_data
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting risk trends: {str(e)}")
            return {
                'trend': 'stable',
                'change_amount': 0.0,
                'data_points': []
            }
    
    def _get_water_intake(self, user_id):
        """Get water intake tracking data"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Look for water intake habit
            cursor.execute("""
                SELECT 
                    uh.id as user_habit_id,
                    h.name,
                    h.target_count,
                    h.target_unit
                FROM user_habits uh
                JOIN eye_habits h ON uh.habit_id = h.id
                WHERE uh.user_id = %s 
                AND uh.is_active = 1
                AND (h.name LIKE '%water%' OR h.name LIKE '%hydration%' OR h.name LIKE '%glass%')
                LIMIT 1
            """, (user_id,))
            
            water_habit = cursor.fetchone()
            
            if not water_habit:
                # Return default water intake data
                return {
                    'target_glasses': 8,
                    'completed_glasses': 0,
                    'completion_percentage': 0.0,
                    'streak_days': 0,
                    'weekly_progress': []
                }
            
            # Get today's water intake
            today = date.today()
            cursor.execute("""
                SELECT completed_count
                FROM habit_tracking 
                WHERE user_habit_id = %s AND date = %s
            """, (water_habit['user_habit_id'], today))
            
            today_data = cursor.fetchone()
            completed_today = today_data['completed_count'] if today_data else 0
            
            # Get water intake streak
            cursor.execute("""
                SELECT COUNT(*) as streak_days
                FROM (
                    SELECT date,
                           CASE WHEN completed_count >= %s THEN 1 ELSE 0 END as met_goal
                    FROM habit_tracking 
                    WHERE user_habit_id = %s 
                    AND date <= CURDATE()
                    ORDER BY date DESC
                ) daily_goals
                WHERE met_goal = 1
                AND (
                    SELECT met_goal 
                    FROM (
                        SELECT date,
                               CASE WHEN completed_count >= %s THEN 1 ELSE 0 END as met_goal
                        FROM habit_tracking 
                        WHERE user_habit_id = %s 
                        AND date <= CURDATE()
                        ORDER BY date DESC
                    ) daily_goals2
                    WHERE daily_goals2.date = DATE_SUB(daily_goals.date, INTERVAL 1 DAY)
                    LIMIT 1
                ) = 1
            """, (water_habit['target_count'], water_habit['user_habit_id'], 
                  water_habit['target_count'], water_habit['user_habit_id']))
            
            streak_data = cursor.fetchone()
            streak_days = streak_data['streak_days'] if streak_data else 0
            
            # Get weekly water intake
            week_start = today - timedelta(days=today.weekday())
            cursor.execute("""
                SELECT date, completed_count
                FROM habit_tracking 
                WHERE user_habit_id = %s 
                AND date >= %s
                ORDER BY date
            """, (water_habit['user_habit_id'], week_start))
            
            weekly_data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            target_glasses = water_habit['target_count']
            completion_percentage = (completed_today / max(target_glasses, 1)) * 100
            
            # Create weekly progress for water intake
            weekly_progress = []
            for i in range(7):
                current_date = week_start + timedelta(days=i)
                day_data = next((day for day in weekly_data if day['date'] == current_date), None)
                
                if day_data:
                    glasses = day_data['completed_count']
                    status = 'completed' if glasses >= target_glasses else 'partial' if glasses > 0 else 'empty'
                else:
                    glasses = 0
                    status = 'empty'
                
                weekly_progress.append({
                    'date': current_date,
                    'day_name': current_date.strftime('%a'),
                    'glasses': glasses,
                    'status': status,
                    'is_today': current_date == today
                })
            
            return {
                'target_glasses': target_glasses,
                'completed_glasses': completed_today,
                'completion_percentage': round(completion_percentage, 1),
                'streak_days': streak_days,
                'weekly_progress': weekly_progress
            }
            
        except Exception as e:
            logger.error(f"Error getting water intake: {str(e)}")
            return {
                'target_glasses': 8,
                'completed_glasses': 0,
                'completion_percentage': 0.0,
                'streak_days': 0,
                'weekly_progress': []
            }
    
    def _get_eye_exercises(self, user_id):
        """Get eye exercise tracking data"""
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            # Look for eye exercise habit
            cursor.execute("""
                SELECT 
                    uh.id as user_habit_id,
                    h.name,
                    h.target_count,
                    h.target_unit
                FROM user_habits uh
                JOIN eye_habits h ON uh.habit_id = h.id
                WHERE uh.user_id = %s 
                AND uh.is_active = 1
                AND (h.name LIKE '%eye%' OR h.name LIKE '%exercise%' OR h.name LIKE '%20-20-20%')
                LIMIT 1
            """, (user_id,))
            
            exercise_habit = cursor.fetchone()
            
            if not exercise_habit:
                # Return default eye exercise data
                return {
                    'target_sessions': 3,
                    'completed_sessions': 0,
                    'completion_percentage': 0.0,
                    'streak_days': 0,
                    'weekly_progress': []
                }
            
            # Get today's eye exercises
            today = date.today()
            cursor.execute("""
                SELECT completed_count
                FROM habit_tracking 
                WHERE user_habit_id = %s AND date = %s
            """, (exercise_habit['user_habit_id'], today))
            
            today_data = cursor.fetchone()
            completed_today = today_data['completed_count'] if today_data else 0
            
            # Get eye exercise streak
            cursor.execute("""
                SELECT COUNT(*) as streak_days
                FROM (
                    SELECT date,
                           CASE WHEN completed_count >= %s THEN 1 ELSE 0 END as met_goal
                    FROM habit_tracking 
                    WHERE user_habit_id = %s 
                    AND date <= CURDATE()
                    ORDER BY date DESC
                ) daily_goals
                WHERE met_goal = 1
                AND (
                    SELECT met_goal 
                    FROM (
                        SELECT date,
                               CASE WHEN completed_count >= %s THEN 1 ELSE 0 END as met_goal
                        FROM habit_tracking 
                        WHERE user_habit_id = %s 
                        AND date <= CURDATE()
                        ORDER BY date DESC
                    ) daily_goals2
                    WHERE daily_goals2.date = DATE_SUB(daily_goals.date, INTERVAL 1 DAY)
                    LIMIT 1
                ) = 1
            """, (exercise_habit['target_count'], exercise_habit['user_habit_id'], 
                  exercise_habit['target_count'], exercise_habit['user_habit_id']))
            
            streak_data = cursor.fetchone()
            streak_days = streak_data['streak_days'] if streak_data else 0
            
            # Get weekly eye exercise progress
            week_start = today - timedelta(days=today.weekday())
            cursor.execute("""
                SELECT date, completed_count
                FROM habit_tracking 
                WHERE user_habit_id = %s 
                AND date >= %s
                ORDER BY date
            """, (exercise_habit['user_habit_id'], week_start))
            
            weekly_data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            target_sessions = exercise_habit['target_count']
            completion_percentage = (completed_today / max(target_sessions, 1)) * 100
            
            # Create weekly progress for eye exercises
            weekly_progress = []
            for i in range(7):
                current_date = week_start + timedelta(days=i)
                day_data = next((day for day in weekly_data if day['date'] == current_date), None)
                
                if day_data:
                    sessions = day_data['completed_count']
                    status = 'completed' if sessions >= target_sessions else 'partial' if sessions > 0 else 'empty'
                else:
                    sessions = 0
                    status = 'empty'
                
                weekly_progress.append({
                    'date': current_date,
                    'day_name': current_date.strftime('%a'),
                    'sessions': sessions,
                    'status': status,
                    'is_today': current_date == today
                })
            
            return {
                'target_sessions': target_sessions,
                'completed_sessions': completed_today,
                'completion_percentage': round(completion_percentage, 1),
                'streak_days': streak_days,
                'weekly_progress': weekly_progress
            }
            
        except Exception as e:
            logger.error(f"Error getting eye exercises: {str(e)}")
            return {
                'target_sessions': 3,
                'completed_sessions': 0,
                'completion_percentage': 0.0,
                'streak_days': 0,
                'weekly_progress': []
            }
    
    def _get_default_dashboard_data(self):
        """Return default dashboard data when database queries fail"""
        return {
            'risk_metrics': {
                'current_risk_score': 8.0,
                'previous_risk_score': 8.0,
                'risk_reduction': 0.0,
                'score_change': 0.0,
                'trend': 'stable',
                'last_assessment': None
            },
            'habit_metrics': {
                'habits_completed_today': 0,
                'total_active_habits': 0,
                'daily_completion_percentage': 0.0,
                'weekly_completion_percentage': 0.0,
                'weekly_completed': 0
            },
            'streak_metrics': {
                'current_streak': 0,
                'longest_streak': 0,
                'streak_status': 'broken'
            },
            'progress_metrics': {
                'weekly_completion': 0.0,
                'monthly_completion': 0.0,
                'weekly_goal': 85.0,
                'monthly_goal': 80.0
            },
            'today_habits': [],
            'weekly_progress': [],
            'risk_trends': {
                'trend': 'stable',
                'change_amount': 0.0,
                'data_points': []
            },
            'water_intake': {
                'target_glasses': 8,
                'completed_glasses': 0,
                'completion_percentage': 0.0,
                'streak_days': 0,
                'weekly_progress': []
            },
            'eye_exercises': {
                'target_sessions': 3,
                'completed_sessions': 0,
                'completion_percentage': 0.0,
                'streak_days': 0,
                'weekly_progress': []
            }
        }

# Create a global instance
dashboard_service = DashboardService()