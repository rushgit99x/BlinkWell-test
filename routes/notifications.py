from flask import Blueprint, request, jsonify, current_app, render_template
from flask_login import login_required, current_user
from services.email_service import email_service
from services.notification_scheduler import notification_scheduler
import logging
import MySQLdb # Added missing import for MySQLdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

notifications_bp = Blueprint('notifications', __name__)

@notifications_bp.route('/email-test')
@login_required
def email_test_page():
    """Serve the email test page"""
    return render_template('email_test.html')

@notifications_bp.route('/api/notifications/test-email', methods=['POST'])
@login_required
def test_email():
    """Test endpoint to send a test email to the current user"""
    try:
        data = request.get_json()
        email_type = data.get('email_type', 'welcome')
        
        if email_type == 'welcome':
            success = email_service.send_welcome_email(
                user_email=current_user.email,
                username=current_user.username
            )
        elif email_type == 'habit_reminder':
            success = email_service.send_habit_reminder_email(
                user_email=current_user.email,
                username=current_user.username,
                habit_name="20-20-20 Rule",
                reminder_time="9:00 AM",
                streak_days=5
            )
        elif email_type == 'recommendations':
            recommendations = [
                {
                    'title': 'Take Regular Screen Breaks',
                    'description': 'Remember to follow the 20-20-20 rule every 20 minutes.',
                    'based_on': 'screen time patterns'
                },
                {
                    'title': 'Maintain Proper Lighting',
                    'description': 'Ensure your workspace has adequate lighting to reduce eye strain.',
                    'based_on': 'work environment analysis'
                }
            ]
            success = email_service.send_recommendation_email(
                user_email=current_user.email,
                username=current_user.username,
                recommendations=recommendations
            )
        elif email_type == 'weekly_progress':
            weekly_stats = {
                'habits_completed': 15,
                'completion_rate': 85.7,
                'best_day': 'Wednesday',
                'streak_info': {
                    'longest_streak': 7,
                    'total_streak_days': 25
                }
            }
            success = email_service.send_weekly_progress_email(
                user_email=current_user.email,
                username=current_user.username,
                weekly_stats=weekly_stats
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid email type. Supported types: welcome, habit_reminder, recommendations, weekly_progress'
            }), 400
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Test {email_type} email sent successfully to {current_user.email}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to send test email. Check email service configuration.'
            }), 500
            
    except Exception as e:
        logger.error(f"Error sending test email: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/send-reminder', methods=['POST'])
@login_required
def send_reminder():
    """Send an immediate reminder for a specific habit"""
    try:
        data = request.get_json()
        habit_name = data.get('habit_name')
        reminder_time = data.get('reminder_time', 'Now')
        
        if not habit_name:
            return jsonify({
                'success': False,
                'error': 'habit_name is required'
            }), 400
        
        success = notification_scheduler.send_immediate_reminder(
            user_id=current_user.id,
            habit_name=habit_name,
            reminder_time=reminder_time
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Reminder sent successfully for {habit_name}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to send reminder'
            }), 500
            
    except Exception as e:
        logger.error(f"Error sending reminder: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/send-recommendations', methods=['POST'])
@login_required
def send_recommendations():
    """Send personalized recommendations to the current user"""
    try:
        # Get user's recommendations
        recommendations = notification_scheduler._get_user_recommendations(current_user.id)
        
        if not recommendations:
            return jsonify({
                'success': False,
                'error': 'No recommendations available for this user'
            }), 400
        
        # Get user's current habits for context
        user_habits = notification_scheduler._get_user_active_habits(current_user.id)
        
        success = email_service.send_recommendation_email(
            user_email=current_user.email,
            username=current_user.username,
            recommendations=recommendations,
            user_habits=user_habits
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Recommendations sent successfully to {current_user.email}',
                'recommendations_count': len(recommendations)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to send recommendations'
            }), 500
            
    except Exception as e:
        logger.error(f"Error sending recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/send-weekly-report', methods=['POST'])
@login_required
def send_weekly_report():
    """Send weekly progress report to the current user"""
    try:
        # Get user's weekly stats
        weekly_stats = notification_scheduler._get_user_weekly_stats(current_user.id)
        
        if not weekly_stats or weekly_stats.get('habits_completed', 0) == 0:
            return jsonify({
                'success': False,
                'error': 'No habit data available for weekly report'
            }), 400
        
        success = email_service.send_weekly_progress_email(
            user_email=current_user.email,
            username=current_user.username,
            weekly_stats=weekly_stats
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Weekly report sent successfully to {current_user.email}',
                'stats': weekly_stats
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to send weekly report'
            }), 500
            
    except Exception as e:
        logger.error(f"Error sending weekly report: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/status', methods=['GET'])
@login_required
def get_notification_status():
    """Get the status of the notification scheduler"""
    try:
        status = {
            'scheduler_running': notification_scheduler.running,
            'email_service_configured': bool(
                current_app.config.get('SMTP_USERNAME') and 
                current_app.config.get('SMTP_PASSWORD') and 
                current_app.config.get('SENDER_EMAIL')
            ),
            'scheduled_notifications': [
                'Daily habit reminders: 9:00 AM, 2:00 PM, 6:00 PM',
                'Weekly progress reports: Sundays at 10:00 AM',
                'Monthly recommendations: First day of month at 9:00 AM'
            ]
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting notification status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/start-scheduler', methods=['POST'])
@login_required
def start_scheduler():
    """Start the notification scheduler (admin only)"""
    try:
        # Check if user is admin (you can implement your own admin check)
        if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
            return jsonify({
                'success': False,
                'error': 'Admin privileges required'
            }), 403
        
        if notification_scheduler.running:
            return jsonify({
                'success': False,
                'error': 'Scheduler is already running'
            }), 400
        
        notification_scheduler.start()
        
        return jsonify({
            'success': True,
            'message': 'Notification scheduler started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/stop-scheduler', methods=['POST'])
@login_required
def stop_scheduler():
    """Stop the notification scheduler (admin only)"""
    try:
        # Check if user is admin
        if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
            return jsonify({
                'success': False,
                'error': 'Admin privileges required'
            }), 403
        
        if not notification_scheduler.running:
            return jsonify({
                'success': False,
                'error': 'Scheduler is not running'
            }), 400
        
        notification_scheduler.stop()
        
        return jsonify({
            'success': True,
            'message': 'Notification scheduler stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping scheduler: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@notifications_bp.route('/api/notifications/preferences', methods=['GET', 'PUT'])
@login_required
def manage_notification_preferences():
    """Get or update user's notification preferences"""
    try:
        if request.method == 'GET':
            # Get user's current notification preferences
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            
            cursor.execute("""
                SELECT 
                    uh.habit_id,
                    h.name as habit_name,
                    uh.reminder_enabled,
                    uh.reminder_time
                FROM user_habits uh
                JOIN eye_habits h ON uh.habit_id = h.id
                WHERE uh.user_id = %s AND uh.is_active = 1
            """, (current_user.id,))
            
            preferences = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'preferences': preferences
            })
            
        elif request.method == 'PUT':
            # Update user's notification preferences
            data = request.get_json()
            habit_id = data.get('habit_id')
            reminder_enabled = data.get('reminder_enabled')
            reminder_time = data.get('reminder_time')
            
            if not habit_id:
                return jsonify({
                    'success': False,
                    'error': 'habit_id is required'
                }), 400
            
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_habits 
                SET reminder_enabled = %s, reminder_time = %s
                WHERE user_id = %s AND habit_id = %s
            """, (reminder_enabled, reminder_time, current_user.id, habit_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': 'Notification preferences updated successfully'
            })
            
    except Exception as e:
        logger.error(f"Error managing notification preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500