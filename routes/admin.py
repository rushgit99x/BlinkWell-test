from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user, login_user, logout_user
from functools import wraps
import MySQLdb
from models.admin import (
    get_admin_user_by_username, get_admin_user_by_id, log_admin_activity,
    check_admin_permission, get_admin_users, create_admin_user,
    update_admin_user, delete_admin_user, update_admin_last_login,
    create_admin_tables
)
from config import Config
import json
from datetime import datetime, timedelta

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(f):
    """Decorator to check if user is admin"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if admin user is logged in via session
        print(f"🔒 Admin access check - Session data: {session}")
        print(f"   admin_user_id: {session.get('admin_user_id')}")
        print(f"   admin_role: {session.get('admin_role')}")
        
        if not session.get('admin_user_id') or not session.get('admin_role'):
            print("❌ Access denied - No admin session data")
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('admin.login'))
        
        print(f"✅ Admin access granted for user: {session.get('admin_username')}")
        return f(*args, **kwargs)
    return decorated_function

def permission_required(resource, action):
    """Decorator to check specific admin permissions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if admin user is logged in via session
            if not session.get('admin_user_id') or not session.get('admin_role'):
                flash('Access denied. Admin privileges required.', 'error')
                return redirect(url_for('admin.login'))
            
            admin_role = session.get('admin_role')
            if not check_admin_permission(admin_role, resource, action):
                flash(f'Access denied. You need {action} permission for {resource}.', 'error')
                return redirect(url_for('admin.dashboard'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_db_connection():
    """Get database connection"""
    return MySQLdb.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        passwd=Config.MYSQL_PASSWORD,
        db=Config.MYSQL_DB
    )

@admin_bp.route('/')
@admin_required
def index():
    """Admin dashboard redirect"""
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"🔐 Login attempt for username: {username}")
        
        admin_user = get_admin_user_by_username(username)
        
        if admin_user:
            print(f"✅ Admin user found: {admin_user.username}, role: {admin_user.role}")
            
            if admin_user.check_password(password):
                print(f"✅ Password correct for user: {username}")
                
                # Store admin user info in session for admin panel
                session['admin_user_id'] = admin_user.id
                session['admin_username'] = admin_user.username
                session['admin_role'] = admin_user.role
                
                print(f"📝 Session data set: {session}")
                
                login_user(admin_user)
                update_admin_last_login(admin_user.id)
                
                log_admin_activity(
                    admin_id=admin_user.id,
                    action='login',
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                
                flash('Login successful!', 'success')
                return redirect(url_for('admin.dashboard'))
            else:
                print(f"❌ Password incorrect for user: {username}")
                flash('Invalid username or password.', 'error')
        else:
            print(f"❌ Admin user not found: {username}")
            flash('Invalid username or password.', 'error')
    
    return render_template('admin/login.html')

@admin_bp.route('/logout')
def logout():
    """Admin logout"""
    admin_user_id = session.get('admin_user_id')
    if admin_user_id:
        log_admin_activity(
            admin_id=admin_user_id,
            action='logout',
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
    
    # Clear session
    session.pop('admin_user_id', None)
    session.pop('admin_username', None)
    session.pop('admin_role', None)
    
    # Also logout from Flask-Login if user was logged in
    if hasattr(current_user, 'id'):
        logout_user()
    
    flash('You have been logged out.', 'info')
    return redirect(url_for('admin.login'))

@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """Admin dashboard with statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # Get total eye habits
        cursor.execute("SELECT COUNT(*) FROM eye_habits")
        total_habits = cursor.fetchone()[0]
        
        # Get total habit tracking records
        cursor.execute("SELECT COUNT(*) FROM habit_tracking")
        total_tracking = cursor.fetchone()[0]
        
        # Get total eye health data records
        cursor.execute("SELECT COUNT(*) FROM user_eye_health_data")
        total_health_data = cursor.fetchone()[0]
        
        # Get recent user registrations (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM users 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        recent_users = cursor.fetchone()[0]
        
        # Get recent habit completions (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM habit_tracking 
            WHERE date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) AND is_completed = 1
        """)
        recent_completions = cursor.fetchone()[0]
        
        # Get top habits by completion rate
        cursor.execute("""
            SELECT h.name, 
                   COUNT(ht.id) as total_attempts,
                   SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed,
                   ROUND((SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) / COUNT(ht.id)) * 100, 2) as completion_rate
            FROM eye_habits h
            LEFT JOIN habit_tracking ht ON h.id = ht.habit_id
            GROUP BY h.id, h.name
            HAVING total_attempts > 0
            ORDER BY completion_rate DESC
            LIMIT 5
        """)
        top_habits = cursor.fetchall()
        
        # Get user activity by day (last 7 days)
        cursor.execute("""
            SELECT DATE(ht.date) as activity_date,
                   COUNT(DISTINCT ht.user_id) as active_users,
                   COUNT(*) as total_activities
            FROM habit_tracking ht
            WHERE ht.date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            GROUP BY DATE(ht.date)
            ORDER BY activity_date DESC
        """)
        daily_activity = cursor.fetchall()
        
        # Get admin activity logs (recent)
        cursor.execute("""
            SELECT al.action, al.table_name, al.created_at, au.username
            FROM admin_activity_logs al
            JOIN admin_users au ON al.admin_id = au.id
            ORDER BY al.created_at DESC
            LIMIT 10
        """)
        recent_admin_activity = cursor.fetchall()
        
        stats = {
            'total_users': total_users,
            'total_habits': total_habits,
            'total_tracking': total_tracking,
            'total_health_data': total_health_data,
            'recent_users': recent_users,
            'recent_completions': recent_completions,
            'top_habits': top_habits,
            'daily_activity': daily_activity,
            'recent_admin_activity': recent_admin_activity
        }
        
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        stats = {}
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/dashboard.html', stats=stats)

@admin_bp.route('/users')
@admin_required
@permission_required('users', 'read')
def users():
    """User management page"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get users with pagination
        page = request.args.get('page', 1, type=int)
        per_page = 20
        offset = (page - 1) * per_page
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # Get users
        cursor.execute("""
            SELECT u.*, 
                   COUNT(DISTINCT uh.id) as total_habits,
                   COUNT(DISTINCT ht.id) as total_tracking_records,
                   MAX(ht.date) as last_activity
            FROM users u
            LEFT JOIN user_habits uh ON u.id = uh.user_id
            LEFT JOIN habit_tracking ht ON u.id = ht.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        
        users = cursor.fetchall()
        
        total_pages = (total_users + per_page - 1) // per_page
        
    except Exception as e:
        print(f"Error fetching users: {e}")
        users = []
        total_pages = 0
        total_users = 0
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/users.html', 
                         users=users, 
                         page=page, 
                         total_pages=total_pages,
                         total_users=total_users)

@admin_bp.route('/users/<int:user_id>')
@admin_required
@permission_required('users', 'read')
def user_detail(user_id):
    """User detail page"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user info
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('admin.users'))
        
        # Get user habits
        cursor.execute("""
            SELECT uh.*, h.name, h.description, h.category
            FROM user_habits uh
            JOIN eye_habits h ON uh.habit_id = h.id
            WHERE uh.user_id = %s
        """, (user_id,))
        user_habits = cursor.fetchall()
        
        # Get habit tracking
        cursor.execute("""
            SELECT ht.*, h.name as habit_name
            FROM habit_tracking ht
            JOIN eye_habits h ON ht.habit_id = h.id
            WHERE ht.user_id = %s
            ORDER BY ht.date DESC
            LIMIT 50
        """, (user_id,))
        habit_tracking = cursor.fetchall()
        
        # Get eye health data
        cursor.execute("""
            SELECT * FROM user_eye_health_data 
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        eye_health_data = cursor.fetchall()
        
        # Get achievements
        cursor.execute("""
            SELECT ha.*, h.name as habit_name
            FROM habit_achievements ha
            LEFT JOIN eye_habits h ON ha.habit_id = h.id
            WHERE ha.user_id = %s
            ORDER BY ha.earned_date DESC
        """, (user_id,))
        achievements = cursor.fetchall()
        
    except Exception as e:
        print(f"Error fetching user detail: {e}")
        user = None
        user_habits = []
        habit_tracking = []
        eye_health_data = []
        achievements = []
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/user_detail.html',
                         user=user,
                         user_habits=user_habits,
                         habit_tracking=habit_tracking,
                         eye_health_data=eye_health_data,
                         achievements=achievements)

@admin_bp.route('/habits')
@admin_required
@permission_required('eye_habits', 'read')
def habits():
    """Eye habits management page"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get habits with usage statistics
        cursor.execute("""
            SELECT h.*, 
                   COUNT(DISTINCT uh.user_id) as total_users,
                   COUNT(ht.id) as total_tracking_records,
                   ROUND(AVG(CASE WHEN ht.is_completed = 1 THEN ht.completion_percentage ELSE 0 END), 2) as avg_completion_rate
            FROM eye_habits h
            LEFT JOIN user_habits uh ON h.id = uh.habit_id
            LEFT JOIN habit_tracking ht ON h.id = ht.habit_id
            GROUP BY h.id
            ORDER BY h.created_at DESC
        """)
        
        habits = cursor.fetchall()
        
    except Exception as e:
        print(f"Error fetching habits: {e}")
        habits = []
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/habits.html', habits=habits)

@admin_bp.route('/habits/create', methods=['GET', 'POST'])
@admin_required
@permission_required('eye_habits', 'write')
def create_habit():
    """Create new eye habit"""
    if request.method == 'POST':
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO eye_habits (
                    name, description, category, icon, target_frequency,
                    target_count, target_unit, instructions, benefits,
                    difficulty_level, estimated_time_minutes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                request.form.get('name'),
                request.form.get('description'),
                request.form.get('category'),
                request.form.get('icon'),
                request.form.get('target_frequency'),
                request.form.get('target_count'),
                request.form.get('target_unit'),
                request.form.get('instructions'),
                request.form.get('benefits'),
                request.form.get('difficulty_level'),
                request.form.get('estimated_time_minutes')
            ))
            
            conn.commit()
            
            log_admin_activity(
                admin_id=session.get('admin_user_id'),
                action='create_habit',
                table_name='eye_habits',
                record_id=cursor.lastrowid,
                details=request.form.to_dict(),
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            
            flash('Habit created successfully!', 'success')
            return redirect(url_for('admin.habits'))
            
        except Exception as e:
            print(f"Error creating habit: {e}")
            flash('Error creating habit.', 'error')
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    return render_template('admin/create_habit.html')

@admin_bp.route('/habits/<int:habit_id>/edit', methods=['GET', 'POST'])
@admin_required
@permission_required('eye_habits', 'write')
def edit_habit(habit_id):
    """Edit eye habit"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            cursor.execute("""
                UPDATE eye_habits SET
                    name = %s, description = %s, category = %s, icon = %s,
                    target_frequency = %s, target_count = %s, target_unit = %s,
                    instructions = %s, benefits = %s, difficulty_level = %s,
                    estimated_time_minutes = %s, is_active = %s
                WHERE id = %s
            """, (
                request.form.get('name'),
                request.form.get('description'),
                request.form.get('category'),
                request.form.get('icon'),
                request.form.get('target_frequency'),
                request.form.get('target_count'),
                request.form.get('target_unit'),
                request.form.get('instructions'),
                request.form.get('benefits'),
                request.form.get('difficulty_level'),
                request.form.get('estimated_time_minutes'),
                request.form.get('is_active', 0),
                habit_id
            ))
            
            conn.commit()
            
            log_admin_activity(
                admin_id=session.get('admin_user_id'),
                action='edit_habit',
                table_name='eye_habits',
                record_id=habit_id,
                details=request.form.to_dict(),
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            
            flash('Habit updated successfully!', 'success')
            return redirect(url_for('admin.habits'))
        
        # Get habit data
        cursor.execute("SELECT * FROM eye_habits WHERE id = %s", (habit_id,))
        habit = cursor.fetchone()
        
        if not habit:
            flash('Habit not found.', 'error')
            return redirect(url_for('admin.habits'))
        
    except Exception as e:
        print(f"Error editing habit: {e}")
        flash('Error editing habit.', 'error')
        return redirect(url_for('admin.habits'))
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/edit_habit.html', habit=habit)

@admin_bp.route('/habits/<int:habit_id>/delete', methods=['POST'])
@admin_required
@permission_required('eye_habits', 'delete')
def delete_habit(habit_id):
    """Delete eye habit"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if habit is being used
        cursor.execute("SELECT COUNT(*) FROM user_habits WHERE habit_id = %s", (habit_id,))
        usage_count = cursor.fetchone()[0]
        
        if usage_count > 0:
            flash(f'Cannot delete habit. It is being used by {usage_count} users.', 'error')
            return redirect(url_for('admin.habits'))
        
        # Soft delete by setting is_active to 0
        cursor.execute("UPDATE eye_habits SET is_active = 0 WHERE id = %s", (habit_id,))
        conn.commit()
        
        log_admin_activity(
            admin_id=session.get('admin_user_id'),
            action='delete_habit',
            table_name='eye_habits',
            record_id=habit_id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        flash('Habit deleted successfully!', 'success')
        
    except Exception as e:
        print(f"Error deleting habit: {e}")
        flash('Error deleting habit.', 'error')
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('admin.habits'))

@admin_bp.route('/analytics')
@admin_required
def analytics():
    """Analytics and reporting page"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get date range from request
        days = request.args.get('days', 30, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        # User registration trend
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as new_users
            FROM users
            WHERE created_at >= %s
            GROUP BY DATE(created_at)
            ORDER BY date
        """, (start_date,))
        user_trend = cursor.fetchall()
        
        # Habit completion trend
        cursor.execute("""
            SELECT DATE(date) as date, 
                   COUNT(*) as total_activities,
                   SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) as completed_activities
            FROM habit_tracking
            WHERE date >= %s
            GROUP BY DATE(date)
            ORDER BY date
        """, (start_date,))
        habit_trend = cursor.fetchall()
        
        # Category performance
        cursor.execute("""
            SELECT h.category,
                   COUNT(ht.id) as total_attempts,
                   SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed,
                   ROUND((SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) / COUNT(ht.id)) * 100, 2) as completion_rate
            FROM eye_habits h
            LEFT JOIN habit_tracking ht ON h.id = ht.habit_id
            WHERE ht.date >= %s
            GROUP BY h.category
            HAVING total_attempts > 0
            ORDER BY completion_rate DESC
        """, (start_date,))
        category_performance = cursor.fetchall()
        
        # User engagement metrics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT u.id) as total_users,
                COUNT(DISTINCT CASE WHEN ht.date >= %s THEN u.id END) as active_users,
                ROUND(COUNT(DISTINCT CASE WHEN ht.date >= %s THEN u.id END) / COUNT(DISTINCT u.id) * 100, 2) as engagement_rate
            FROM users u
            LEFT JOIN habit_tracking ht ON u.id = ht.user_id
        """, (start_date, start_date))
        engagement_metrics = cursor.fetchone()
        
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        user_trend = []
        habit_trend = []
        category_performance = []
        engagement_metrics = {}
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/analytics.html',
                         user_trend=user_trend,
                         habit_trend=habit_trend,
                         category_performance=category_performance,
                         engagement_metrics=engagement_metrics,
                         days=days)

@admin_bp.route('/admin-users')
@admin_required
@permission_required('admin_users', 'read')
def admin_users():
    """Admin users management page"""
    try:
        admins = get_admin_users()
    except Exception as e:
        print(f"Error fetching admin users: {e}")
        admins = []
    
    return render_template('admin/admin_users.html', admins=admins)

@admin_bp.route('/admin-users/create', methods=['GET', 'POST'])
@admin_required
@permission_required('admin_users', 'write')
def create_admin_user():
    """Create new admin user"""
    if request.method == 'POST':
        try:
            admin_id = create_admin_user(
                username=request.form.get('username'),
                email=request.form.get('email'),
                password=request.form.get('password'),
                role=request.form.get('role')
            )
            
            if admin_id:
                            log_admin_activity(
                admin_id=session.get('admin_user_id'),
                action='create_admin_user',
                table_name='admin_users',
                record_id=admin_id,
                details=request.form.to_dict(),
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
                
                flash('Admin user created successfully!', 'success')
                return redirect(url_for('admin.admin_users'))
            else:
                flash('Error creating admin user.', 'error')
        except Exception as e:
            print(f"Error creating admin user: {e}")
            flash('Error creating admin user.', 'error')
    
    return render_template('admin/create_admin_user.html')

@admin_bp.route('/admin-users/<int:admin_id>/edit', methods=['GET', 'POST'])
@admin_required
@permission_required('admin_users', 'write')
def edit_admin_user(admin_id):
    """Edit admin user"""
    if request.method == 'POST':
        try:
            success = update_admin_user(
                admin_id=admin_id,
                username=request.form.get('username'),
                email=request.form.get('email'),
                role=request.form.get('role'),
                is_active=request.form.get('is_active', 0)
            )
            
            if success:
                            log_admin_activity(
                admin_id=session.get('admin_user_id'),
                action='edit_admin_user',
                table_name='admin_users',
                record_id=admin_id,
                details=request.form.to_dict(),
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
                
                flash('Admin user updated successfully!', 'success')
                return redirect(url_for('admin.admin_users'))
            else:
                flash('Error updating admin user.', 'error')
        except Exception as e:
            print(f"Error updating admin user: {e}")
            flash('Error updating admin user.', 'error')
    
    admin_user = get_admin_user_by_id(admin_id)
    if not admin_user:
        flash('Admin user not found.', 'error')
        return redirect(url_for('admin.admin_users'))
    
    return render_template('admin/edit_admin_user.html', admin_user=admin_user)

@admin_bp.route('/admin-users/<int:admin_id>/delete', methods=['POST'])
@admin_required
@permission_required('admin_users', 'delete')
def delete_admin_user(admin_id):
    """Delete admin user"""
    try:
        if admin_id == session.get('admin_user_id'):
            flash('You cannot delete your own account.', 'error')
            return redirect(url_for('admin.admin_users'))
        
        success = delete_admin_user(admin_id)
        
        if success:
            log_admin_activity(
                admin_id=session.get('admin_user_id'),
                action='delete_admin_user',
                table_name='admin_users',
                record_id=admin_id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            
            flash('Admin user deleted successfully!', 'success')
        else:
            flash('Error deleting admin user.', 'error')
    except Exception as e:
        print(f"Error deleting admin user: {e}")
        flash('Error deleting admin user.', 'error')
    
    return redirect(url_for('admin.admin_users'))

@admin_bp.route('/activity-logs')
@admin_required
def activity_logs():
    """Admin activity logs page"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 50
        offset = (page - 1) * per_page
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM admin_activity_logs")
        total_logs = cursor.fetchone()[0]
        
        # Get activity logs
        cursor.execute("""
            SELECT al.*, au.username
            FROM admin_activity_logs al
            JOIN admin_users au ON al.admin_id = au.id
            ORDER BY al.created_at DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        
        logs = cursor.fetchall()
        total_pages = (total_logs + per_page - 1) // per_page
        
    except Exception as e:
        print(f"Error fetching activity logs: {e}")
        logs = []
        total_pages = 0
        total_logs = 0
    finally:
        cursor.close()
        conn.close()
    
    return render_template('admin/activity_logs.html',
                         logs=logs,
                         page=page,
                         total_pages=total_pages,
                         total_logs=total_logs)

@admin_bp.route('/api/stats')
@admin_required
def api_stats():
    """API endpoint for dashboard statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get real-time stats
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM habit_tracking WHERE date = CURDATE()")
        today_activities = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE created_at >= CURDATE()")
        today_users = cursor.fetchone()[0]
        
        stats = {
            'total_users': total_users,
            'today_activities': today_activities,
            'today_users': today_users,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error fetching API stats: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500
    finally:
        cursor.close()
        conn.close()

@admin_bp.route('/setup')
def setup():
    """Initial admin setup"""
    try:
        create_admin_tables()
        flash('Admin tables created successfully! Default admin user: admin/admin123', 'success')
    except Exception as e:
        print(f"Error setting up admin: {e}")
        flash('Error setting up admin tables.', 'error')
    
    return redirect(url_for('admin.login'))