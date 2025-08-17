from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from functools import wraps
import MySQLdb
from datetime import datetime, timedelta
import json
import os

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(f):
    """Decorator to check if user is admin"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
            flash('Admin privileges required', 'error')
            return redirect(url_for('main.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard with system overview"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get user statistics
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE DATE(created_at) = CURDATE()")
        new_users_today = cursor.fetchone()[0]
        
        # Get system statistics (you can expand this based on your actual database schema)
        cursor.execute("SELECT COUNT(*) FROM users WHERE last_login >= %s", 
                      ((datetime.now() - timedelta(days=7)),))
        active_users_week = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        stats = {
            'total_users': total_users,
            'new_users_today': new_users_today,
            'active_users_week': active_users_week,
        }
        
        return render_template('admin/dashboard.html', stats=stats)
        
    except Exception as e:
        flash(f'Error loading admin dashboard: {str(e)}', 'error')
        return render_template('admin/dashboard.html', stats={})

@admin_bp.route('/users')
@login_required
@admin_required
def manage_users():
    """User management page"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get all users with pagination
        page = request.args.get('page', 1, type=int)
        per_page = 20
        offset = (page - 1) * per_page
        
        cursor.execute("SELECT id, username, email, created_at, last_login, is_active FROM users LIMIT %s OFFSET %s", 
                      (per_page, offset))
        users = cursor.fetchall()
        
        # Get total count for pagination
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        total_pages = (total_users + per_page - 1) // per_page
        
        return render_template('admin/users.html', 
                             users=users, 
                             page=page, 
                             total_pages=total_pages,
                             total_users=total_users)
                             
    except Exception as e:
        flash(f'Error loading users: {str(e)}', 'error')
        return render_template('admin/users.html', users=[], page=1, total_pages=1, total_users=0)

@admin_bp.route('/users/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user details"""
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            is_active = request.form.get('is_active') == 'on'
            is_admin = request.form.get('is_admin') == 'on'
            
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE users 
                SET username = %s, email = %s, is_active = %s, is_admin = %s 
                WHERE id = %s
            """, (username, email, is_active, is_admin, user_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            flash('User updated successfully', 'success')
            return redirect(url_for('admin.manage_users'))
            
        except Exception as e:
            flash(f'Error updating user: {str(e)}', 'error')
    
    # GET request - show edit form
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, email, is_active, is_admin, created_at FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not user:
            flash('User not found', 'error')
            return redirect(url_for('admin.manage_users'))
        
        return render_template('admin/edit_user.html', user=user)
        
    except Exception as e:
        flash(f'Error loading user: {str(e)}', 'error')
        return redirect(url_for('admin.manage_users'))

@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user"""
    if user_id == current_user.id:
        flash('Cannot delete your own account', 'error')
        return redirect(url_for('admin.manage_users'))
    
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        flash('User deleted successfully', 'success')
        
    except Exception as e:
        flash(f'Error deleting user: {str(e)}', 'error')
    
    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/system')
@login_required
@admin_required
def system_status():
    """System status and health monitoring"""
    try:
        # Get database connection status
        db_status = "Connected"
        try:
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
        except:
            db_status = "Disconnected"
        
        # Get system information
        import psutil
        
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'db_status': db_status,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'flask_version': '2.0.1',  # You can get this dynamically
        }
        
        return render_template('admin/system.html', system_info=system_info)
        
    except Exception as e:
        flash(f'Error loading system status: {str(e)}', 'error')
        return render_template('admin/system.html', system_info={})

@admin_bp.route('/logs')
@login_required
@admin_required
def view_logs():
    """View system logs"""
    try:
        # This is a simple log viewer - you might want to implement a proper logging system
        log_file = 'app.log'
        logs = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = f.readlines()[-100:]  # Last 100 lines
        
        return render_template('admin/logs.html', logs=logs)
        
    except Exception as e:
        flash(f'Error loading logs: {str(e)}', 'error')
        return render_template('admin/logs.html', logs=[])

@admin_bp.route('/analytics')
@login_required
@admin_required
def analytics():
    """Analytics and reporting"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get user growth over time
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM users 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at) 
            ORDER BY date
        """)
        user_growth = cursor.fetchall()
        
        # Get user activity
        cursor.execute("""
            SELECT DATE(last_login) as date, COUNT(*) as count 
            FROM users 
            WHERE last_login >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY DATE(last_login) 
            ORDER BY date
        """)
        user_activity = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return render_template('admin/analytics.html', 
                             user_growth=user_growth,
                             user_activity=user_activity)
                             
    except Exception as e:
        flash(f'Error loading analytics: {str(e)}', 'error')
        return render_template('admin/analytics.html', user_growth=[], user_activity=[])

@admin_bp.route('/api/stats')
@login_required
@admin_required
def api_stats():
    """API endpoint for admin statistics"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get real-time statistics
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE DATE(created_at) = CURDATE()")
        new_users_today = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE last_login >= %s", 
                      ((datetime.now() - timedelta(days=1)),))
        active_users_today = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'total_users': total_users,
            'new_users_today': new_users_today,
            'active_users_today': active_users_today,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500