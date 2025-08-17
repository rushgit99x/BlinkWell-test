from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, login_user, logout_user, current_user
from functools import wraps
from models.admin import (
    authenticate_admin, load_admin_user, create_admin_user, 
    get_all_admin_users, log_admin_action, get_system_stats, 
    get_recent_admin_logs
)
from models.user import get_all_users
import os
import psutil
import datetime

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not hasattr(current_user, 'is_admin'):
            flash('Please log in as an administrator.', 'error')
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

def super_admin_required(f):
    """Decorator to require super admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not hasattr(current_user, 'is_admin'):
            flash('Please log in as an administrator.', 'error')
            return redirect(url_for('admin.login'))
        if current_user.role != 'super_admin':
            flash('Super admin privileges required.', 'error')
            return redirect(url_for('admin.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin_user = authenticate_admin(username, password)
        if admin_user:
            login_user(admin_user)
            log_admin_action(admin_user.id, 'login', f'Admin {username} logged in', request.remote_addr)
            flash('Successfully logged in as administrator.', 'success')
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('admin/login_standalone.html')

@admin_bp.route('/logout')
@login_required
def logout():
    """Admin logout"""
    if hasattr(current_user, 'is_admin'):
        log_admin_action(current_user.id, 'logout', f'Admin {current_user.username} logged out', request.remote_addr)
    logout_user()
    flash('Successfully logged out.', 'success')
    return redirect(url_for('admin.login'))

@admin_bp.route('/')
@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """Admin dashboard"""
    try:
        # Get system statistics
        stats = get_system_stats()
        
        # Get recent admin logs
        recent_logs = get_recent_admin_logs(10)
        
        # System information
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'uptime': datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
        }
        
        # File system info
        uploads_size = 0
        models_size = 0
        if os.path.exists('uploads'):
            uploads_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk('uploads')
                              for filename in filenames)
        if os.path.exists('models'):
            models_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk('models')
                             for filename in filenames)
        
        system_info['uploads_size_mb'] = round(uploads_size / (1024 * 1024), 2)
        system_info['models_size_mb'] = round(models_size / (1024 * 1024), 2)
        
        log_admin_action(current_user.id, 'dashboard_access', 'Accessed admin dashboard', request.remote_addr)
        
        return render_template('admin/dashboard_standalone.html', 
                             stats=stats, 
                             recent_logs=recent_logs,
                             system_info=system_info)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('admin/dashboard_standalone.html', 
                             stats={}, 
                             recent_logs=[],
                             system_info={})

@admin_bp.route('/users')
@admin_required
def users():
    """User management page"""
    try:
        users = get_all_users()
        log_admin_action(current_user.id, 'users_access', 'Accessed user management', request.remote_addr)
        return render_template('admin/users_standalone.html', users=users)
    except Exception as e:
        flash(f'Error loading users: {str(e)}', 'error')
        return render_template('admin/users_standalone.html', users=[])

@admin_bp.route('/admins')
@admin_required
def admins():
    """Admin user management page"""
    try:
        admin_users = get_all_admin_users()
        log_admin_action(current_user.id, 'admins_access', 'Accessed admin management', request.remote_addr)
        return render_template('admin/admins.html', admin_users=admin_users)
    except Exception as e:
        flash(f'Error loading admin users: {str(e)}', 'error')
        return render_template('admin/admins.html', admin_users=[])

@admin_bp.route('/create-admin', methods=['GET', 'POST'])
@super_admin_required
def create_admin():
    """Create new admin user"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'admin')
        
        if create_admin_user(username, email, password, role):
            log_admin_action(current_user.id, 'admin_created', f'Created admin user: {username}', request.remote_addr)
            flash('Admin user created successfully.', 'success')
            return redirect(url_for('admin.admins'))
        else:
            flash('Failed to create admin user. Username or email may already exist.', 'error')
    
    return render_template('admin/create_admin.html')

@admin_bp.route('/logs')
@admin_required
def logs():
    """Admin action logs"""
    try:
        page = request.args.get('page', 1, type=int)
        limit = 50
        offset = (page - 1) * limit
        
        # Get logs with pagination
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM admin_logs")
        total_logs = cursor.fetchone()[0]
        
        # Get paginated logs
        cursor.execute("""
            SELECT al.*, au.username 
            FROM admin_logs al 
            JOIN admin_users au ON al.admin_id = au.id 
            ORDER BY al.timestamp DESC 
            LIMIT %s OFFSET %s
        """, (limit, offset))
        logs = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        total_pages = (total_logs + limit - 1) // limit
        
        log_admin_action(current_user.id, 'logs_access', 'Accessed admin logs', request.remote_addr)
        
        return render_template('admin/logs.html', 
                             logs=logs, 
                             page=page, 
                             total_pages=total_pages,
                             total_logs=total_logs)
    except Exception as e:
        flash(f'Error loading logs: {str(e)}', 'error')
        return render_template('admin/logs.html', logs=[], page=1, total_pages=1, total_logs=0)

@admin_bp.route('/system')
@admin_required
def system():
    """System monitoring page"""
    try:
        # System metrics
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory(),
            'disk': psutil.disk_usage('/'),
            'network': psutil.net_io_counters(),
            'boot_time': datetime.datetime.fromtimestamp(psutil.boot_time()),
            'uptime': datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
        }
        
        # Process information
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        processes = processes[:20]  # Top 20 processes
        
        log_admin_action(current_user.id, 'system_access', 'Accessed system monitoring', request.remote_addr)
        
        return render_template('admin/system.html', 
                             system_info=system_info,
                             processes=processes)
    except Exception as e:
        flash(f'Error loading system information: {str(e)}', 'error')
        return render_template('admin/system.html', system_info={}, processes=[])

@admin_bp.route('/api/stats')
@admin_required
def api_stats():
    """API endpoint for real-time statistics"""
    try:
        stats = get_system_stats()
        stats['timestamp'] = datetime.datetime.now().isoformat()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/system')
@admin_required
def api_system():
    """API endpoint for real-time system information"""
    try:
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.datetime.now().isoformat()
        }
        return jsonify(system_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500