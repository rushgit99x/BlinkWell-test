from flask import current_app
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import MySQLdb
from datetime import datetime

class AdminUser(UserMixin):
    def __init__(self, id, username, email, role, created_at, last_login):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.created_at = created_at
        self.last_login = last_login
        self.is_admin = True

def init_admin_db(app):
    """Initialize admin tables if they don't exist"""
    conn = app.config['get_db_connection']()
    cursor = conn.cursor()
    
    # Create admin_users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            email VARCHAR(100) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('super_admin', 'admin', 'moderator') DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)
    
    # Create admin_logs table for audit trail
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            admin_id INT,
            action VARCHAR(255) NOT NULL,
            details TEXT,
            ip_address VARCHAR(45),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES admin_users(id)
        )
    """)
    
    # Create system_metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            metric_name VARCHAR(100) NOT NULL,
            metric_value TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

def create_admin_user(username, email, password, role='admin'):
    """Create a new admin user"""
    password_hash = generate_password_hash(password)
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO admin_users (username, email, password_hash, role) 
            VALUES (%s, %s, %s, %s)
        """, (username, email, password_hash, role))
        conn.commit()
        return True
    except MySQLdb.IntegrityError:
        return False
    finally:
        cursor.close()
        conn.close()

def authenticate_admin(username, password):
    """Authenticate admin user"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, email, password_hash, role, created_at, last_login 
        FROM admin_users 
        WHERE username = %s AND is_active = TRUE
    """, (username,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user_data and check_password_hash(user_data[3], password):
        # Update last login
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        cursor.execute("UPDATE admin_users SET last_login = NOW() WHERE id = %s", (user_data[0],))
        conn.commit()
        cursor.close()
        conn.close()
        
        return AdminUser(
            user_data[0], user_data[1], user_data[2], 
            user_data[4], user_data[5], user_data[6]
        )
    return None

def load_admin_user(user_id):
    """Load admin user by ID"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, email, password_hash, role, created_at, last_login 
        FROM admin_users 
        WHERE id = %s AND is_active = TRUE
    """, (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user_data:
        return AdminUser(
            user_data[0], user_data[1], user_data[2], 
            user_data[4], user_data[5], user_data[6]
        )
    return None

def get_all_admin_users():
    """Get all admin users"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, email, role, created_at, last_login, is_active 
        FROM admin_users 
        ORDER BY created_at DESC
    """)
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return users

def log_admin_action(admin_id, action, details=None, ip_address=None):
    """Log admin actions for audit trail"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO admin_logs (admin_id, action, details, ip_address) 
            VALUES (%s, %s, %s, %s)
        """, (admin_id, action, details, ip_address))
        conn.commit()
    except Exception as e:
        print(f"Error logging admin action: {e}")
    finally:
        cursor.close()
        conn.close()

def get_system_stats():
    """Get system statistics"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    
    stats = {}
    
    # Total users
    cursor.execute("SELECT COUNT(*) FROM users")
    stats['total_users'] = cursor.fetchone()[0]
    
    # Total admin users
    cursor.execute("SELECT COUNT(*) FROM admin_users WHERE is_active = TRUE")
    stats['total_admins'] = cursor.fetchone()[0]
    
    # Recent registrations (last 7 days)
    cursor.execute("SELECT COUNT(*) FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)")
    stats['recent_users'] = cursor.fetchone()[0]
    
    # Admin actions in last 24 hours
    cursor.execute("SELECT COUNT(*) FROM admin_logs WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)")
    stats['recent_admin_actions'] = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    return stats

def get_recent_admin_logs(limit=50):
    """Get recent admin action logs"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT al.*, au.username 
        FROM admin_logs al 
        JOIN admin_users au ON al.admin_id = au.id 
        ORDER BY al.timestamp DESC 
        LIMIT %s
    """, (limit,))
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return logs