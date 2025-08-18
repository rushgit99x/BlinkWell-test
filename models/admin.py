import MySQLdb
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

class AdminUser:
    def __init__(self, id, username, email, role, is_active, created_at):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.is_active = is_active
        self.created_at = created_at
        self.is_authenticated = True
        self.is_anonymous = False

    def get_id(self):
        return str(self.id)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

def get_admin_db_connection():
    """Get database connection for admin operations"""
    import config
    return MySQLdb.connect(
        host=config.Config.MYSQL_HOST,
        user=config.Config.MYSQL_USER,
        passwd=config.Config.MYSQL_PASSWORD,
        db=config.Config.MYSQL_DB
    )

def create_admin_tables():
    """Create admin-related tables if they don't exist"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create admin_users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id INT NOT NULL AUTO_INCREMENT,
                username VARCHAR(80) NOT NULL UNIQUE,
                email VARCHAR(120) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('super_admin', 'admin', 'moderator') NOT NULL DEFAULT 'admin',
                is_active TINYINT(1) DEFAULT 1,
                last_login TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (id)
            ) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """)
        
        # Create admin_activity_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_activity_logs (
                id INT NOT NULL AUTO_INCREMENT,
                admin_id INT NOT NULL,
                action VARCHAR(100) NOT NULL,
                table_name VARCHAR(50),
                record_id INT,
                details JSON,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                KEY admin_id (admin_id),
                KEY action (action),
                KEY created_at (created_at)
            ) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """)
        
        # Create admin_permissions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_permissions (
                id INT NOT NULL AUTO_INCREMENT,
                role VARCHAR(50) NOT NULL,
                resource VARCHAR(100) NOT NULL,
                action VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                UNIQUE KEY role_resource_action (role, resource, action)
            ) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """)
        
        conn.commit()
        
        # Insert default admin user if not exists
        cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_password = generate_password_hash('admin123')
            cursor.execute("""
                INSERT INTO admin_users (username, email, password_hash, role) 
                VALUES ('admin', 'admin@blinkwell.com', %s, 'super_admin')
            """, (admin_password,))
            
            # Insert default permissions
            permissions = [
                ('super_admin', 'users', 'read'),
                ('super_admin', 'users', 'write'),
                ('super_admin', 'users', 'delete'),
                ('super_admin', 'eye_habits', 'read'),
                ('super_admin', 'eye_habits', 'write'),
                ('super_admin', 'eye_habits', 'delete'),
                ('super_admin', 'habit_tracking', 'read'),
                ('super_admin', 'habit_tracking', 'write'),
                ('super_admin', 'habit_tracking', 'delete'),
                ('super_admin', 'user_eye_health_data', 'read'),
                ('super_admin', 'user_eye_health_data', 'write'),
                ('super_admin', 'user_eye_health_data', 'delete'),
                ('super_admin', 'admin_users', 'read'),
                ('super_admin', 'admin_users', 'write'),
                ('super_admin', 'admin_users', 'delete'),
                ('admin', 'users', 'read'),
                ('admin', 'eye_habits', 'read'),
                ('admin', 'eye_habits', 'write'),
                ('admin', 'habit_tracking', 'read'),
                ('admin', 'user_eye_health_data', 'read'),
                ('moderator', 'users', 'read'),
                ('moderator', 'eye_habits', 'read'),
                ('moderator', 'habit_tracking', 'read')
            ]
            
            for role, resource, action in permissions:
                cursor.execute("""
                    INSERT IGNORE INTO admin_permissions (role, resource, action) 
                    VALUES (%s, %s, %s)
                """, (role, resource, action))
            
            conn.commit()
            
    except Exception as e:
        print(f"Error creating admin tables: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_admin_user_by_username(username):
    """Get admin user by username"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, username, email, role, is_active, created_at, password_hash
            FROM admin_users WHERE username = %s AND is_active = 1
        """, (username,))
        
        user_data = cursor.fetchone()
        if user_data:
            user = AdminUser(*user_data[:-1])  # Exclude password_hash from constructor
            user.password_hash = user_data[-1]  # Set password_hash separately
            return user
        return None
    finally:
        cursor.close()
        conn.close()

def get_admin_user_by_id(admin_id):
    """Get admin user by ID"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, username, email, role, is_active, created_at
            FROM admin_users WHERE id = %s AND is_active = 1
        """, (admin_id,))
        
        user_data = cursor.fetchone()
        if user_data:
            return AdminUser(*user_data)
        return None
    finally:
        cursor.close()
        conn.close()

def log_admin_activity(admin_id, action, table_name=None, record_id=None, details=None, ip_address=None, user_agent=None):
    """Log admin activity"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO admin_activity_logs 
            (admin_id, action, table_name, record_id, details, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (admin_id, action, table_name, record_id, 
               json.dumps(details) if details else None, ip_address, user_agent))
        conn.commit()
    except Exception as e:
        print(f"Error logging admin activity: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def check_admin_permission(role, resource, action):
    """Check if admin role has permission for resource and action"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM admin_permissions 
            WHERE role = %s AND resource = %s AND action = %s
        """, (role, resource, action))
        
        return cursor.fetchone()[0] > 0
    finally:
        cursor.close()
        conn.close()

def get_admin_users():
    """Get all admin users"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, username, email, role, is_active, created_at, last_login
            FROM admin_users ORDER BY created_at DESC
        """)
        
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def create_admin_user(username, email, password, role='admin'):
    """Create a new admin user"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO admin_users (username, email, password_hash, role)
            VALUES (%s, %s, %s, %s)
        """, (username, email, password_hash, role))
        
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        print(f"Error creating admin user: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

def update_admin_user(admin_id, username=None, email=None, role=None, is_active=None):
    """Update admin user"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        updates = []
        values = []
        
        if username is not None:
            updates.append("username = %s")
            values.append(username)
        if email is not None:
            updates.append("email = %s")
            values.append(email)
        if role is not None:
            updates.append("role = %s")
            values.append(role)
        if is_active is not None:
            updates.append("is_active = %s")
            values.append(is_active)
        
        if updates:
            values.append(admin_id)
            query = f"UPDATE admin_users SET {', '.join(updates)} WHERE id = %s"
            cursor.execute(query, values)
            conn.commit()
            return True
        return False
    except Exception as e:
        print(f"Error updating admin user: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def delete_admin_user(admin_id):
    """Delete admin user (soft delete by setting is_active to 0)"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("UPDATE admin_users SET is_active = 0 WHERE id = %s", (admin_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting admin user: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def update_admin_last_login(admin_id):
    """Update admin user's last login timestamp"""
    conn = get_admin_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("UPDATE admin_users SET last_login = NOW() WHERE id = %s", (admin_id,))
        conn.commit()
    except Exception as e:
        print(f"Error updating last login: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()