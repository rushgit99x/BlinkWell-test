#!/usr/bin/env python3
"""
BlinkWell Admin Panel Setup Script

This script helps you set up the admin panel for the first time.
Run this script after setting up your database and environment variables.
"""

import os
import sys
import MySQLdb
from werkzeug.security import generate_password_hash

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def create_admin_tables():
    """Create admin-related tables"""
    try:
        # Connect to database
        conn = MySQLdb.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            passwd=Config.MYSQL_PASSWORD,
            db=Config.MYSQL_DB
        )
        cursor = conn.cursor()
        
        print("✅ Connected to database successfully")
        
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
        print("✅ Admin tables created successfully")
        
        return conn, cursor
        
    except Exception as e:
        print(f"❌ Error creating admin tables: {e}")
        return None, None

def create_default_admin_user(cursor, conn):
    """Create default admin user"""
    try:
        # Check if admin user already exists
        cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
        if cursor.fetchone()[0] > 0:
            print("ℹ️  Admin user already exists")
            return True
        
        # Create default admin user
        admin_password = generate_password_hash('admin123')
        cursor.execute("""
            INSERT INTO admin_users (username, email, password_hash, role) 
            VALUES ('admin', 'admin@blinkwell.com', %s, 'super_admin')
        """, (admin_password,))
        
        conn.commit()
        print("✅ Default admin user created successfully")
        print("   Username: admin")
        print("   Password: admin123")
        print("   Role: super_admin")
        print("   ⚠️  IMPORTANT: Change this password after first login!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False

def create_default_permissions(cursor, conn):
    """Create default permissions"""
    try:
        # Define permissions for each role
        permissions = [
            # Super Admin - Full access
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
            
            # Admin - Most access
            ('admin', 'users', 'read'),
            ('admin', 'eye_habits', 'read'),
            ('admin', 'eye_habits', 'write'),
            ('admin', 'habit_tracking', 'read'),
            ('admin', 'user_eye_health_data', 'read'),
            
            # Moderator - Read-only access
            ('moderator', 'users', 'read'),
            ('moderator', 'eye_habits', 'read'),
            ('moderator', 'habit_tracking', 'read'),
        ]
        
        # Insert permissions
        for role, resource, action in permissions:
            cursor.execute("""
                INSERT IGNORE INTO admin_permissions (role, resource, action) 
                VALUES (%s, %s, %s)
            """, (role, resource, action))
        
        conn.commit()
        print("✅ Default permissions created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating permissions: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 BlinkWell Admin Panel Setup")
    print("=" * 40)
    
    # Check if we can import config
    try:
        print(f"📁 Database: {Config.MYSQL_DB}")
        print(f"📁 Host: {Config.MYSQL_HOST}")
        print(f"📁 User: {Config.MYSQL_USER}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("   Make sure config.py exists and has correct database settings")
        return False
    
    # Create admin tables
    conn, cursor = create_admin_tables()
    if not conn:
        return False
    
    try:
        # Create default admin user
        if not create_default_admin_user(cursor, conn):
            return False
        
        # Create default permissions
        if not create_default_permissions(cursor, conn):
            return False
        
        print("\n🎉 Admin panel setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Start your Flask application")
        print("   2. Navigate to /admin in your browser")
        print("   3. Login with admin/admin123")
        print("   4. Change the default password immediately")
        print("   5. Create additional admin users as needed")
        
        return True
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)