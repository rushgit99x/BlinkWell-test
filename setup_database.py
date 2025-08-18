#!/usr/bin/env python3
"""
BlinkWell Database Setup Script

This script executes the complete database setup including:
- Creating all tables (original + admin panel)
- Setting up default data
- Creating indexes, views, procedures, and triggers
- Verifying the setup

Usage: python setup_database.py
"""

import os
import sys
import MySQLdb
import time
from datetime import datetime

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🚀 BlinkWell Database Setup")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_step(step, description):
    """Print a setup step"""
    print(f"🔧 Step {step}: {description}")
    print("-" * 40)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def get_database_config():
    """Get database configuration from config.py"""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from config import Config
        
        return {
            'host': Config.MYSQL_HOST,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD,
            'database': Config.MYSQL_DB
        }
    except ImportError as e:
        print_error(f"Could not import config: {e}")
        print_info("Please ensure config.py exists with correct database settings")
        return None
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return None

def create_database_connection(config, create_db=False):
    """Create database connection"""
    try:
        if create_db:
            # Connect without specifying database to create it
            conn = MySQLdb.connect(
                host=config['host'],
                user=config['user'],
                passwd=config['password']
            )
        else:
            # Connect to specific database
            conn = MySQLdb.connect(
                host=config['host'],
                user=config['user'],
                passwd=config['password'],
                db=config['database']
            )
        
        return conn
    except MySQLdb.Error as e:
        print_error(f"Database connection failed: {e}")
        return None

def create_database(config):
    """Create the database if it doesn't exist"""
    try:
        conn = create_database_connection(config, create_db=True)
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{config['database']}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
        print_success(f"Database '{config['database']}' created/verified successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except MySQLdb.Error as e:
        print_error(f"Error creating database: {e}")
        return False

def execute_sql_file(conn, sql_file):
    """Execute SQL file"""
    try:
        if not os.path.exists(sql_file):
            print_error(f"SQL file not found: {sql_file}")
            return False
        
        cursor = conn.cursor()
        
        # Read SQL file
        with open(sql_file, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        
        # Split SQL statements (basic splitting by semicolon)
        statements = []
        current_statement = ""
        delimiter = ";"
        
        for line in sql_content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('--') or line.startswith('/*') or not line:
                continue
            
            # Check for delimiter changes
            if line.upper().startswith('DELIMITER'):
                delimiter = line.split()[1]
                continue
            
            current_statement += line + "\n"
            
            if line.endswith(delimiter):
                # Remove delimiter and trim
                statement = current_statement[:-len(delimiter)].strip()
                if statement:
                    statements.append(statement)
                current_statement = ""
        
        # Execute each statement
        total_statements = len(statements)
        executed_statements = 0
        
        print_info(f"Executing {total_statements} SQL statements...")
        
        for i, statement in enumerate(statements, 1):
            try:
                if statement.strip():
                    cursor.execute(statement)
                    executed_statements += 1
                    
                    # Show progress every 10 statements
                    if i % 10 == 0 or i == total_statements:
                        print(f"   Progress: {i}/{total_statements} statements executed")
                        
            except MySQLdb.Error as e:
                print_error(f"Error executing statement {i}: {e}")
                print(f"   Statement: {statement[:100]}...")
                return False
        
        conn.commit()
        print_success(f"All {executed_statements} SQL statements executed successfully")
        return True
        
    except Exception as e:
        print_error(f"Error executing SQL file: {e}")
        return False

def verify_setup(conn):
    """Verify the database setup"""
    try:
        cursor = conn.cursor()
        
        print_info("Verifying database setup...")
        
        # Check tables
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        expected_tables = [
            'admin_users', 'admin_activity_logs', 'admin_permissions',
            'users', 'eye_habits', 'user_habits', 'habit_tracking',
            'user_eye_health_data', 'habit_achievements', 'habit_summaries',
            'user_notification_preferences', 'user_privacy_settings', 'user_recommendations'
        ]
        
        missing_tables = [table for table in expected_tables if table not in tables]
        
        if missing_tables:
            print_error(f"Missing tables: {', '.join(missing_tables)}")
            return False
        
        print_success(f"All {len(tables)} tables created successfully")
        
        # Check admin user
        cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            print_error("Default admin user not found")
            return False
        
        print_success("Default admin user created successfully")
        
        # Check permissions
        cursor.execute("SELECT COUNT(*) FROM admin_permissions")
        permission_count = cursor.fetchone()[0]
        
        if permission_count == 0:
            print_error("No permissions found")
            return False
        
        print_success(f"{permission_count} permissions created successfully")
        
        # Check sample habits
        cursor.execute("SELECT COUNT(*) FROM eye_habits")
        habit_count = cursor.fetchone()[0]
        
        if habit_count == 0:
            print_error("No sample habits found")
            return False
        
        print_success(f"{habit_count} sample habits created successfully")
        
        return True
        
    except MySQLdb.Error as e:
        print_error(f"Error verifying setup: {e}")
        return False

def show_setup_summary(conn):
    """Show setup summary"""
    try:
        cursor = conn.cursor()
        
        print("\n" + "=" * 60)
        print("📊 Database Setup Summary")
        print("=" * 60)
        
        # Table count
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s", (config['database'],))
        table_count = cursor.fetchone()[0]
        print(f"📋 Total Tables: {table_count}")
        
        # Admin users
        cursor.execute("SELECT COUNT(*) FROM admin_users")
        admin_count = cursor.fetchone()[0]
        print(f"👨‍💼 Admin Users: {admin_count}")
        
        # Regular users
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"👥 Regular Users: {user_count}")
        
        # Eye habits
        cursor.execute("SELECT COUNT(*) FROM eye_habits")
        habit_count = cursor.fetchone()[0]
        print(f"🏃‍♂️ Eye Habits: {habit_count}")
        
        # Permissions
        cursor.execute("SELECT COUNT(*) FROM admin_permissions")
        permission_count = cursor.fetchone()[0]
        print(f"🔐 Permissions: {permission_count}")
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Start your Flask application")
        print("   2. Navigate to /admin in your browser")
        print("   3. Login with admin/admin123")
        print("   4. Change the default password immediately")
        
    except MySQLdb.Error as e:
        print_error(f"Error showing summary: {e}")

def main():
    """Main setup function"""
    global config
    
    print_header()
    
    # Get configuration
    print_step(1, "Loading Configuration")
    config = get_database_config()
    if not config:
        return False
    
    print_info(f"Host: {config['host']}")
    print_info(f"User: {config['user']}")
    print_info(f"Database: {config['database']}")
    print()
    
    # Create database
    print_step(2, "Creating Database")
    if not create_database(config):
        return False
    print()
    
    # Connect to database
    print_step(3, "Connecting to Database")
    conn = create_database_connection(config)
    if not conn:
        return False
    print_success("Connected to database successfully")
    print()
    
    # Execute SQL setup
    print_step(4, "Executing Database Setup")
    sql_file = "database_setup.sql"
    if not execute_sql_file(conn, sql_file):
        conn.close()
        return False
    print()
    
    # Verify setup
    print_step(5, "Verifying Setup")
    if not verify_setup(conn):
        conn.close()
        return False
    print()
    
    # Show summary
    show_setup_summary(conn)
    
    conn.close()
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Database setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Database setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)