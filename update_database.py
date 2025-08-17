#!/usr/bin/env python3
"""
Script to update the BlinkWell database schema for admin panel support
"""

import os
import MySQLdb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection"""
    try:
        connection = MySQLdb.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', ''),
            database=os.environ.get('MYSQL_DB', 'b_test8'),  # Using your database name
            charset='utf8mb4'
        )
        return connection
    except MySQLdb.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def update_database_schema():
    """Update database schema to add admin columns"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("SHOW COLUMNS FROM users LIKE 'is_admin'")
        has_is_admin = cursor.fetchone()
        
        cursor.execute("SHOW COLUMNS FROM users LIKE 'is_active'")
        has_is_active = cursor.fetchone()
        
        cursor.execute("SHOW COLUMNS FROM users LIKE 'last_login'")
        has_last_login = cursor.fetchone()
        
        # Add missing columns
        if not has_is_admin:
            print("Adding is_admin column...")
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE AFTER is_google_user")
            print("✅ is_admin column added")
        else:
            print("ℹ️ is_admin column already exists")
        
        if not has_is_active:
            print("Adding is_active column...")
            cursor.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE AFTER is_admin")
            print("✅ is_active column added")
        else:
            print("ℹ️ is_active column already exists")
        
        if not has_last_login:
            print("Adding last_login column...")
            cursor.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP NULL AFTER updated_at")
            print("✅ last_login column added")
        else:
            print("ℹ️ last_login column already exists")
        
        # Update existing users to be active
        cursor.execute("UPDATE users SET is_active = TRUE WHERE is_active IS NULL")
        updated_rows = cursor.rowcount
        if updated_rows > 0:
            print(f"✅ Updated {updated_rows} users to be active")
        
        # Make the first user an admin for testing
        cursor.execute("UPDATE users SET is_admin = TRUE WHERE id = 1")
        if cursor.rowcount > 0:
            print("✅ Made user ID 1 an admin")
        
        conn.commit()
        print("\n🎉 Database schema updated successfully!")
        return True
        
    except MySQLdb.Error as e:
        print(f"Error updating database: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def show_current_schema():
    """Show current users table structure"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("DESCRIBE users")
        columns = cursor.fetchall()
        
        print("\n📋 Current users table structure:")
        print("-" * 50)
        for column in columns:
            print(f"{column[0]:<20} {column[1]:<20} {column[2]:<10} {column[3]:<10}")
        
        # Show sample user data
        cursor.execute("SELECT id, username, email, is_admin, is_active FROM users LIMIT 3")
        users = cursor.fetchall()
        
        print("\n👥 Sample user data:")
        print("-" * 50)
        for user in users:
            print(f"ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Admin: {user[3]}, Active: {user[4]}")
        
    except MySQLdb.Error as e:
        print(f"Error showing schema: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function"""
    print("=== BlinkWell Database Schema Updater ===\n")
    
    print("This script will update your database to support the admin panel.")
    print("Make sure you have backed up your database before proceeding.\n")
    
    response = input("Do you want to continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    print("\n🔄 Updating database schema...")
    
    if update_database_schema():
        print("\n📊 Showing updated schema...")
        show_current_schema()
        
        print("\n🎯 Next steps:")
        print("1. Restart your Flask application")
        print("2. Try logging in again")
        print("3. Access the admin panel at /admin")
    else:
        print("\n❌ Failed to update database schema")
        print("Please check your database connection and try again")

if __name__ == "__main__":
    main()