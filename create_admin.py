#!/usr/bin/env python3
"""
Script to create an admin user for BlinkWell
Run this script to create your first admin user
"""

import os
import sys
from werkzeug.security import generate_password_hash
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
            database=os.environ.get('MYSQL_DB', 'blinkwell'),
            charset='utf8mb4'
        )
        return connection
    except MySQLdb.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def create_admin_user(username, email, password):
    """Create an admin user"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            print("User already exists with this username or email")
            return False
        
        # Create admin user
        password_hash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, is_admin, is_active) 
            VALUES (%s, %s, %s, %s, %s)
        """, (username, email, password_hash, True, True))
        
        conn.commit()
        print(f"Admin user '{username}' created successfully!")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Admin privileges: Enabled")
        return True
        
    except MySQLdb.Error as e:
        print(f"Error creating admin user: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function"""
    print("=== BlinkWell Admin User Creator ===\n")
    
    # Get user input
    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty")
        return
    
    email = input("Enter email: ").strip()
    if not email or '@' not in email:
        print("Please enter a valid email address")
        return
    
    password = input("Enter password: ").strip()
    if len(password) < 6:
        print("Password must be at least 6 characters long")
        return
    
    confirm_password = input("Confirm password: ").strip()
    if password != confirm_password:
        print("Passwords do not match")
        return
    
    print(f"\nCreating admin user '{username}'...")
    
    if create_admin_user(username, email, password):
        print("\n✅ Admin user created successfully!")
        print("You can now log in to the admin panel at /admin")
    else:
        print("\n❌ Failed to create admin user")
        print("Please check your database connection and try again")

if __name__ == "__main__":
    main()