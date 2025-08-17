#!/usr/bin/env python3
"""
Admin Panel Setup Script for BlinkWell
This script initializes the admin database and creates the first super admin user.
"""

import os
import sys
import getpass
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask
    from models.admin import init_admin_db, create_admin_user
    from config import Config
    from models.database import init_db
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def create_app():
    """Create a minimal Flask app for database operations"""
    app = Flask(__name__)
    app.config.from_object(Config)
    return app

def setup_admin_panel():
    """Set up the admin panel database and create first admin user"""
    print("🚀 Setting up BlinkWell Admin Panel...")
    print("=" * 50)
    
    # Create Flask app
    app = create_app()
    
    with app.app_context():
        try:
            # Initialize databases
            print("📊 Initializing databases...")
            init_db(app)
            init_admin_db(app)
            print("✅ Databases initialized successfully!")
            
            # Check if admin users already exist
            from models.admin import get_all_admin_users
            existing_admins = get_all_admin_users()
            
            if existing_admins:
                print(f"ℹ️  Found {len(existing_admins)} existing admin user(s)")
                return
            
            # Create first super admin user
            print("\n👑 Creating first super administrator...")
            print("Please provide the details for the super admin account:")
            
            while True:
                username = input("Username: ").strip()
                if username:
                    break
                print("❌ Username cannot be empty")
            
            while True:
                email = input("Email: ").strip()
                if email and '@' in email:
                    break
                print("❌ Please enter a valid email address")
            
            while True:
                password = getpass.getpass("Password: ")
                if len(password) >= 8:
                    break
                print("❌ Password must be at least 8 characters long")
            
            password_confirm = getpass.getpass("Confirm Password: ")
            if password != password_confirm:
                print("❌ Passwords do not match!")
                return
            
            # Create the super admin user
            if create_admin_user(username, email, password, 'super_admin'):
                print("✅ Super administrator created successfully!")
                print(f"   Username: {username}")
                print(f"   Email: {email}")
                print(f"   Role: super_admin")
                print("\n🔐 You can now log in to the admin panel at:")
                print("   http://localhost:5000/admin/login")
            else:
                print("❌ Failed to create super administrator")
                print("   This might be due to duplicate username/email")
                
        except Exception as e:
            print(f"❌ Error during setup: {e}")
            print("Please check your database configuration and try again.")
            return False
    
    return True

def main():
    """Main function"""
    print("BlinkWell Admin Panel Setup")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found!")
        print("Please run this script from the BlinkWell project root directory.")
        sys.exit(1)
    
    # Run setup
    if setup_admin_panel():
        print("\n🎉 Admin panel setup completed successfully!")
        print("\nNext steps:")
        print("1. Start your Flask application: python app.py")
        print("2. Navigate to: http://localhost:5000/admin/login")
        print("3. Log in with your super admin credentials")
        print("4. Create additional admin users as needed")
    else:
        print("\n❌ Admin panel setup failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()