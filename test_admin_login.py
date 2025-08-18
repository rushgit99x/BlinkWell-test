#!/usr/bin/env python3
"""
Test script to verify admin user login functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.admin import get_admin_user_by_username, get_admin_db_connection
from werkzeug.security import check_password_hash, generate_password_hash

def test_admin_user():
    """Test admin user existence and credentials"""
    print("🔍 Testing Admin User Setup")
    print("=" * 40)
    
    try:
        # Test database connection
        print("📡 Testing database connection...")
        conn = get_admin_db_connection()
        cursor = conn.cursor()
        print("✅ Database connection successful")
        
        # Check if admin_users table exists
        cursor.execute("SHOW TABLES LIKE 'admin_users'")
        if not cursor.fetchone():
            print("❌ admin_users table not found!")
            return False
        print("✅ admin_users table exists")
        
        # Check admin user
        print("\n👤 Checking admin user...")
        admin_user = get_admin_user_by_username('admin')
        
        if not admin_user:
            print("❌ Admin user 'admin' not found!")
            return False
        
        print(f"✅ Admin user found:")
        print(f"   ID: {admin_user.id}")
        print(f"   Username: {admin_user.username}")
        print(f"   Email: {admin_user.email}")
        print(f"   Role: {admin_user.role}")
        print(f"   Active: {admin_user.is_active}")
        
        # Test password
        print("\n🔐 Testing password...")
        test_password = 'admin123'
        
        if admin_user.check_password(test_password):
            print("✅ Password 'admin123' is correct")
        else:
            print("❌ Password 'admin123' is incorrect")
            
            # Show current password hash
            print(f"   Current password hash: {admin_user.password_hash}")
            
            # Generate new password hash
            new_hash = generate_password_hash(test_password)
            print(f"   Expected hash for 'admin123': {new_hash}")
            
            # Update password in database
            print("\n🔄 Updating admin password...")
            cursor.execute("UPDATE admin_users SET password_hash = %s WHERE username = 'admin'", (new_hash,))
            conn.commit()
            print("✅ Admin password updated to 'admin123'")
        
        # Check permissions
        print("\n🔑 Checking permissions...")
        cursor.execute("SELECT COUNT(*) FROM admin_permissions")
        permission_count = cursor.fetchone()[0]
        print(f"✅ Found {permission_count} permissions")
        
        # Check sample habits
        print("\n🏃‍♂️ Checking sample habits...")
        cursor.execute("SELECT COUNT(*) FROM eye_habits")
        habit_count = cursor.fetchone()[0]
        print(f"✅ Found {habit_count} sample habits")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 All tests passed! Admin user is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_admin_user():
    """Create admin user if it doesn't exist"""
    print("\n🔧 Creating admin user if needed...")
    
    try:
        conn = get_admin_db_connection()
        cursor = conn.cursor()
        
        # Check if admin user exists
        cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
        if cursor.fetchone()[0] > 0:
            print("✅ Admin user already exists")
            return True
        
        # Create admin user
        password_hash = generate_password_hash('admin123')
        cursor.execute("""
            INSERT INTO admin_users (username, email, password_hash, role) 
            VALUES ('admin', 'admin@blinkwell.com', %s, 'super_admin')
        """, (password_hash,))
        
        conn.commit()
        print("✅ Admin user created successfully")
        print("   Username: admin")
        print("   Password: admin123")
        print("   Role: super_admin")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False

if __name__ == "__main__":
    print("🚀 BlinkWell Admin User Test")
    print("=" * 50)
    
    # Create admin user if needed
    if not create_admin_user():
        print("❌ Failed to create admin user")
        sys.exit(1)
    
    # Test admin user
    if test_admin_user():
        print("\n✅ Admin user setup is complete and working!")
        print("\n📋 You can now:")
        print("   1. Start your Flask app: python app.py")
        print("   2. Navigate to: http://127.0.0.1:5000/admin/login")
        print("   3. Login with: admin / admin123")
    else:
        print("\n❌ Admin user setup has issues!")
        sys.exit(1)