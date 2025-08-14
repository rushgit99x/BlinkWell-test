#!/usr/bin/env python3
"""
Simple test script for the email notification system
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from services.email_service import EmailService
        print("✓ EmailService imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import EmailService: {e}")
        return False
    
    try:
        from services.notification_scheduler import NotificationScheduler
        print("✓ NotificationScheduler imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NotificationScheduler: {e}")
        return False
    
    try:
        import schedule
        print("✓ schedule library imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import schedule: {e}")
        return False
    
    return True

def test_email_service():
    """Test email service initialization"""
    print("\nTesting EmailService...")
    
    try:
        from services.email_service import EmailService
        from flask import Flask
        
        # Create a minimal Flask app for testing
        app = Flask(__name__)
        app.config.update({
            'SMTP_SERVER': 'smtp.gmail.com',
            'SMTP_PORT': 587,
            'SMTP_USERNAME': 'test@example.com',
            'SMTP_PASSWORD': 'test_password',
            'SENDER_EMAIL': 'test@example.com',
            'SENDER_NAME': 'TestApp',
            'BASE_URL': 'http://localhost:5000'
        })
        
        # Initialize email service
        email_service = EmailService()
        email_service.init_app(app)
        
        print("✓ EmailService initialized successfully")
        print(f"  - SMTP Server: {email_service.smtp_server}")
        print(f"  - SMTP Port: {email_service.smtp_port}")
        print(f"  - Sender: {email_service.sender_email}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test EmailService: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_notification_scheduler():
    """Test notification scheduler initialization"""
    print("\nTesting NotificationScheduler...")
    
    try:
        from services.notification_scheduler import NotificationScheduler
        from flask import Flask
        
        # Create a minimal Flask app for testing
        app = Flask(__name__)
        app.config.update({
            'SMTP_SERVER': 'smtp.gmail.com',
            'SMTP_PORT': 587,
            'SMTP_USERNAME': 'test@example.com',
            'SMTP_PASSWORD': 'test_password',
            'SENDER_EMAIL': 'test@example.com',
            'SENDER_NAME': 'TestApp',
            'BASE_URL': 'http://localhost:5000'
        })
        
        # Initialize scheduler
        scheduler = NotificationScheduler()
        scheduler.init_app(app)
        
        print("✓ NotificationScheduler initialized successfully")
        print(f"  - Running: {scheduler.running}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test NotificationScheduler: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing BlinkWell Email Notification System")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test email service
    if not test_email_service():
        print("\n❌ EmailService tests failed.")
        return False
    
    # Test notification scheduler
    if not test_notification_scheduler():
        print("\n❌ NotificationScheduler tests failed.")
        return False
    
    print("\n🎉 All tests passed! The email notification system is ready to use.")
    print("\nNext steps:")
    print("1. Configure your .env file with email credentials")
    print("2. Start your Flask application")
    print("3. Visit /email-test to test the system")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)