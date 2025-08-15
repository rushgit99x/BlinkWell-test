#!/usr/bin/env python3
"""
Test script for the BlinkWell Settings System
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing settings system imports...")
    
    try:
        from routes.settings import settings_bp
        print("✓ Settings blueprint imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import settings blueprint: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Flask: {e}")
        return False
    
    return True

def test_blueprint_registration():
    """Test if the settings blueprint can be registered"""
    print("\nTesting blueprint registration...")
    
    try:
        from flask import Flask
        from routes.settings import settings_bp
        
        # Create a minimal Flask app for testing
        app = Flask(__name__)
        app.config.update({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key'
        })
        
        # Register the blueprint
        app.register_blueprint(settings_bp)
        
        print("✓ Settings blueprint registered successfully")
        
        # Check if routes are registered
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        settings_routes = [route for route in routes if 'settings' in route]
        
        print(f"  - Registered settings routes: {len(settings_routes)}")
        for route in settings_routes:
            print(f"    * {route}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test blueprint registration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_route_functions():
    """Test if route functions exist and are callable"""
    print("\nTesting route functions...")
    
    try:
        from routes.settings import (
            settings_page, update_profile, update_password,
            update_notification_preferences, update_privacy_settings,
            export_user_data, clear_user_cache, delete_user_account
        )
        
        functions = [
            ('settings_page', settings_page),
            ('update_profile', update_profile),
            ('update_password', update_password),
            ('update_notification_preferences', update_notification_preferences),
            ('update_privacy_settings', update_privacy_settings),
            ('export_user_data', export_user_data),
            ('clear_user_cache', clear_user_cache),
            ('delete_user_account', delete_user_account)
        ]
        
        for name, func in functions:
            if callable(func):
                print(f"✓ {name} is callable")
            else:
                print(f"✗ {name} is not callable")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import route functions: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing BlinkWell Settings System")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test blueprint registration
    if not test_blueprint_registration():
        print("\n❌ Blueprint registration tests failed.")
        return False
    
    # Test route functions
    if not test_route_functions():
        print("\n❌ Route function tests failed.")
        return False
    
    print("\n🎉 All tests passed! The settings system is ready to use.")
    print("\nFeatures available:")
    print("- Profile management (username, email)")
    print("- Password updates")
    print("- Notification preferences")
    print("- Privacy settings")
    print("- Data export")
    print("- Cache management")
    print("- Account deletion")
    print("\nNext steps:")
    print("1. Run the database setup script: database_settings_tables.sql")
    print("2. Start your Flask application")
    print("3. Visit /settings to access the settings page")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)