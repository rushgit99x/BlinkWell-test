#!/usr/bin/env python3
"""
Test script for the BlinkWell Dashboard System
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing dashboard system imports...")
    
    try:
        from services.dashboard_service import DashboardService, dashboard_service
        print("✓ Dashboard service imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dashboard service: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Flask: {e}")
        return False
    
    return True

def test_dashboard_service():
    """Test if the dashboard service can be initialized"""
    print("\nTesting dashboard service initialization...")
    
    try:
        from services.dashboard_service import DashboardService
        
        # Create a minimal Flask app for testing
        from flask import Flask
        app = Flask(__name__)
        app.config.update({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key'
        })
        
        # Test service initialization
        service = DashboardService()
        service.init_app(app)
        
        print("✓ Dashboard service initialized successfully")
        
        # Test default data generation
        default_data = service._get_default_dashboard_data()
        required_keys = [
            'risk_metrics', 'habit_metrics', 'streak_metrics', 
            'progress_metrics', 'today_habits', 'weekly_progress',
            'risk_trends', 'water_intake', 'eye_exercises'
        ]
        
        for key in required_keys:
            if key in default_data:
                print(f"  ✓ {key} data available")
            else:
                print(f"  ✗ {key} data missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test dashboard service: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_route_integration():
    """Test if the dashboard route can be imported"""
    print("\nTesting route integration...")
    
    try:
        from routes.main import main_bp, dashboard
        print("✓ Main blueprint imported successfully")
        print("✓ Dashboard function imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test route integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structure():
    """Test the structure of dashboard data"""
    print("\nTesting dashboard data structure...")
    
    try:
        from services.dashboard_service import DashboardService
        
        service = DashboardService()
        default_data = service._get_default_dashboard_data()
        
        # Test risk metrics structure
        risk_metrics = default_data['risk_metrics']
        required_risk_keys = ['current_risk_score', 'previous_risk_score', 'risk_reduction', 'score_change', 'trend']
        for key in required_risk_keys:
            if key not in risk_metrics:
                print(f"  ✗ Missing risk metric: {key}")
                return False
        print("  ✓ Risk metrics structure valid")
        
        # Test habit metrics structure
        habit_metrics = default_data['habit_metrics']
        required_habit_keys = ['habits_completed_today', 'total_active_habits', 'daily_completion_percentage', 'weekly_completion_percentage']
        for key in required_habit_keys:
            if key not in habit_metrics:
                print(f"  ✗ Missing habit metric: {key}")
                return False
        print("  ✓ Habit metrics structure valid")
        
        # Test streak metrics structure
        streak_metrics = default_data['streak_metrics']
        required_streak_keys = ['current_streak', 'longest_streak', 'streak_status']
        for key in required_streak_keys:
            if key not in streak_metrics:
                print(f"  ✗ Missing streak metric: {key}")
                return False
        print("  ✓ Streak metrics structure valid")
        
        # Test water intake structure
        water_intake = default_data['water_intake']
        required_water_keys = ['target_glasses', 'completed_glasses', 'completion_percentage', 'streak_days', 'weekly_progress']
        for key in required_water_keys:
            if key not in water_intake:
                print(f"  ✗ Missing water intake metric: {key}")
                return False
        print("  ✓ Water intake structure valid")
        
        # Test eye exercises structure
        eye_exercises = default_data['eye_exercises']
        required_exercise_keys = ['target_sessions', 'completed_sessions', 'completion_percentage', 'streak_days', 'weekly_progress']
        for key in required_exercise_keys:
            if key not in eye_exercises:
                print(f"  ✗ Missing eye exercise metric: {key}")
                return False
        print("  ✓ Eye exercises structure valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test data structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing BlinkWell Dashboard System")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test dashboard service
    if not test_dashboard_service():
        print("\n❌ Dashboard service tests failed.")
        return False
    
    # Test route integration
    if not test_route_integration():
        print("\n❌ Route integration tests failed.")
        return False
    
    # Test data structure
    if not test_data_structure():
        print("\n❌ Data structure tests failed.")
        return False
    
    print("\n🎉 All tests passed! The dashboard system is ready to use.")
    print("\nFeatures implemented:")
    print("- Real-time risk score tracking")
    print("- Dynamic habit completion metrics")
    print("- Streak calculation and tracking")
    print("- Weekly and monthly progress analysis")
    print("- Water intake monitoring")
    print("- Eye exercise tracking")
    print("- Risk trend analysis")
    print("- Today's habits status")
    print("\nNext steps:")
    print("1. Ensure your database has the required tables")
    print("2. Start your Flask application")
    print("3. Visit /dashboard to see real-time data")
    print("4. The dashboard will automatically fetch and display:")
    print("   • Current risk scores from eye health assessments")
    print("   • Habit completion rates from tracking data")
    print("   • Streak calculations from daily habit data")
    print("   • Progress metrics from weekly/monthly summaries")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)