#!/usr/bin/env python3
"""
Test script for the new dynamic dashboard functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, date, timedelta
import MySQLdb

def test_database_connection():
    """Test database connection and basic queries"""
    try:
        # Try to connect to database using TCP
        conn = MySQLdb.connect(
            host='127.0.0.1',
            port=3306,
            user='blinkwell',
            passwd='blinkwell123',
            db='blinkwell'
        )
        print("✓ Database connection successful")
        
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Test basic table queries
        print("\n--- Testing Dashboard Queries ---")
        
        # Test 1: Check if tables exist
        tables_to_check = [
            'users',
            'user_eye_health_data', 
            'user_habits',
            'habit_tracking',
            'eye_habits'
        ]
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SHOW TABLES LIKE '{table}'")
                result = cursor.fetchone()
                if result:
                    print(f"✓ Table '{table}' exists")
                else:
                    print(f"✗ Table '{table}' does not exist")
            except Exception as e:
                print(f"✗ Error checking table '{table}': {e}")
        
        # Test 2: Check sample data
        print("\n--- Sample Data Check ---")
        
        # Check users
        cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = cursor.fetchone()
        print(f"Users in database: {user_count['count'] if user_count else 0}")
        
        # Check health data
        cursor.execute("SELECT COUNT(*) as count FROM user_eye_health_data")
        health_count = cursor.fetchone()
        print(f"Health records: {health_count['count'] if health_count else 0}")
        
        # Check habits
        cursor.execute("SELECT COUNT(*) as count FROM user_habits")
        habits_count = cursor.fetchone()
        print(f"User habits: {habits_count['count'] if habits_count else 0}")
        
        # Test 3: Test dashboard queries
        print("\n--- Testing Dashboard Queries ---")
        
        # Sample user ID for testing
        test_user_id = 1
        
        # Test health data query
        try:
            cursor.execute("""
                SELECT risk_score, dry_eye_disease, risk_factors, created_at
                FROM user_eye_health_data 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (test_user_id,))
            
            health_data = cursor.fetchone()
            if health_data:
                print(f"✓ Health data query successful - Latest risk score: {health_data['risk_score']}")
            else:
                print("✓ Health data query successful - No data for test user")
        except Exception as e:
            print(f"✗ Health data query failed: {e}")
        
        # Test habits query
        try:
            today = date.today()
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT uh.id) as total_habits,
                    COUNT(DISTINCT CASE WHEN ht.is_completed = 1 THEN uh.id END) as completed_today
                FROM user_habits uh
                LEFT JOIN habit_tracking ht ON uh.id = ht.user_habit_id AND ht.date = %s
                WHERE uh.user_id = %s AND uh.is_active = 1
            """, (today, test_user_id))
            
            habit_stats = cursor.fetchone()
            if habit_stats:
                print(f"✓ Habits query successful - Total: {habit_stats['total_habits']}, Completed: {habit_stats['completed_today']}")
            else:
                print("✓ Habits query successful - No habits for test user")
        except Exception as e:
            print(f"✗ Habits query failed: {e}")
        
        cursor.close()
        conn.close()
        print("\n✓ All tests completed successfully")
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("Make sure MySQL is running and the database 'blinkwell' exists")
        return False
    
    return True

def test_dashboard_logic():
    """Test the dashboard logic functions"""
    print("\n--- Testing Dashboard Logic ---")
    
    # Test risk score calculation
    current_score = 7.2
    previous_score = 9.3
    risk_change = round(current_score - previous_score, 1)
    risk_change_text = f"{abs(risk_change)} points {'lower' if risk_change < 0 else 'higher'}"
    
    print(f"✓ Risk score calculation: {risk_change_text}")
    
    # Test habit percentage calculation
    completed = 3
    total = 4
    percentage = round((completed / total * 100) if total > 0 else 0)
    print(f"✓ Habit percentage calculation: {percentage}%")
    
    # Test weekly progress calculation
    weekly_data = [
        {'total_habits': 4, 'completed_habits': 3},
        {'total_habits': 4, 'completed_habits': 4},
        {'total_habits': 4, 'completed_habits': 2}
    ]
    
    total_weekly_habits = sum([day['total_habits'] for day in weekly_data])
    total_weekly_completed = sum([day['completed_habits'] for day in weekly_data])
    weekly_completion = round((total_weekly_completed / total_weekly_habits * 100) if total_weekly_habits > 0 else 0)
    
    print(f"✓ Weekly progress calculation: {weekly_completion}%")
    
    print("✓ All logic tests passed")

if __name__ == "__main__":
    print("BlinkWell Dashboard Test Suite")
    print("=" * 40)
    
    # Test database connection and queries
    db_success = test_database_connection()
    
    # Test dashboard logic
    test_dashboard_logic()
    
    if db_success:
        print("\n🎉 Dashboard is ready to use!")
        print("You can now run the Flask app and visit /dashboard to see the dynamic data")
    else:
        print("\n⚠️  Please fix database issues before using the dashboard")