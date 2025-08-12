#!/usr/bin/env python3
"""
Test script for the BlinkWell AI Chatbot
Run this script to test the chatbot functionality without starting the full web application.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chatbot_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.chatbot import ChatbotAI
        print("✓ ChatbotAI class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ChatbotAI: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False
    
    return True

def test_chatbot_functionality():
    """Test basic chatbot functionality with mock data"""
    print("\nTesting chatbot functionality...")
    
    try:
        from models.chatbot import ChatbotAI
        
        # Mock database configuration
        mock_db_config = {
            'host': 'localhost',
            'user': 'test_user',
            'password': 'test_password',
            'database': 'test_db'
        }
        
        # Create chatbot instance
        chatbot = ChatbotAI(mock_db_config)
        print("✓ ChatbotAI instance created successfully")
        
        # Test with sample knowledge base
        sample_data = [
            {
                'question': 'What is this web app about?',
                'answer': 'This is a comprehensive web application for eye health.',
                'category': 'General',
                'keywords': 'web app, features, purpose'
            },
            {
                'question': 'How do I use the eye disease detection?',
                'answer': 'Upload an image and the AI will analyze it.',
                'category': 'Eye Detection',
                'keywords': 'eye detection, upload image, AI analysis'
            }
        ]
        
        # Manually set knowledge base for testing
        chatbot.knowledge_base = sample_data
        chatbot.vectors = None  # Will be set when load_knowledge_base is called
        
        # Test response generation
        test_questions = [
            "What is this app about?",
            "How do I detect eye diseases?",
            "Hello there!",
            "Thank you for helping",
            "What is the weather like?"
        ]
        
        print("\nTesting responses to various questions:")
        for question in test_questions:
            response = chatbot.get_response(question)
            print(f"\nQ: {question}")
            print(f"A: {response['response'][:100]}...")
            print(f"   Type: {response['type']}, Confidence: {response['confidence']:.2f}")
        
        print("\n✓ Chatbot functionality tests completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Chatbot functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_schema():
    """Test if the database schema is properly defined"""
    print("\nTesting database schema...")
    
    try:
        # Read the database.sql file
        with open('database.sql', 'r') as f:
            schema = f.read()
        
        # Check for required tables
        required_tables = ['knowledge_base', 'chat_history']
        for table in required_tables:
            if table in schema:
                print(f"✓ Table '{table}' found in schema")
            else:
                print(f"✗ Table '{table}' not found in schema")
                return False
        
        # Check for sample data
        if 'INSERT INTO knowledge_base' in schema:
            print("✓ Sample FAQ data found in schema")
        else:
            print("✗ Sample FAQ data not found in schema")
            return False
        
        print("✓ Database schema tests completed successfully")
        return True
        
    except FileNotFoundError:
        print("✗ database.sql file not found")
        return False
    except Exception as e:
        print(f"✗ Database schema test failed: {e}")
        return False

def test_routes():
    """Test if all required routes are properly defined"""
    print("\nTesting route definitions...")
    
    try:
        # Test chatbot routes
        from routes.chatbot import chatbot_bp
        print("✓ Chatbot blueprint imported successfully")
        
        # Test admin routes
        from routes.admin import admin_bp
        print("✓ Admin blueprint imported successfully")
        
        # Check if routes are registered
        chatbot_routes = [rule.rule for rule in chatbot_bp.url_map.iter_rules()]
        admin_routes = [rule.rule for rule in admin_bp.url_map.iter_rules()]
        
        expected_chatbot_routes = [
            '/chat',
            '/chat/api/send',
            '/chat/api/suggestions',
            '/chat/api/refresh',
            '/chat/api/faq',
            '/chat/faq'
        ]
        
        expected_admin_routes = [
            '/admin/knowledge-base',
            '/admin/knowledge-base/add',
            '/admin/knowledge-base/import',
            '/admin/knowledge-base/export'
        ]
        
        for route in expected_chatbot_routes:
            if route in chatbot_routes:
                print(f"✓ Chatbot route '{route}' found")
            else:
                print(f"✗ Chatbot route '{route}' not found")
                return False
        
        for route in expected_admin_routes:
            if route in admin_routes:
                print(f"✓ Admin route '{route}' found")
            else:
                print(f"✗ Admin route '{route}' not found")
                return False
        
        print("✓ Route tests completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Route tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_templates():
    """Test if all required templates exist"""
    print("\nTesting template files...")
    
    required_templates = [
        'templates/chatbot/chat.html',
        'templates/chatbot/faq.html',
        'templates/admin/knowledge_base.html'
    ]
    
    for template in required_templates:
        if os.path.exists(template):
            print(f"✓ Template '{template}' found")
        else:
            print(f"✗ Template '{template}' not found")
            return False
    
    print("✓ Template tests completed successfully")
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("BlinkWell AI Chatbot Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_chatbot_imports),
        ("Database Schema Tests", test_database_schema),
        ("Template Tests", test_templates),
        ("Route Tests", test_routes),
        ("Chatbot Functionality Tests", test_chatbot_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The chatbot system is ready to use.")
        print("\nNext steps:")
        print("1. Run the database.sql script to create required tables")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start your Flask application: python app.py")
        print("4. Access the chatbot at /chat")
        print("5. Manage knowledge base at /admin/knowledge-base")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())