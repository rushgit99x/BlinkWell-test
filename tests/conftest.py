import pytest
import sys
import os
from flask import template_rendered
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app
from models.database import init_db
import MySQLdb

@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    flask_app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'MYSQL_DB': 'blinkwell_test',  # Use test database
        'SECRET_KEY': 'test-secret-key',
        'SERVER_NAME': 'localhost.localdomain'
    })
    
    # Initialize test database
    with flask_app.app_context():
        init_db(flask_app)
        setup_test_database()
    
    yield flask_app
    
    # Cleanup after tests
    with flask_app.app_context():
        cleanup_test_database()


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner"""
    return app.test_cli_runner()


@pytest.fixture
def auth_client(client):
    """Create authenticated test client"""
    # Register and login a test user
    client.post('/register', data={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'testpass123'
    })
    
    client.post('/login', data={
        'username': 'testuser',
        'password': 'testpass123'
    })
    
    return client


@pytest.fixture
def mock_db_connection(monkeypatch):
    """Mock database connection"""
    class MockCursor:
        def __init__(self):
            self.results = []
            self.execute_calls = []
        
        def execute(self, query, params=None):
            self.execute_calls.append((query, params))
            return None
        
        def fetchone(self):
            return self.results[0] if self.results else None
        
        def fetchall(self):
            return self.results
        
        def close(self):
            pass
    
    class MockConnection:
        def __init__(self):
            self.cursor_obj = MockCursor()
        
        def cursor(self, *args, **kwargs):
            return self.cursor_obj
        
        def commit(self):
            pass
        
        def rollback(self):
            pass
        
        def close(self):
            pass
        
        def begin(self):
            pass
    
    mock_conn = MockConnection()
    
    def mock_get_connection():
        return mock_conn
    
    monkeypatch.setattr('config.get_db_connection', mock_get_connection)
    return mock_conn


@contextmanager
def captured_templates(app):
    """Capture templates rendered during testing"""
    recorded = []
    
    def record(sender, template, context, **extra):
        recorded.append((template, context))
    
    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)


def setup_test_database():
    """Set up test database schema"""
    try:
        from config import Config
        conn = MySQLdb.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        
        # Create test database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS blinkwell_test")
        cursor.execute("USE blinkwell_test")
        
        # Create tables (simplified for testing)
        create_tables(cursor)
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error setting up test database: {e}")


def cleanup_test_database():
    """Clean up test database"""
    try:
        from config import Config
        conn = MySQLdb.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("DROP DATABASE IF EXISTS blinkwell_test")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error cleaning up test database: {e}")


def create_tables(cursor):
    """Create necessary tables for testing"""
    tables = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            google_id VARCHAR(255),
            profile_pic TEXT,
            is_google_user TINYINT(1) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS eye_habits (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            category VARCHAR(50),
            icon VARCHAR(50),
            target_frequency VARCHAR(20),
            target_count INT,
            target_unit VARCHAR(20),
            instructions TEXT,
            benefits TEXT,
            difficulty_level VARCHAR(20),
            estimated_time_minutes INT,
            is_active TINYINT(1) DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_habits (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            habit_id INT NOT NULL,
            custom_target_count INT,
            custom_target_unit VARCHAR(20),
            reminder_time TIME,
            reminder_enabled TINYINT(1) DEFAULT 1,
            is_active TINYINT(1) DEFAULT 1,
            start_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS habit_tracking (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            user_habit_id INT NOT NULL,
            habit_id INT NOT NULL,
            date DATE NOT NULL,
            completed_count INT DEFAULT 0,
            target_count INT,
            completion_percentage DECIMAL(5,2),
            completion_time TIME,
            notes TEXT,
            mood_before INT,
            mood_after INT,
            is_completed TINYINT(1) DEFAULT 0,
            streak_day INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_recommendations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            analysis_id INT,
            category VARCHAR(50),
            recommendation_text TEXT,
            priority VARCHAR(20),
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            completed_at TIMESTAMP NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_eye_health_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            gender VARCHAR(10),
            age INT,
            sleep_duration DECIMAL(4,2),
            sleep_quality INT,
            stress_level INT,
            blood_pressure VARCHAR(20),
            heart_rate INT,
            daily_steps INT,
            physical_activity INT,
            height INT,
            weight INT,
            sleep_disorder CHAR(1),
            wake_up_during_night CHAR(1),
            feel_sleepy_during_day CHAR(1),
            caffeine_consumption CHAR(1),
            alcohol_consumption CHAR(1),
            smoking CHAR(1),
            medical_issue CHAR(1),
            ongoing_medication CHAR(1),
            smart_device_before_bed CHAR(1),
            average_screen_time DECIMAL(4,2),
            blue_light_filter CHAR(1),
            discomfort_eye_strain CHAR(1),
            redness_in_eye CHAR(1),
            itchiness_irritation_in_eye CHAR(1),
            dry_eye_disease CHAR(1),
            risk_score DECIMAL(5,2),
            risk_factors TEXT,
            recommendations_saved TINYINT(1) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
    ]
    
    for table_sql in tables:
        cursor.execute(table_sql)