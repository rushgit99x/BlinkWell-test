"""
WAMP Server Configuration for BlinkWell AI Chatbot
This configuration is optimized for local WAMP server development
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class WampConfig:
    """Configuration class for WAMP server environment"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = True
    
    # WAMP MySQL Configuration (default values)
    MYSQL_HOST = os.environ.get('MYSQL_HOST') or 'localhost'
    MYSQL_USER = os.environ.get('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD') or ''  # WAMP default is often empty
    MYSQL_DB = os.environ.get('MYSQL_DB') or 'flask_auth_db'
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT') or 3306)
    
    # MySQL Connection Pool Settings
    MYSQL_POOL_SIZE = 10
    MYSQL_POOL_RECYCLE = 3600
    
    # File Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Chatbot Configuration
    CHATBOT_CONFIDENCE_THRESHOLD = 0.3
    CHATBOT_MAX_RESPONSE_LENGTH = 500
    
    # Session Configuration
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # Security Configuration
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = os.environ.get('WTF_CSRET_KEY') or 'csrf-secret-key'
    
    @classmethod
    def get_mysql_config(cls):
        """Get MySQL configuration as dictionary"""
        return {
            'host': cls.MYSQL_HOST,
            'user': cls.MYSQL_USER,
            'password': cls.MYSQL_PASSWORD,
            'database': cls.MYSQL_DB,
            'port': cls.MYSQL_PORT,
            'charset': 'utf8mb4',
            'autocommit': True,
            'pool_size': cls.MYSQL_POOL_SIZE,
            'pool_recycle': cls.MYSQL_POOL_RECYCLE
        }
    
    @classmethod
    def get_database_url(cls):
        """Get database URL for SQLAlchemy (if needed)"""
        return f"mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DB}?charset=utf8mb4"

# Create a .env file template
ENV_TEMPLATE = """
# WAMP Server Environment Configuration
# Copy this to .env file and update values as needed

# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DB=flask_auth_db
MYSQL_PORT=3306

# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
DEBUG=True

# Security
WTF_CSRET_KEY=your-csrf-secret-key
"""

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(ENV_TEMPLATE.strip())
        print("Created .env file with default WAMP configuration")
        print("Please update the values in .env file as needed")

if __name__ == "__main__":
    create_env_file()
    print("\nWAMP Configuration:")
    print(f"MySQL Host: {WampConfig.MYSQL_HOST}")
    print(f"MySQL User: {WampConfig.MYSQL_USER}")
    print(f"MySQL Database: {WampConfig.MYSQL_DB}")
    print(f"MySQL Port: {WampConfig.MYSQL_PORT}")
    print(f"Debug Mode: {WampConfig.DEBUG}")