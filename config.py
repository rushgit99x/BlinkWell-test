import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    MYSQL_HOST = os.environ.get('MYSQL_HOST') or 'localhost'
    MYSQL_USER = os.environ.get('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD') or ''
    MYSQL_DB = os.environ.get('MYSQL_DB') or 'b_test8'
    MYSQL_CURSORCLASS = 'DictCursor'

    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID') or 'your-google-client-id'
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET') or 'your-google-client-secret'
    
    # Email Configuration
    SMTP_SERVER = os.environ.get('SMTP_SERVER') or 'smtp.gmail.com'
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME') or None
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD') or None
    SENDER_EMAIL = os.environ.get('SENDER_EMAIL') or None
    SENDER_NAME = os.environ.get('SENDER_NAME') or 'BlinkWell'
    
    # Base URL for email links
    BASE_URL = os.environ.get('BASE_URL') or 'http://localhost:5000'