from flask import Flask, send_from_directory
from flask_login import LoginManager
from config import Config
from routes.auth import auth_bp
from routes.main import main_bp
from routes.eye_detection import eye_detection_bp
from models.database import init_db
from models.user import load_user
from oauth import init_oauth
from routes.chatbot import chatbot_bp, initialize_chatbot
from routes.notifications import notifications_bp
from routes.settings import settings_bp
from services.email_service import email_service
from services.notification_scheduler import notification_scheduler
import os

# Add this to your app.py after the existing imports
from routes.habits import habits_bp

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('temp_uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/exports', exist_ok=True)

# Initialize database
init_db(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize OAuth
init_oauth(app)

# Register user_loader
login_manager.user_loader(load_user)

# Initialize email service
email_service.init_app(app)

# Initialize notification scheduler
notification_scheduler.init_app(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)
app.register_blueprint(eye_detection_bp)
app.register_blueprint(habits_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(notifications_bp)
app.register_blueprint(settings_bp)

# @app.before_first_request
# def init_chatbot():
#     initialize_chatbot()

if __name__ == '__main__':
    # Start the notification scheduler
    notification_scheduler.start()
    app.run(debug=True)