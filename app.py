from flask import Flask, send_from_directory
from flask_login import LoginManager
from config import Config
from routes.auth import auth_bp
from routes.main import main_bp
from routes.eye_detection import eye_detection_bp
from models.database import init_db
from models.user import load_user
from oauth import init_oauth
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

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)
app.register_blueprint(eye_detection_bp)
app.register_blueprint(habits_bp)

if __name__ == '__main__':
    app.run(debug=True)