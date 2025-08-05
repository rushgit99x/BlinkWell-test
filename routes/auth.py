from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_user, logout_user, login_required
from models.user import register_user, authenticate_user, register_google_user, get_user_by_google_id
import re
from authlib.common.security import generate_token

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Validation
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return render_template('register.html')
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            flash('Please enter a valid email address.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        # Register user
        if register_user(username, email, password):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('Username or email already exists.', 'error')
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate_user(username, password)
        if user:
            login_user(user)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.dashboard'))
        
        flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@auth_bp.route('/login/google')
def google_login():
    oauth = current_app.config['OAUTH']
    nonce = generate_token()
    session['google_nonce'] = nonce
    redirect_uri = url_for('auth.google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri, nonce=nonce)

@auth_bp.route('/login/google/callback')
def google_callback():
    oauth = current_app.config['OAUTH']
    try:
        token = oauth.google.authorize_access_token()
        nonce = session.pop('google_nonce', None)
        if not nonce:
            flash('Google login failed: Nonce not found in session.', 'error')
            return redirect(url_for('auth.login'))
        
        google_user = oauth.google.parse_id_token(token, nonce=nonce)
        
        user = get_user_by_google_id(google_user['sub'])
        
        if not user:
            # Generate unique username
            base_username = google_user['email'].split('@')[0]
            username = base_username
            count = 1
            conn = current_app.config['get_db_connection']()
            cursor = conn.cursor()
            while True:
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
                if cursor.fetchone()[0] == 0:
                    break
                username = f"{base_username}{count}"
                count += 1
            cursor.close()
            conn.close()
            
            # Register new Google user
            register_google_user(
                google_id=google_user['sub'],
                username=username,
                email=google_user['email'],
                profile_pic=google_user.get('picture')
            )
            user = get_user_by_google_id(google_user['sub'])
        
        login_user(user)
        session['google_token'] = token['access_token']
        flash('Google login successful!', 'success')
        return redirect(url_for('main.dashboard'))
    except Exception as e:
        flash(f'Google login failed: {str(e)}', 'error')
        return redirect(url_for('auth.login'))
    
@auth_bp.route('/logout')
@login_required
def logout():
    session.pop('google_token', None)
    session.pop('google_nonce', None)
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))