from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_required
from models.admin import authenticate_admin, create_admin_tables_and_seed, log_admin_activity, create_eye_habit, list_eye_habits


admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def _ensure_admin_tables():
	# Initialize tables if not present
	try:
		create_admin_tables_and_seed()
	except Exception as exc:
		current_app.logger.error(f"Failed to ensure admin tables: {exc}")


def _require_admin():
	if 'admin' not in session:
		return redirect(url_for('admin.login'))
	return None


@admin_bp.before_app_request
def _before_any_request():
	# Make sure admin tables exist
	_ensure_admin_tables()


@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'POST':
		username = request.form.get('username')
		password = request.form.get('password')
		admin = authenticate_admin(username, password)
		if admin:
			session['admin'] = admin
			log_admin_activity(admin['id'], 'login', f"Admin {admin['username']} logged in")
			flash('Welcome to Admin Panel', 'success')
			return redirect(url_for('admin.dashboard'))
		flash('Invalid credentials', 'error')
	return render_template('admin/login.html')


@admin_bp.route('/logout')
def logout():
	admin = session.pop('admin', None)
	if admin:
		log_admin_activity(admin['id'], 'logout', f"Admin {admin['username']} logged out")
	flash('You have been logged out from admin.', 'info')
	return redirect(url_for('admin.login'))


@admin_bp.route('/')
def root_redirect():
	if 'admin' in session:
		return redirect(url_for('admin.dashboard'))
	return redirect(url_for('admin.login'))


@admin_bp.route('/dashboard')
def dashboard():
	guard = _require_admin()
	if guard:
		return guard
	# Minimal metrics
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor()
	cursor.execute("SELECT COUNT(*) FROM users")
	users_count = cursor.fetchone()[0]
	cursor.execute("SELECT COUNT(*) FROM eye_habits")
	habits_count = cursor.fetchone()[0]
	cursor.close()
	conn.close()
	return render_template('admin/dashboard.html', users_count=users_count, habits_count=habits_count, admin=session['admin'])


@admin_bp.route('/eye-habits', methods=['GET', 'POST'])
def eye_habits():
	guard = _require_admin()
	if guard:
		return guard
	if request.method == 'POST':
		name = request.form.get('name')
		description = request.form.get('description')
		category = request.form.get('category')
		icon = request.form.get('icon') or None
		target_frequency = request.form.get('target_frequency') or 'daily'
		target_count = int(request.form.get('target_count') or 1)
		target_unit = request.form.get('target_unit') or 'times'
		instructions = request.form.get('instructions') or None
		benefits = request.form.get('benefits') or None
		difficulty_level = request.form.get('difficulty_level') or 'easy'
		estimated_time_minutes = int(request.form.get('estimated_time_minutes') or 5)
		is_active = 1 if request.form.get('is_active', '1') == '1' else 0
		create_eye_habit(
			name, description, category, icon,
			target_frequency, target_count, target_unit,
			instructions, benefits, difficulty_level,
			estimated_time_minutes, is_active
		)
		log_admin_activity(session['admin']['id'], 'create_eye_habit', f"Created habit: {name}")
		flash('Eye habit created', 'success')
		return redirect(url_for('admin.eye_habits'))
	rows = list_eye_habits(200)
	return render_template('admin/eye_habits.html', habits=rows, admin=session['admin'])

