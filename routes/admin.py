from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, jsonify
from werkzeug.utils import secure_filename
from flask_login import login_required
from models.admin import (
	authenticate_admin, create_admin_tables_and_seed, log_admin_activity,
	create_eye_habit, list_eye_habits, get_eye_habit_by_id, update_eye_habit, delete_eye_habit,
	list_admin_users, create_admin_user, get_admin_by_id, update_admin_user, delete_admin_user,
	update_admin_profile, set_admin_password,
)


admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def _ensure_admin_tables():
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
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor()
	cursor.execute("SELECT COUNT(*) FROM users")
	users_count = cursor.fetchone()[0]
	cursor.execute("SELECT COUNT(*) FROM eye_habits")
	habits_count = cursor.fetchone()[0]
	cursor.close()
	conn.close()
	return render_template('admin/dashboard.html', users_count=users_count, habits_count=habits_count, admin=session['admin'])


# Eye habits CRUD -----------------------------------------------------------
@admin_bp.route('/eye-habits', methods=['GET', 'POST'])
def eye_habits():
	guard = _require_admin()
	if guard:
		return guard
	if request.method == 'POST':
		action = request.form.get('action') or 'create'
		try:
			if action == 'delete':
				habit_id = int(request.form.get('habit_id'))
				delete_eye_habit(habit_id)
				log_admin_activity(session['admin']['id'], 'delete_eye_habit', f"Deleted habit id: {habit_id}")
				flash('Eye habit deleted', 'success')
			elif action == 'update':
				habit_id = int(request.form.get('habit_id'))
				fields = {
					'name': request.form.get('name'),
					'description': request.form.get('description'),
					'category': request.form.get('category'),
					'icon': request.form.get('icon') or None,
					'target_frequency': request.form.get('target_frequency') or 'daily',
					'target_count': int(request.form.get('target_count') or 1),
					'target_unit': request.form.get('target_unit') or 'times',
					'instructions': request.form.get('instructions') or None,
					'benefits': request.form.get('benefits') or None,
					'difficulty_level': request.form.get('difficulty_level') or 'easy',
					'estimated_time_minutes': int(request.form.get('estimated_time_minutes') or 5),
					'is_active': 1 if request.form.get('is_active', '1') == '1' else 0,
				}
				update_eye_habit(habit_id, **fields)
				log_admin_activity(session['admin']['id'], 'update_eye_habit', f"Updated habit id: {habit_id}")
				flash('Eye habit updated', 'success')
			else:  # create
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
		except Exception as exc:
			flash(f'Action failed: {exc}', 'error')
		return redirect(url_for('admin.eye_habits'))
	rows = list_eye_habits(200)
	return render_template('admin/eye_habits.html', habits=rows, admin=session['admin'])


@admin_bp.route('/eye-habits/<int:habit_id>')
def eye_habit_detail(habit_id: int):
	guard = _require_admin()
	if guard:
		return guard
	habit = get_eye_habit_by_id(habit_id)
	if not habit:
		flash('Habit not found', 'error')
		return redirect(url_for('admin.eye_habits'))
	return render_template('admin/eye_habit_edit.html', habit=habit, admin=session['admin'])


# Admin users management ----------------------------------------------------
@admin_bp.route('/users', methods=['GET', 'POST'])
def admin_users():
	guard = _require_admin()
	if guard:
		return guard
	if request.method == 'POST':
		try:
			username = request.form.get('username')
			email = request.form.get('email')
			password = request.form.get('password')
			is_superadmin = 1 if request.form.get('is_superadmin', '0') == '1' else 0
			is_active = 1 if request.form.get('is_active', '1') == '1' else 0
			create_admin_user(username, email, password, is_superadmin, is_active)
			log_admin_activity(session['admin']['id'], 'create_admin_user', f"Created admin: {username}")
			flash('Admin user created', 'success')
		except Exception as exc:
			flash(f'Failed to create admin: {exc}', 'error')
		return redirect(url_for('admin.admin_users'))
	users = list_admin_users(200)
	return render_template('admin/admin_users.html', users=users, admin=session['admin'])


@admin_bp.route('/users/<int:admin_id>', methods=['GET', 'POST'])
def admin_user_detail(admin_id: int):
	guard = _require_admin()
	if guard:
		return guard
	if request.method == 'POST':
		action = request.form.get('action')
		try:
			if action == 'delete':
				delete_admin_user(admin_id)
				log_admin_activity(session['admin']['id'], 'delete_admin_user', f"Deleted admin id: {admin_id}")
				flash('Admin user deleted', 'success')
				return redirect(url_for('admin.admin_users'))
			elif action == 'password':
				new_password = request.form.get('password')
				set_admin_password(admin_id, new_password)
				log_admin_activity(session['admin']['id'], 'set_admin_password', f"Updated password for admin id: {admin_id}")
				flash('Password updated', 'success')
			else:
				fields = {
					'username': request.form.get('username'),
					'email': request.form.get('email'),
					'is_superadmin': 1 if request.form.get('is_superadmin', '0') == '1' else 0,
					'is_active': 1 if request.form.get('is_active', '1') == '1' else 0,
				}
				update_admin_user(admin_id, **fields)
				log_admin_activity(session['admin']['id'], 'update_admin_user', f"Updated admin id: {admin_id}")
				flash('Admin user updated', 'success')
		except Exception as exc:
			flash(f'Failed to update admin: {exc}', 'error')
		return redirect(url_for('admin.admin_user_detail', admin_id=admin_id))
	user = get_admin_by_id(admin_id)
	if not user:
		flash('Admin not found', 'error')
		return redirect(url_for('admin.admin_users'))
	return render_template('admin/admin_user_edit.html', user=user, admin=session['admin'])


# Admin profile -------------------------------------------------------------
@admin_bp.route('/profile', methods=['GET', 'POST'])
def profile():
	guard = _require_admin()
	if guard:
		return guard
	if request.method == 'POST':
		try:
			username = request.form.get('username') or None
			email = request.form.get('email') or None
			password = request.form.get('password') or None
			update_admin_profile(session['admin']['id'], username=username, email=email, password=password)
			if username:
				session['admin']['username'] = username
			if email:
				session['admin']['email'] = email
			log_admin_activity(session['admin']['id'], 'update_profile', 'Updated profile')
			flash('Profile updated', 'success')
		except Exception as exc:
			flash(f'Failed to update profile: {exc}', 'error')
		return redirect(url_for('admin.profile'))
	return render_template('admin/profile.html', admin=session['admin'])


# Training and datasets -----------------------------------------------------
_training_status = {
	'image': {'running': False, 'started_at': None, 'ended_at': None, 'message': ''},
	'text': {'running': False, 'started_at': None, 'ended_at': None, 'message': ''},
}


@admin_bp.route('/training', methods=['GET', 'POST'])
def training():
	guard = _require_admin()
	if guard:
		return guard
	if request.method == 'POST':
		image_dataset = request.form.get('image_dataset_path')
		text_dataset = request.form.get('text_dataset_path')
		image_epochs = int(request.form.get('image_epochs') or 25)
		text_epochs = int(request.form.get('text_epochs') or 150)
		import subprocess, sys, datetime
		try:
			if image_dataset:
				_training_status['image'] = {'running': True, 'started_at': datetime.datetime.utcnow().isoformat() + 'Z', 'ended_at': None, 'message': 'Training image model...'}
				proc = subprocess.Popen([
					sys.executable, 'train_model.py', '--dataset_path', image_dataset, '--epochs', str(image_epochs), '--model_save_path', 'models/best_eye_model.pth', '--plot_results'
				])
				# Detach a lightweight waiter that flips status when done
				def _wait_image(p):
					code = p.wait()
					_training_status['image']['running'] = False
					_training_status['image']['ended_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
					_training_status['image']['message'] = 'Training completed' if code == 0 else f'Training failed (exit {code})'
				import threading
				threading.Thread(target=_wait_image, args=(proc,), daemon=True).start()
				log_admin_activity(session['admin']['id'], 'train_image_model', f"dataset={image_dataset}, epochs={image_epochs}")
			if text_dataset:
				_training_status['text'] = {'running': True, 'started_at': datetime.datetime.utcnow().isoformat() + 'Z', 'ended_at': None, 'message': 'Training text model...'}
				proc2 = subprocess.Popen([
					sys.executable, 'train_text_model.py', '--dataset_path', text_dataset, '--epochs', str(text_epochs), '--model_save_path', 'models/best_text_model.pth'
				])
				def _wait_text(p):
					code = p.wait()
					_training_status['text']['running'] = False
					_training_status['text']['ended_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
					_training_status['text']['message'] = 'Training completed' if code == 0 else f'Training failed (exit {code})'
				import threading
				threading.Thread(target=_wait_text, args=(proc2,), daemon=True).start()
				log_admin_activity(session['admin']['id'], 'train_text_model', f"dataset={text_dataset}, epochs={text_epochs}")
			flash('Training started in background. Check logs and models directory for progress.', 'info')
		except Exception as exc:
			flash(f'Failed to start training: {exc}', 'error')
		return redirect(url_for('admin.training'))
	return render_template('admin/training.html', admin=session['admin'])


@admin_bp.route('/training/status')
def training_status():
	guard = _require_admin()
	if guard:
		return guard
	return jsonify(_training_status)


@admin_bp.route('/datasets/images', methods=['GET', 'POST'])
def datasets_images():
	guard = _require_admin()
	if guard:
		return guard
	import os, time, uuid
	base_path = os.path.abspath(os.path.join(current_app.root_path, 'datasets', 'eyes'))
	dry_dir = os.path.join(base_path, 'dry_eyes')
	healthy_dir = os.path.join(base_path, 'no_dry_eyes')
	os.makedirs(dry_dir, exist_ok=True)
	os.makedirs(healthy_dir, exist_ok=True)
	allowed = {'.png', '.jpg', '.jpeg'}
	if request.method == 'POST':
		try:
			category = request.form.get('category')
			if category not in {'dry_eyes', 'no_dry_eyes'}:
				raise ValueError('Invalid category')
			target_dir = dry_dir if category == 'dry_eyes' else healthy_dir
			files = request.files.getlist('images')
			saved = 0
			for f in files:
				if not f or not f.filename:
					continue
				ext = os.path.splitext(f.filename)[1].lower()
				if ext not in allowed:
					continue
				name = secure_filename(f.filename)
				unique = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{name}"
				f.save(os.path.join(target_dir, unique))
				saved += 1
			flash(f'Uploaded {saved} images to {category.replace("_", " ")}', 'success')
		except Exception as exc:
			flash(f'Upload failed: {exc}', 'error')
		return redirect(url_for('admin.datasets_images'))
	def count_images(path):
		return sum(1 for fn in os.listdir(path) if os.path.splitext(fn)[1].lower() in allowed)
	dry_count = count_images(dry_dir)
	healthy_count = count_images(healthy_dir)
	return render_template('admin/datasets_images.html', admin=session['admin'], base_path=base_path, dry_count=dry_count, healthy_count=healthy_count)

