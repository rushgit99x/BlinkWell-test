from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
import MySQLdb
import os


def create_admin_tables_and_seed():
	"""Create admin-specific tables and ensure a default admin exists.

	Tables created (MyISAM to match existing schema):
	- admin_users
	- admin_activity_logs
	"""
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor()

	# Create admin_users table
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS admin_users (
		  id INT NOT NULL AUTO_INCREMENT,
		  username VARCHAR(80) NOT NULL,
		  email VARCHAR(120) NOT NULL,
		  password_hash VARCHAR(255) NOT NULL,
		  is_superadmin TINYINT(1) DEFAULT '0',
		  is_active TINYINT(1) DEFAULT '1',
		  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
		  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
		  PRIMARY KEY (id),
		  UNIQUE KEY username (username),
		  UNIQUE KEY email (email)
		) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
		"""
	)

	# Create admin_activity_logs table
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS admin_activity_logs (
		  id INT NOT NULL AUTO_INCREMENT,
		  admin_id INT NOT NULL,
		  action VARCHAR(100) NOT NULL,
		  details TEXT,
		  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
		  PRIMARY KEY (id),
		  KEY admin_id (admin_id),
		  KEY action (action)
		) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
		"""
	)

	# Ensure a default admin exists (configurable via env)
	admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
	admin_email = os.environ.get('ADMIN_EMAIL', 'admin@blinkwell.com')
	admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')

	cursor.execute("SELECT COUNT(*) FROM admin_users")
	count = cursor.fetchone()[0]
	if count == 0:
		password_hash = generate_password_hash(admin_password)
		cursor.execute(
			"INSERT INTO admin_users (username, email, password_hash, is_superadmin) VALUES (%s, %s, %s, %s)",
			(admin_username, admin_email, password_hash, 1),
		)
		conn.commit()

	cursor.close()
	conn.close()


def authenticate_admin(username: str, password: str):
	"""Return admin dict if credentials valid else None."""
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute("SELECT * FROM admin_users WHERE username = %s AND is_active = 1", (username,))
	admin = cursor.fetchone()
	cursor.close()
	conn.close()
	if not admin:
		return None
	if check_password_hash(admin['password_hash'], password):
		return {
			'id': admin['id'],
			'username': admin['username'],
			'email': admin['email'],
			'is_superadmin': admin['is_superadmin'] == 1,
		}
	return None


def log_admin_activity(admin_id: int, action: str, details: str = None):
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor()
	cursor.execute(
		"INSERT INTO admin_activity_logs (admin_id, action, details) VALUES (%s, %s, %s)",
		(admin_id, action, details),
	)
	conn.commit()
	cursor.close()
	conn.close()


def create_eye_habit(name: str, description: str, category: str, icon: str = None,
					target_frequency: str = 'daily', target_count: int = 1, target_unit: str = 'times',
					instructions: str = None, benefits: str = None, difficulty_level: str = 'easy',
					estimated_time_minutes: int = 5, is_active: int = 1):
	"""Minimal admin-side creator for existing app table `eye_habits`."""
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor()
	cursor.execute(
		"""
		INSERT INTO eye_habits (
			name, description, category, icon, target_frequency, target_count, target_unit,
			instructions, benefits, difficulty_level, estimated_time_minutes, is_active
		) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
		""",
		(
			name, description, category, icon, target_frequency, target_count, target_unit,
			instructions, benefits, difficulty_level, estimated_time_minutes, is_active,
		),
	)
	conn.commit()
	cursor.close()
	conn.close()


def list_eye_habits(limit: int = 100):
	conn = current_app.config['get_db_connection']()
	cursor = conn.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute("SELECT * FROM eye_habits ORDER BY created_at DESC LIMIT %s", (limit,))
	rows = cursor.fetchall()
	cursor.close()
	conn.close()
	return rows

