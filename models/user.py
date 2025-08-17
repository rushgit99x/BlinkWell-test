from flask import current_app
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import MySQLdb

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

def load_user(user_id):
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user_data:
        user_dict = {
            'id': user_data[0],
            'username': user_data[1],
            'email': user_data[2],
            'password_hash': user_data[3]
        }
        return User(user_dict['id'], user_dict['username'], user_dict['email'])
    return None

def register_user(username, email, password):
    password_hash = generate_password_hash(password)
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                       (username, email, password_hash))
        conn.commit()
        return True
    except MySQLdb.IntegrityError:
        return False
    finally:
        cursor.close()
        conn.close()

def register_google_user(google_id, username, email, profile_pic):
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, email, google_id, profile_pic, is_google_user) VALUES (%s, %s, %s, %s, %s)",
                       (username, email, google_id, profile_pic, 1))
        conn.commit()
        return True
    except MySQLdb.IntegrityError:
        return False
    finally:
        cursor.close()
        conn.close()

def authenticate_user(username, password):
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user_data:
        user_dict = {
            'id': user_data[0],
            'username': user_data[1],
            'email': user_data[2],
            'password_hash': user_data[3]
        }
        if check_password_hash(user_dict['password_hash'], password):
            return User(user_dict['id'], user_dict['username'], user_dict['email'])
    return None

def get_user_by_google_id(google_id):
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE google_id = %s", (google_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

def get_all_users():
    """Get all users for admin purposes"""
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, username, email, created_at FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
        return users
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []
    finally:
        cursor.close()
        conn.close()
