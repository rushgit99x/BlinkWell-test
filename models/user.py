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

def save_eye_health_data(user_id, data):
    conn = current_app.config['get_db_connection']()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO user_eye_health_data (
                user_id, gender, age, sleep_duration, sleep_quality, stress_level, blood_pressure,
                heart_rate, daily_steps, physical_activity, height, weight, sleep_disorder,
                wake_up_during_night, feel_sleepy_during_day, caffeine_consumption,
                alcohol_consumption, smoking, medical_issue, ongoing_medication,
                smart_device_before_bed, average_screen_time, blue_light_filter,
                discomfort_eye_strain, redness_in_eye, itchiness_irritation_in_eye,
                dry_eye_disease, eye_image_path, risk_score, risk_factors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            user_id,
            data['gender'],
            data['age'],
            data['sleep_duration'],
            data['sleep_quality'],
            data['stress_level'],
            data['blood_pressure'],
            data['heart_rate'],
            data['daily_steps'],
            data['physical_activity'],
            data['height'],
            data['weight'],
            data['sleep_disorder'],
            data['wake_up_during_night'],
            data['feel_sleepy_during_day'],
            data['caffeine_consumption'],
            data['alcohol_consumption'],
            data['smoking'],
            data['medical_issue'],
            data['ongoing_medication'],
            data['smart_device_before_bed'],
            data['average_screen_time'],
            data['blue_light_filter'],
            data['discomfort_eye_strain'],
            data['redness_in_eye'],
            data['itchiness_irritation_in_eye'],
            data['dry_eye_disease'],
            data['eye_image_path'],
            data['risk_score'],
            data['risk_factors']
        ))
        conn.commit()
        return True
    except MySQLdb.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()