from flask import Blueprint, current_app, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os

main_bp = Blueprint('main', __name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact')
def contact():
    return render_template('contact.html')

@main_bp.route('/eye-analysis')
@login_required
def eye_analysis():
    return render_template('eye-analysis.html', user=current_user)

@main_bp.route('/recommendations')
@login_required
def recommendations():
    """Display user's personalized recommendations page"""
    return render_template('recommendations.html', user=current_user)

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@main_bp.route('/habits')
@login_required
def habits():
    """Eye habits tracking and practice page"""
    return render_template('habits.html', user=current_user)


@main_bp.route('/my-recommendations')
@login_required
def get_my_recommendations():
    """API endpoint to get user's recommendations"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Get user's latest recommendations grouped by category
        cursor.execute("""
            SELECT ur.*, ued.risk_score 
            FROM user_recommendations ur
            LEFT JOIN user_eye_health_data ued ON ur.user_id = ued.user_id
            WHERE ur.user_id = %s 
            ORDER BY ued.created_at DESC, ur.created_at DESC
        """, (current_user.id,))
        
        recommendations_data = cursor.fetchall()
        
        if not recommendations_data:
            cursor.close()
            conn.close()
            return jsonify({
                'success': True,
                'recommendations': {},
                'stats': {
                    'total_recommendations': 0,
                    'completed_count': 0,
                    'pending_count': 0,
                    'in_progress_count': 0,
                    'risk_score': 0.0
                }
            })

        # Group recommendations by category
        recommendations = {
            'immediate_actions': [],
            'medical_advice': [],
            'lifestyle_changes': [],
            'monitoring': []
        }
        
        # Stats tracking
        total_count = len(recommendations_data)
        completed_count = 0
        pending_count = 0
        in_progress_count = 0
        risk_score = recommendations_data[0].get('risk_score', 0.0) if recommendations_data else 0.0
        
        for rec in recommendations_data:
            rec_dict = {
                'id': rec['id'],
                'text': rec['recommendation_text'],
                'priority': rec['priority'],
                'status': rec['status'],
                'created_at': rec['created_at'].isoformat() if rec['created_at'] else None,
                'completed_at': rec['completed_at'].isoformat() if rec.get('completed_at') else None
            }
            
            # Add to appropriate category
            category = rec['category']
            if category in recommendations:
                recommendations[category].append(rec_dict)
            
            # Update stats
            if rec['status'] == 'completed':
                completed_count += 1
            elif rec['status'] == 'pending':
                pending_count += 1
            elif rec['status'] == 'in_progress':
                in_progress_count += 1
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'stats': {
                'total_recommendations': total_count,
                'completed_count': completed_count,
                'pending_count': pending_count,
                'in_progress_count': in_progress_count,
                'risk_score': float(risk_score) if risk_score else 0.0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main_bp.route('/update-recommendation-status', methods=['POST'])
@login_required
def update_recommendation_status():
    """Update the status of a recommendation"""
    try:
        data = request.get_json()
        recommendation_id = data.get('recommendation_id')
        new_status = data.get('status')
        
        valid_statuses = ['pending', 'in_progress', 'completed', 'dismissed']
        if new_status not in valid_statuses:
            return jsonify({'success': False, 'error': 'Invalid status'})
        
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Update recommendation status
        update_query = """
            UPDATE user_recommendations 
            SET status = %s, 
                completed_at = CASE WHEN %s = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
        """
        
        cursor.execute(update_query, (new_status, new_status, recommendation_id, current_user.id))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Recommendation not found or access denied'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Recommendation status updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main_bp.route('/start-new-analysis', methods=['POST'])
@login_required  
def start_new_analysis():
    """Clear all user data and start fresh analysis"""
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()
        
        # Clear all user's recommendations
        cursor.execute("DELETE FROM user_recommendations WHERE user_id = %s", (current_user.id,))
        
        # Clear all user's eye health data (keep user account)
        cursor.execute("DELETE FROM user_eye_health_data WHERE user_id = %s", (current_user.id,))
        
        # Clear habit tracking data
        cursor.execute("DELETE FROM habit_tracking WHERE user_id = %s", (current_user.id,))
        cursor.execute("DELETE FROM user_habits WHERE user_id = %s", (current_user.id,))
        cursor.execute("DELETE FROM habit_achievements WHERE user_id = %s", (current_user.id,))
        cursor.execute("DELETE FROM habit_summaries WHERE user_id = %s", (current_user.id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'All data cleared successfully. Ready for new analysis.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})