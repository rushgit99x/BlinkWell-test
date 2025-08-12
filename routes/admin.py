from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_login import login_required, current_user
from models.database import get_db_connection
import mysql.connector

admin_bp = Blueprint('admin', __name__)

def is_admin():
    """Check if current user is admin (you can implement your own admin logic)"""
    # For now, allow any authenticated user to access admin
    # In production, implement proper admin role checking
    return current_user.is_authenticated

@admin_bp.route('/admin/knowledge-base')
@login_required
def knowledge_base_management():
    """Admin page to manage knowledge base"""
    if not is_admin():
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.dashboard'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get all knowledge base items
        cursor.execute("""
            SELECT id, question, answer, category, keywords, is_active, created_at, updated_at 
            FROM knowledge_base 
            ORDER BY category, created_at DESC
        """)
        knowledge_items = cursor.fetchall()
        
        # Get unique categories
        cursor.execute("SELECT DISTINCT category FROM knowledge_base ORDER BY category")
        categories = [row['category'] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return render_template('admin/knowledge_base.html', 
                             knowledge_items=knowledge_items, 
                             categories=categories)
    
    except Exception as e:
        flash(f'Error loading knowledge base: {str(e)}', 'error')
        return redirect(url_for('main.dashboard'))

@admin_bp.route('/admin/knowledge-base/add', methods=['POST'])
@login_required
def add_knowledge_item():
    """Add new knowledge base item"""
    if not is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        category = data.get('category', 'General').strip()
        keywords = data.get('keywords', '').strip()
        
        if not question or not answer:
            return jsonify({'error': 'Question and answer are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO knowledge_base (question, answer, category, keywords) 
            VALUES (%s, %s, %s, %s)
        """, (question, answer, category, keywords))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Knowledge item added successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/admin/knowledge-base/<int:item_id>', methods=['PUT'])
@login_required
def update_knowledge_item(item_id):
    """Update existing knowledge base item"""
    if not is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        category = data.get('category', 'General').strip()
        keywords = data.get('keywords', '').strip()
        is_active = data.get('is_active', True)
        
        if not question or not answer:
            return jsonify({'error': 'Question and answer are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE knowledge_base 
            SET question = %s, answer = %s, category = %s, keywords = %s, is_active = %s
            WHERE id = %s
        """, (question, answer, category, keywords, is_active, item_id))
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'Knowledge item not found'}), 404
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Knowledge item updated successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/admin/knowledge-base/<int:item_id>', methods=['DELETE'])
@login_required
def delete_knowledge_item(item_id):
    """Delete knowledge base item"""
    if not is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM knowledge_base WHERE id = %s", (item_id,))
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'Knowledge item not found'}), 404
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Knowledge item deleted successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/admin/knowledge-base/import', methods=['POST'])
@login_required
def import_knowledge_base():
    """Import knowledge base from JSON file"""
    if not is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Only JSON files are supported'}), 400
        
        # Read and parse JSON file
        import json
        data = json.load(file)
        
        if not isinstance(data, list):
            return jsonify({'error': 'Invalid JSON format. Expected array of objects.'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        imported_count = 0
        for item in data:
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            category = item.get('category', 'General').strip()
            keywords = item.get('keywords', '').strip()
            
            if question and answer:
                cursor.execute("""
                    INSERT INTO knowledge_base (question, answer, category, keywords) 
                    VALUES (%s, %s, %s, %s)
                """, (question, answer, category, keywords))
                imported_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Successfully imported {imported_count} knowledge items'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/admin/knowledge-base/export')
@login_required
def export_knowledge_base():
    """Export knowledge base to JSON file"""
    if not is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT question, answer, category, keywords 
            FROM knowledge_base 
            WHERE is_active = TRUE 
            ORDER BY category, created_at
        """)
        knowledge_items = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        from flask import send_file
        import json
        import tempfile
        import os
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(knowledge_items, f, indent=2)
            temp_path = f.name
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name='knowledge_base.json',
            mimetype='application/json'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500