from flask import Blueprint, request, jsonify, render_template, session
from flask_login import login_required, current_user
from models.chatbot import ChatbotAI
from config import Config
import json

chatbot_bp = Blueprint('chatbot', __name__)

# Initialize chatbot AI
chatbot_ai = ChatbotAI({
    'host': Config.MYSQL_HOST,
    'user': Config.MYSQL_USER,
    'password': Config.MYSQL_PASSWORD,
    'database': Config.MYSQL_DB
})

@chatbot_bp.route('/chat')
@login_required
def chat_page():
    """Render the main chat page"""
    return render_template('chatbot/chat.html')

@chatbot_bp.route('/chat/api/send', methods=['POST'])
@login_required
def send_message():
    """API endpoint to send a message and get AI response"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get AI response
        response_data = chatbot_ai.get_response(message)
        
        # Save chat history
        user_id = current_user.id if current_user.is_authenticated else None
        chatbot_ai.save_chat_history(user_id, message, response_data['response'])
        
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'confidence': response_data['confidence'],
            'type': response_data['type'],
            'category': response_data.get('category'),
            'question': response_data.get('question'),
            'suggestions': response_data.get('suggestions', [])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/chat/api/suggestions')
@login_required
def get_suggestions():
    """Get chat suggestions for quick questions"""
    try:
        suggestions = chatbot_ai.get_chat_suggestions()
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/chat/api/refresh')
@login_required
def refresh_knowledge_base():
    """Refresh the knowledge base from database"""
    try:
        chatbot_ai.refresh_knowledge_base()
        return jsonify({
            'success': True,
            'message': 'Knowledge base refreshed successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/chat/api/faq')
@login_required
def get_faq_list():
    """Get list of all FAQ items"""
    try:
        # This would typically come from the database
        # For now, return the sample data
        faqs = [
            {
                'question': 'What is this web app about?',
                'answer': 'This is a comprehensive web application that includes eye disease detection, habit tracking, and various other features to help users maintain their health and wellness.',
                'category': 'General'
            },
            {
                'question': 'How do I use the eye disease detection?',
                'answer': 'You can upload an image of an eye through the eye detection feature. The AI model will analyze the image and provide insights about potential eye conditions.',
                'category': 'Eye Detection'
            },
            {
                'question': 'How do I track my habits?',
                'answer': 'Use the habits feature to log your daily activities, set goals, and monitor your progress over time. You can create custom habits and track their completion.',
                'category': 'Habits'
            },
            {
                'question': 'How do I create an account?',
                'answer': 'Click on the login button and then select "Sign Up" to create a new account. You can use your email or sign up with Google OAuth.',
                'category': 'Account'
            },
            {
                'question': 'What file formats are supported for image uploads?',
                'answer': 'The app supports common image formats including JPG, JPEG, PNG, and GIF. Maximum file size is 16MB.',
                'category': 'Technical'
            }
        ]
        
        return jsonify({
            'success': True,
            'faqs': faqs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/chat/faq')
@login_required
def faq_page():
    """Render the FAQ page"""
    return render_template('chatbot/faq.html')