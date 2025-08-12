from flask import Blueprint, request, jsonify, render_template
import json
import torch
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import random
import os
import torch.nn as nn

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

chatbot_bp = Blueprint('chatbot', __name__)

# Neural Network Model (same as in training)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out

class NLTKUtils:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence, words):
        stemmed_sentence = [self.stem(w) for w in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in stemmed_sentence:
                bag[idx] = 1
        return bag

# Global chatbot instance
chatbot_instance = None

class BlinkWellChatbot:
    def __init__(self, model_path='models/blinkwell_chatbot_model.pth', intents_path='models/intents.json'):
        self.nltk_utils = NLTKUtils()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(intents_path):
            raise FileNotFoundError(f"Intents file not found: {intents_path}")
        
        # Load intents
        with open(intents_path, 'r') as f:
            self.intents = json.load(f)
        
        # Load model
        data = torch.load(model_path, map_location=self.device)
        
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        
        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

    def get_response(self, user_input):
        if not user_input.strip():
            return "I'm sorry, I didn't understand that. Could you please ask me something about eye health?"
        
        # Preprocess input
        sentence = self.nltk_utils.tokenize(user_input)
        X = self.nltk_utils.bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        
        # Predict
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        
        tag = self.tags[predicted.item()]
        
        # Get probability
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        # If confidence is too low
        if prob.item() < 0.6:
            return "I'm not sure about that specific question. Could you please rephrase or ask about dry eyes, eye health tips, BlinkWell features, or prevention methods?"
        
        # Find intent and return response
        for intent in self.intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
        
        return "I'm sorry, I couldn't understand your question. Please ask me about eye health, dry eyes, or BlinkWell features."

def initialize_chatbot():
    """Initialize the chatbot instance"""
    global chatbot_instance
    try:
        if chatbot_instance is None:
            chatbot_instance = BlinkWellChatbot()
            print("Chatbot initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return False

@chatbot_bp.route('/chatbot')
def chatbot_page():
    """Render the chatbot page"""
    return render_template('chatbot.html')

@chatbot_bp.route('/api/chatbot/message', methods=['POST'])
def chat_message():
    """Handle chatbot messages"""
    global chatbot_instance
    
    try:
        # Get user message
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        # Initialize chatbot if needed
        if chatbot_instance is None:
            if not initialize_chatbot():
                return jsonify({
                    'success': False,
                    'error': 'Chatbot not available. Please try again later.'
                }), 503
        
        # Get response
        response = chatbot_instance.get_response(user_message)
        
        return jsonify({
            'success': True,
            'response': response,
            'user_message': user_message
        })
        
    except Exception as e:
        print(f"Error in chat_message: {e}")
        return jsonify({
            'success': False,
            'error': 'Sorry, I encountered an error. Please try again.'
        }), 500

@chatbot_bp.route('/api/chatbot/health', methods=['GET'])
def chatbot_health():
    """Check if chatbot is available"""
    global chatbot_instance
    
    if chatbot_instance is None:
        success = initialize_chatbot()
    else:
        success = True
    
    return jsonify({
        'success': success,
        'status': 'ready' if success else 'unavailable'
    })

@chatbot_bp.route('/api/chatbot/reset', methods=['POST'])
def reset_chatbot():
    """Reset chatbot conversation (if needed for session management)"""
    return jsonify({
        'success': True,
        'message': 'Chatbot conversation reset'
    })