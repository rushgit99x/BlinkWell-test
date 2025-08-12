import re
import sqlite3
import mysql.connector
from typing import List, Tuple, Optional
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class ChatbotAI:
    def __init__(self, db_config):
        self.db_config = db_config
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.knowledge_base = []
        self.vectors = None
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load knowledge base from database and prepare vectors"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT question, answer, category, keywords FROM knowledge_base WHERE is_active = TRUE")
            results = cursor.fetchall()
            
            self.knowledge_base = []
            questions = []
            
            for row in results:
                question, answer, category, keywords = row
                # Combine question, answer, category, and keywords for better matching
                combined_text = f"{question} {answer} {category} {keywords or ''}"
                self.knowledge_base.append({
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'keywords': keywords,
                    'combined_text': combined_text
                })
                questions.append(combined_text)
            
            if questions:
                # Create TF-IDF vectors for similarity matching
                self.vectors = self.vectorizer.fit_transform(questions)
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.knowledge_base = []
            self.vectors = None
    
    def find_best_match(self, user_input: str, threshold: float = 0.3) -> Optional[dict]:
        """Find the best matching FAQ using TF-IDF and cosine similarity"""
        if not self.knowledge_base or self.vectors is None:
            return None
        
        # Transform user input
        user_vector = self.vectorizer.transform([user_input.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.vectors).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return {
                **self.knowledge_base[best_idx],
                'confidence': float(best_score)
            }
        
        return None
    
    def get_response(self, user_input: str) -> dict:
        """Get AI response based on user input"""
        user_input = user_input.strip()
        
        if not user_input:
            return {
                'response': 'Please provide a question or message.',
                'confidence': 0.0,
                'type': 'error'
            }
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in user_input.lower() for greeting in greetings):
            return {
                'response': 'Hello! I\'m your AI assistant. How can I help you today? You can ask me questions about the app, features, or any other topics.',
                'confidence': 1.0,
                'type': 'greeting'
            }
        
        # Check for thanks
        thanks = ['thank', 'thanks', 'appreciate']
        if any(thank in user_input.lower() for thank in thanks):
            return {
                'response': 'You\'re welcome! Is there anything else I can help you with?',
                'confidence': 1.0,
                'type': 'thanks'
            }
        
        # Check for goodbye
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell']
        if any(goodbye in user_input.lower() for goodbye in goodbyes):
            return {
                'response': 'Goodbye! Feel free to come back if you have more questions.',
                'confidence': 1.0,
                'type': 'goodbye'
            }
        
        # Try to find best match from knowledge base
        best_match = self.find_best_match(user_input)
        
        if best_match:
            return {
                'response': best_match['answer'],
                'confidence': best_match['confidence'],
                'type': 'faq',
                'category': best_match['category'],
                'question': best_match['question']
            }
        
        # If no good match found, provide helpful response
        return {
            'response': 'I\'m not sure I understand your question. Could you please rephrase it or ask something else? You can ask me about:\n• How to use the app features\n• Account management\n• Technical support\n• General questions about the application',
            'confidence': 0.0,
            'type': 'no_match',
            'suggestions': [
                'How do I use the eye disease detection?',
                'How do I track my habits?',
                'How do I create an account?',
                'What file formats are supported?'
            ]
        }
    
    def save_chat_history(self, user_id: Optional[int], message: str, response: str):
        """Save chat conversation to database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO chat_history (user_id, message, response) VALUES (%s, %s, %s)",
                (user_id, message, response)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def get_chat_suggestions(self) -> List[str]:
        """Get popular questions as chat suggestions"""
        suggestions = []
        for item in self.knowledge_base[:5]:  # Top 5 questions
            suggestions.append(item['question'])
        return suggestions
    
    def refresh_knowledge_base(self):
        """Refresh knowledge base from database"""
        self.load_knowledge_base()