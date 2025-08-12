CREATE DATABASE IF NOT EXISTS flask_auth_db;
USE flask_auth_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL
);

-- Knowledge base table for chatbot FAQ
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'General',
    keywords TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Chat history table to store user conversations
CREATE TABLE IF NOT EXISTS chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Insert some sample FAQ data
INSERT INTO knowledge_base (question, answer, category, keywords) VALUES
('What is this web app about?', 'This is a comprehensive web application that includes eye disease detection, habit tracking, and various other features to help users maintain their health and wellness.', 'General', 'web app, features, purpose'),
('How do I use the eye disease detection?', 'You can upload an image of an eye through the eye detection feature. The AI model will analyze the image and provide insights about potential eye conditions.', 'Eye Detection', 'eye detection, upload image, AI analysis'),
('How do I track my habits?', 'Use the habits feature to log your daily activities, set goals, and monitor your progress over time. You can create custom habits and track their completion.', 'Habits', 'habit tracking, goals, progress'),
('How do I create an account?', 'Click on the login button and then select "Sign Up" to create a new account. You can use your email or sign up with Google OAuth.', 'Account', 'signup, registration, account creation'),
('How do I reset my password?', 'If you forget your password, you can use the "Forgot Password" link on the login page to reset it via email.', 'Account', 'password reset, forgot password'),
('What file formats are supported for image uploads?', 'The app supports common image formats including JPG, JPEG, PNG, and GIF. Maximum file size is 16MB.', 'Technical', 'file formats, image upload, file size'),
('Is my data secure?', 'Yes, we use industry-standard security practices to protect your data. All passwords are hashed and we follow security best practices.', 'Security', 'data security, privacy, protection'),
('How accurate is the AI detection?', 'Our AI models are trained on extensive datasets and provide high accuracy. However, results should not replace professional medical advice.', 'AI', 'accuracy, AI model, medical advice'),
('Can I use the app without an account?', 'Some features may be available for guest users, but creating an account provides access to all features and saves your data.', 'Account', 'guest access, account benefits'),
('How do I contact support?', 'For technical support or questions, please use the chat feature or contact us through the support section.', 'Support', 'contact, support, help');