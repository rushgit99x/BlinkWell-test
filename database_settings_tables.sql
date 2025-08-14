-- Database tables for BlinkWell Settings functionality

-- User notification preferences table
CREATE TABLE IF NOT EXISTS user_notification_preferences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    eye_exercise_reminders BOOLEAN DEFAULT TRUE,
    daily_habit_tracking BOOLEAN DEFAULT TRUE,
    weekly_progress_reports BOOLEAN DEFAULT FALSE,
    risk_assessment_updates BOOLEAN DEFAULT TRUE,
    email_frequency ENUM('daily', 'weekly', 'monthly', 'never') DEFAULT 'weekly',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_preferences (user_id)
);

-- User privacy settings table
CREATE TABLE IF NOT EXISTS user_privacy_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    share_data_research BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_privacy (user_id)
);

-- Add missing columns to users table if they don't exist
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;

-- Insert default notification preferences for existing users
INSERT IGNORE INTO user_notification_preferences (user_id, eye_exercise_reminders, daily_habit_tracking, weekly_progress_reports, risk_assessment_updates, email_frequency)
SELECT id, TRUE, TRUE, FALSE, TRUE, 'weekly' FROM users;

-- Insert default privacy settings for existing users
INSERT IGNORE INTO user_privacy_settings (user_id, share_data_research)
SELECT id, FALSE FROM users;