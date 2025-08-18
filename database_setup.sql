-- =====================================================
-- BlinkWell Database Setup Script
-- Complete database creation with admin panel support
-- =====================================================

-- Create database
CREATE DATABASE IF NOT EXISTS `b_test9` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE `b_test9`;

-- =====================================================
-- ADMIN PANEL TABLES
-- =====================================================

-- Admin users table
DROP TABLE IF EXISTS `admin_users`;
CREATE TABLE IF NOT EXISTS `admin_users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(80) NOT NULL,
  `email` varchar(120) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `role` enum('super_admin','admin','moderator') NOT NULL DEFAULT 'admin',
  `is_active` tinyint(1) DEFAULT '1',
  `last_login` timestamp NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Admin activity logs table
DROP TABLE IF EXISTS `admin_activity_logs`;
CREATE TABLE IF NOT EXISTS `admin_activity_logs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `admin_id` int NOT NULL,
  `action` varchar(100) NOT NULL,
  `table_name` varchar(50) DEFAULT NULL,
  `record_id` int DEFAULT NULL,
  `details` json DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `user_agent` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `admin_id` (`admin_id`),
  KEY `action` (`action`),
  KEY `created_at` (`created_at`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Admin permissions table
DROP TABLE IF EXISTS `admin_permissions`;
CREATE TABLE IF NOT EXISTS `admin_permissions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `role` varchar(50) NOT NULL,
  `resource` varchar(100) NOT NULL,
  `action` varchar(50) NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `role_resource_action` (`role`, `resource`, `action`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- =====================================================
-- ORIGINAL APPLICATION TABLES
-- =====================================================

-- Users table
DROP TABLE IF EXISTS `users`;
CREATE TABLE IF NOT EXISTS `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(80) NOT NULL,
  `email` varchar(120) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `google_id` varchar(191) DEFAULT NULL,
  `profile_pic` varchar(255) DEFAULT NULL,
  `is_google_user` tinyint(1) DEFAULT '0',
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`),
  UNIQUE KEY `google_id` (`google_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Eye habits table
DROP TABLE IF EXISTS `eye_habits`;
CREATE TABLE IF NOT EXISTS `eye_habits` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `description` text NOT NULL,
  `category` enum('screen_health','exercise','hydration','sleep','environment','nutrition') NOT NULL,
  `icon` varchar(50) DEFAULT NULL,
  `target_frequency` enum('daily','weekly','custom') DEFAULT 'daily',
  `target_count` int DEFAULT '1',
  `target_unit` varchar(20) DEFAULT 'times',
  `instructions` text,
  `benefits` text,
  `difficulty_level` enum('easy','medium','hard') DEFAULT 'easy',
  `estimated_time_minutes` int DEFAULT '5',
  `is_active` tinyint(1) DEFAULT '1',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- User habits table
DROP TABLE IF EXISTS `user_habits`;
CREATE TABLE IF NOT EXISTS `user_habits` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `habit_id` int NOT NULL,
  `is_active` tinyint(1) DEFAULT '1',
  `custom_target_count` int DEFAULT NULL,
  `custom_target_unit` varchar(20) DEFAULT NULL,
  `reminder_time` time DEFAULT NULL,
  `reminder_enabled` tinyint(1) DEFAULT '1',
  `start_date` date DEFAULT NULL,
  `notes` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `user_habit_unique` (`user_id`,`habit_id`),
  KEY `user_id` (`user_id`),
  KEY `habit_id` (`habit_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Habit tracking table
DROP TABLE IF EXISTS `habit_tracking`;
CREATE TABLE IF NOT EXISTS `habit_tracking` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `user_habit_id` int NOT NULL,
  `habit_id` int NOT NULL,
  `date` date NOT NULL,
  `completed_count` int DEFAULT '0',
  `target_count` int NOT NULL,
  `completion_percentage` decimal(5,2) DEFAULT '0.00',
  `completion_time` time DEFAULT NULL,
  `notes` text,
  `mood_before` int DEFAULT NULL,
  `mood_after` int DEFAULT NULL,
  `difficulty_rating` int DEFAULT NULL,
  `is_completed` tinyint(1) DEFAULT '0',
  `streak_day` int DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `user_habit_date_unique` (`user_id`,`user_habit_id`,`date`),
  KEY `user_id` (`user_id`),
  KEY `user_habit_id` (`user_habit_id`),
  KEY `habit_id` (`habit_id`),
  KEY `date` (`date`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- User eye health data table
DROP TABLE IF EXISTS `user_eye_health_data`;
CREATE TABLE IF NOT EXISTS `user_eye_health_data` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `gender` enum('M','F','Other') NOT NULL,
  `age` int NOT NULL,
  `sleep_duration` decimal(4,1) NOT NULL,
  `sleep_quality` int NOT NULL,
  `stress_level` int NOT NULL,
  `blood_pressure` varchar(10) NOT NULL,
  `heart_rate` int NOT NULL,
  `daily_steps` int NOT NULL,
  `physical_activity` int NOT NULL,
  `height` int NOT NULL,
  `weight` int NOT NULL,
  `sleep_disorder` enum('Y','N') NOT NULL,
  `wake_up_during_night` enum('Y','N') NOT NULL,
  `feel_sleepy_during_day` enum('Y','N') NOT NULL,
  `caffeine_consumption` enum('Y','N') NOT NULL,
  `alcohol_consumption` enum('Y','N') NOT NULL,
  `smoking` enum('Y','N') NOT NULL,
  `medical_issue` enum('Y','N') NOT NULL,
  `ongoing_medication` enum('Y','N') NOT NULL,
  `smart_device_before_bed` enum('Y','N') NOT NULL,
  `average_screen_time` decimal(4,1) NOT NULL,
  `blue_light_filter` enum('Y','N') NOT NULL,
  `discomfort_eye_strain` enum('Y','N') NOT NULL,
  `redness_in_eye` enum('Y','N') NOT NULL,
  `itchiness_irritation_in_eye` enum('Y','N') NOT NULL,
  `dry_eye_disease` enum('Y','N') NOT NULL,
  `eye_image_path` varchar(255) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `risk_score` decimal(5,1) DEFAULT '0.0',
  `risk_factors` json DEFAULT NULL,
  `recommendations_saved` tinyint(1) DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Habit achievements table
DROP TABLE IF EXISTS `habit_achievements`;
CREATE TABLE IF NOT EXISTS `habit_achievements` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `habit_id` int DEFAULT NULL,
  `achievement_type` enum('streak','consistency','improvement','milestone') NOT NULL,
  `achievement_name` varchar(100) NOT NULL,
  `achievement_description` text,
  `badge_icon` varchar(50) DEFAULT NULL,
  `value` int NOT NULL,
  `earned_date` date NOT NULL,
  `is_claimed` tinyint(1) DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `habit_id` (`habit_id`),
  KEY `earned_date` (`earned_date`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Habit summaries table
DROP TABLE IF EXISTS `habit_summaries`;
CREATE TABLE IF NOT EXISTS `habit_summaries` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `habit_id` int NOT NULL,
  `period_type` enum('week','month') NOT NULL,
  `period_start` date NOT NULL,
  `period_end` date NOT NULL,
  `total_target_count` int NOT NULL,
  `total_completed_count` int NOT NULL,
  `completion_percentage` decimal(5,2) NOT NULL,
  `streak_days` int DEFAULT '0',
  `average_mood_improvement` decimal(3,2) DEFAULT NULL,
  `average_difficulty` decimal(3,2) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `habit_id` (`habit_id`),
  KEY `period_start` (`period_start`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- User notification preferences table
DROP TABLE IF EXISTS `user_notification_preferences`;
CREATE TABLE IF NOT EXISTS `user_notification_preferences` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `eye_exercise_reminders` tinyint(1) DEFAULT '1',
  `daily_habit_tracking` tinyint(1) DEFAULT '1',
  `weekly_progress_reports` tinyint(1) DEFAULT '0',
  `risk_assessment_updates` tinyint(1) DEFAULT '1',
  `email_frequency` enum('daily','weekly','monthly','never') DEFAULT 'weekly',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_user_preferences` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- User privacy settings table
DROP TABLE IF EXISTS `user_privacy_settings`;
CREATE TABLE IF NOT EXISTS `user_privacy_settings` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `share_data_research` tinyint(1) DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_user_privacy` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- User recommendations table
DROP TABLE IF EXISTS `user_recommendations`;
CREATE TABLE IF NOT EXISTS `user_recommendations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `analysis_id` int DEFAULT NULL,
  `category` enum('immediate_actions','lifestyle_changes','medical_advice','monitoring') NOT NULL,
  `recommendation_text` text NOT NULL,
  `priority` enum('high','medium','low') DEFAULT 'medium',
  `status` enum('pending','in_progress','completed','dismissed') DEFAULT 'pending',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `completed_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `analysis_id` (`analysis_id`),
  KEY `status` (`status`),
  KEY `category` (`category`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- =====================================================
-- INSERT DEFAULT DATA
-- =====================================================

-- Insert default admin user (password: admin123)
INSERT INTO `admin_users` (`username`, `email`, `password_hash`, `role`) VALUES
('admin', 'admin@blinkwell.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iQmO', 'super_admin');

-- Insert default permissions
INSERT INTO `admin_permissions` (`role`, `resource`, `action`) VALUES
-- Super Admin permissions
('super_admin', 'users', 'read'),
('super_admin', 'users', 'write'),
('super_admin', 'users', 'delete'),
('super_admin', 'eye_habits', 'read'),
('super_admin', 'eye_habits', 'write'),
('super_admin', 'eye_habits', 'delete'),
('super_admin', 'habit_tracking', 'read'),
('super_admin', 'habit_tracking', 'write'),
('super_admin', 'habit_tracking', 'delete'),
('super_admin', 'user_eye_health_data', 'read'),
('super_admin', 'user_eye_health_data', 'write'),
('super_admin', 'user_eye_health_data', 'delete'),
('super_admin', 'admin_users', 'read'),
('super_admin', 'admin_users', 'write'),
('super_admin', 'admin_users', 'delete'),

-- Admin permissions
('admin', 'users', 'read'),
('admin', 'eye_habits', 'read'),
('admin', 'eye_habits', 'write'),
('admin', 'habit_tracking', 'read'),
('admin', 'user_eye_health_data', 'read'),

-- Moderator permissions
('moderator', 'users', 'read'),
('moderator', 'eye_habits', 'read'),
('moderator', 'habit_tracking', 'read');

-- Insert sample eye habits
INSERT INTO `eye_habits` (`name`, `description`, `category`, `icon`, `target_frequency`, `target_count`, `target_unit`, `instructions`, `benefits`, `difficulty_level`, `estimated_time_minutes`) VALUES
('20-20-20 Rule', 'Every 20 minutes, look at something 20 feet away for 20 seconds to reduce eye strain.', 'screen_health', 'fas fa-eye', 'daily', 8, 'times', 'Set a timer for every 20 minutes while working on screens. Look away at a distant object for 20 seconds.', 'Reduces eye strain, prevents digital eye fatigue, improves focus', 'easy', 1),
('Eye Exercises', 'Simple eye exercises to strengthen eye muscles and improve vision.', 'exercise', 'fas fa-dumbbell', 'daily', 3, 'sets', '1. Blink rapidly for 30 seconds\n2. Roll your eyes in circles\n3. Focus on near and far objects', 'Strengthens eye muscles, improves focus, reduces eye strain', 'easy', 5),
('Hydration Reminder', 'Stay hydrated to maintain healthy eyes and reduce dryness.', 'hydration', 'fas fa-tint', 'daily', 8, 'glasses', 'Drink a glass of water every 2 hours. Aim for 8 glasses per day.', 'Prevents dry eyes, maintains tear production, overall health', 'easy', 1),
('Blue Light Filter', 'Use blue light filters on devices to reduce eye strain.', 'screen_health', 'fas fa-shield-alt', 'daily', 1, 'time', 'Enable blue light filter on all devices, especially in the evening.', 'Reduces eye strain, improves sleep quality, protects retina', 'easy', 2),
('Proper Lighting', 'Ensure proper lighting to reduce eye strain while reading or working.', 'environment', 'fas fa-lightbulb', 'daily', 1, 'check', 'Check that your workspace has adequate, non-glaring lighting.', 'Reduces eye strain, prevents headaches, improves productivity', 'easy', 1),
('Screen Distance', 'Maintain proper distance from screens to reduce eye strain.', 'screen_health', 'fas fa-ruler', 'daily', 1, 'check', 'Keep screens at arm\'s length (20-28 inches) and slightly below eye level.', 'Prevents eye strain, maintains good posture, reduces fatigue', 'easy', 1),
('Regular Breaks', 'Take regular breaks from screen work to rest your eyes.', 'screen_health', 'fas fa-clock', 'daily', 6, 'breaks', 'Take a 5-minute break every hour. Stand up and move around.', 'Reduces eye strain, improves circulation, prevents fatigue', 'easy', 5),
('Sleep Hygiene', 'Maintain good sleep habits for healthy eyes.', 'sleep', 'fas fa-bed', 'daily', 1, 'night', 'Get 7-9 hours of quality sleep. Avoid screens 1 hour before bed.', 'Allows eyes to rest, reduces eye strain, improves vision', 'medium', 1),
('Eye Massage', 'Gentle eye massage to relieve tension and improve circulation.', 'exercise', 'fas fa-hands', 'daily', 2, 'sessions', 'Gently massage around your eyes with clean hands for 2-3 minutes.', 'Relieves eye tension, improves circulation, reduces puffiness', 'easy', 3),
('Nutrition for Eyes', 'Eat foods rich in vitamins and nutrients that support eye health.', 'nutrition', 'fas fa-apple-alt', 'daily', 3, 'meals', 'Include foods rich in Vitamin A, C, E, and Omega-3 fatty acids.', 'Supports eye health, prevents age-related issues, improves vision', 'medium', 1);

-- =====================================================
-- CREATE INDEXES FOR BETTER PERFORMANCE
-- =====================================================

-- Add additional indexes for better query performance
CREATE INDEX `idx_users_created_at` ON `users` (`created_at`);
CREATE INDEX `idx_users_email` ON `users` (`email`);
CREATE INDEX `idx_eye_habits_category` ON `eye_habits` (`category`);
CREATE INDEX `idx_eye_habits_active` ON `eye_habits` (`is_active`);
CREATE INDEX `idx_habit_tracking_user_date` ON `habit_tracking` (`user_id`, `date`);
CREATE INDEX `idx_habit_tracking_completed` ON `habit_tracking` (`is_completed`);
CREATE INDEX `idx_user_eye_health_data_user` ON `user_eye_health_data` (`user_id`);
CREATE INDEX `idx_habit_achievements_user` ON `habit_achievements` (`user_id`);
CREATE INDEX `idx_habit_summaries_user_period` ON `habit_summaries` (`user_id`, `period_start`);

-- =====================================================
-- CREATE VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for user habit summary
CREATE OR REPLACE VIEW `user_habit_summary` AS
SELECT 
    u.id as user_id,
    u.username,
    u.email,
    COUNT(DISTINCT uh.id) as total_habits,
    COUNT(DISTINCT ht.id) as total_tracking_records,
    MAX(ht.date) as last_activity,
    u.created_at as registration_date
FROM users u
LEFT JOIN user_habits uh ON u.id = uh.user_id
LEFT JOIN habit_tracking ht ON u.id = ht.user_id
GROUP BY u.id, u.username, u.email, u.created_at;

-- View for habit performance
CREATE OR REPLACE VIEW `habit_performance` AS
SELECT 
    h.id,
    h.name,
    h.category,
    h.difficulty_level,
    COUNT(DISTINCT uh.user_id) as total_users,
    COUNT(ht.id) as total_tracking_records,
    SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_records,
    ROUND(AVG(CASE WHEN ht.is_completed = 1 THEN ht.completion_percentage ELSE 0 END), 2) as avg_completion_rate,
    ROUND((SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) / COUNT(ht.id)) * 100, 2) as overall_completion_rate
FROM eye_habits h
LEFT JOIN user_habits uh ON h.id = uh.habit_id
LEFT JOIN habit_tracking ht ON h.id = ht.habit_id
GROUP BY h.id, h.name, h.category, h.difficulty_level;

-- View for daily activity summary
CREATE OR REPLACE VIEW `daily_activity_summary` AS
SELECT 
    DATE(ht.date) as activity_date,
    COUNT(DISTINCT ht.user_id) as active_users,
    COUNT(*) as total_activities,
    SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_activities,
    ROUND((SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) as completion_rate
FROM habit_tracking ht
GROUP BY DATE(ht.date)
ORDER BY activity_date DESC;

-- =====================================================
-- CREATE STORED PROCEDURES
-- =====================================================

DELIMITER //

-- Procedure to get user statistics
CREATE PROCEDURE GetUserStats(IN user_id INT)
BEGIN
    SELECT 
        u.username,
        u.email,
        u.created_at,
        COUNT(DISTINCT uh.id) as total_habits,
        COUNT(ht.id) as total_tracking_records,
        SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_habits,
        MAX(ht.date) as last_activity,
        ROUND(AVG(CASE WHEN ht.is_completed = 1 THEN ht.completion_percentage ELSE 0 END), 2) as avg_completion_rate
    FROM users u
    LEFT JOIN user_habits uh ON u.id = uh.user_id
    LEFT JOIN habit_tracking ht ON u.id = ht.user_id
    WHERE u.id = user_id
    GROUP BY u.id, u.username, u.email, u.created_at;
END //

-- Procedure to get habit statistics
CREATE PROCEDURE GetHabitStats(IN habit_id INT)
BEGIN
    SELECT 
        h.name,
        h.category,
        h.difficulty_level,
        COUNT(DISTINCT uh.user_id) as total_users,
        COUNT(ht.id) as total_attempts,
        SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) as completed_attempts,
        ROUND((SUM(CASE WHEN ht.is_completed = 1 THEN 1 ELSE 0 END) / COUNT(ht.id)) * 100, 2) as completion_rate
    FROM eye_habits h
    LEFT JOIN user_habits uh ON h.id = uh.habit_id
    LEFT JOIN habit_tracking ht ON h.id = ht.habit_id
    WHERE h.id = habit_id
    GROUP BY h.id, h.name, h.category, h.difficulty_level;
END //

-- Procedure to clean up old activity logs
CREATE PROCEDURE CleanupOldLogs(IN days_to_keep INT)
BEGIN
    DELETE FROM admin_activity_logs 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL days_to_keep DAY);
    
    SELECT ROW_COUNT() as deleted_records;
END //

DELIMITER ;

-- =====================================================
-- CREATE TRIGGERS
-- =====================================================

DELIMITER //

-- Trigger to update user_habits updated_at timestamp
CREATE TRIGGER update_user_habits_timestamp 
BEFORE UPDATE ON user_habits
FOR EACH ROW
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END //

-- Trigger to update habit_tracking updated_at timestamp
CREATE TRIGGER update_habit_tracking_timestamp 
BEFORE UPDATE ON habit_tracking
FOR EACH ROW
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END //

-- Trigger to update user_eye_health_data updated_at timestamp
CREATE TRIGGER update_user_eye_health_data_timestamp 
BEFORE UPDATE ON user_eye_health_data
FOR EACH ROW
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END //

-- Trigger to log admin user updates
CREATE TRIGGER log_admin_user_update 
AFTER UPDATE ON admin_users
FOR EACH ROW
BEGIN
    IF OLD.username != NEW.username OR OLD.email != NEW.email OR OLD.role != NEW.role OR OLD.is_active != NEW.is_active THEN
        INSERT INTO admin_activity_logs (admin_id, action, table_name, record_id, details)
        VALUES (NEW.id, 'update_admin_user', 'admin_users', NEW.id, 
                JSON_OBJECT('old_username', OLD.username, 'new_username', NEW.username, 
                           'old_email', OLD.email, 'new_email', NEW.email,
                           'old_role', OLD.role, 'new_role', NEW.role,
                           'old_active', OLD.is_active, 'new_active', NEW.is_active));
    END IF;
END //

DELIMITER ;

-- =====================================================
-- FINAL SETUP COMPLETION
-- =====================================================

-- Show completion message
SELECT 'BlinkWell Database Setup Completed Successfully!' as status;

-- Show table count
SELECT 
    'Total Tables Created' as description,
    COUNT(*) as count
FROM information_schema.tables 
WHERE table_schema = 'b_test9';

-- Show admin user created
SELECT 
    'Default Admin User' as description,
    username,
    email,
    role
FROM admin_users 
WHERE username = 'admin';

-- Show permissions count
SELECT 
    'Total Permissions' as description,
    COUNT(*) as count
FROM admin_permissions;