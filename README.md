# BlinkWell - Eye Health Application

A comprehensive eye health application with AI-powered disease detection, habit tracking, and automated email notifications.

## Features

- **Eye Disease Detection**: AI-powered analysis of eye images
- **Habit Tracking**: Monitor and track eye health habits
- **Email Notifications**: Automated email system for reminders and reports
- **User Authentication**: Secure login with Google OAuth support
- **Progress Monitoring**: Track your eye health journey

## Email Notification System

The application includes a comprehensive email notification system that sends:

### 1. Account Creation Emails
- **Welcome emails** sent automatically when users register
- Beautiful HTML templates with eye health tips
- Available for both regular and Google OAuth registrations

### 2. Eye Habit Reminder Emails
- **Daily reminders** at 9:00 AM, 2:00 PM, and 6:00 PM
- Personalized based on user's selected habits
- Includes current streak information and motivation tips
- Customizable reminder times per habit

### 3. Recommendation Emails
- **Monthly personalized recommendations** based on user behavior
- Generated from habit completion patterns
- Includes actionable tips for improving eye health
- Sent on the first day of each month at 9:00 AM

### 4. Progress Report Emails
- **Weekly progress summaries** sent every Sunday at 10:00 AM
- Habit completion statistics and streak information
- Goal-setting suggestions for the upcoming week
- Motivation and encouragement messages

## Setup Instructions

### 1. Environment Variables

Create a `.env` file in your project root with the following email configuration:

```bash
# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SENDER_EMAIL=your-email@gmail.com
SENDER_NAME=BlinkWell

# Base URL for email links
BASE_URL=http://localhost:5000

# Other existing variables
SECRET_KEY=your-secret-key
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DB=blinkwell
```

### 2. Gmail App Password Setup

For Gmail, you'll need to create an App Password:

1. Go to your Google Account settings
2. Enable 2-Factor Authentication if not already enabled
3. Go to Security → App passwords
4. Generate a new app password for "Mail"
5. Use this password in your `SMTP_PASSWORD` environment variable

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Database Setup

Ensure your database has the necessary tables for user habits and tracking:

```sql
-- Users table (should already exist)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    google_id VARCHAR(255),
    profile_pic TEXT,
    is_google_user BOOLEAN DEFAULT FALSE
);

-- Eye habits table
CREATE TABLE IF NOT EXISTS eye_habits (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    icon VARCHAR(100),
    target_count INT DEFAULT 1,
    target_unit VARCHAR(50),
    instructions TEXT,
    benefits TEXT,
    difficulty_level VARCHAR(50),
    estimated_time_minutes INT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User habits table
CREATE TABLE IF NOT EXISTS user_habits (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    habit_id INT NOT NULL,
    custom_target_count INT,
    custom_target_unit VARCHAR(50),
    reminder_time TIME,
    reminder_enabled BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (habit_id) REFERENCES eye_habits(id)
);

-- Habit tracking table
CREATE TABLE IF NOT EXISTS habit_tracking (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_habit_id INT NOT NULL,
    date DATE NOT NULL,
    completed_count INT DEFAULT 0,
    completion_percentage DECIMAL(5,2),
    is_completed BOOLEAN DEFAULT FALSE,
    mood_before VARCHAR(50),
    mood_after VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_habit_id) REFERENCES user_habits(id)
);
```

## Usage

### 1. Start the Application

```bash
python app.py
```

The notification scheduler will start automatically and begin sending scheduled emails.

### 2. Test the Email System

Visit `/email-test` in your browser (requires login) to test all email types:

- Send test welcome emails
- Test habit reminders
- Send recommendation emails
- Test weekly progress reports
- Control the scheduler
- Manage notification preferences

### 3. API Endpoints

The system provides several API endpoints for managing notifications:

#### Test Emails
- `POST /api/notifications/test-email` - Send test emails of various types

#### Manual Notifications
- `POST /api/notifications/send-reminder` - Send immediate habit reminder
- `POST /api/notifications/send-recommendations` - Send personalized recommendations
- `POST /api/notifications/send-weekly-report` - Send weekly progress report

#### Scheduler Control
- `GET /api/notifications/status` - Get scheduler and email service status
- `POST /api/notifications/start-scheduler` - Start the notification scheduler
- `POST /api/notifications/stop-scheduler` - Stop the notification scheduler

#### User Preferences
- `GET /api/notifications/preferences` - Get user's notification preferences
- `PUT /api/notifications/preferences` - Update notification preferences

## Email Templates

The system includes professionally designed HTML email templates:

- **Welcome Email**: Introduction to BlinkWell features
- **Habit Reminders**: Motivational reminders with streak information
- **Recommendations**: Personalized tips based on user behavior
- **Weekly Reports**: Progress summaries with goal suggestions

All emails include:
- Responsive design for mobile and desktop
- Professional branding and styling
- Clear call-to-action buttons
- Fallback text versions for email clients that don't support HTML

## Customization

### 1. Email Templates

Modify email templates in `services/email_service.py`:
- Update HTML content and styling
- Change email subjects and content
- Add new email types

### 2. Scheduling

Adjust notification timing in `services/notification_scheduler.py`:
- Change reminder times
- Modify weekly/monthly schedules
- Add new notification types

### 3. Content

Personalize email content based on:
- User's habit completion patterns
- Streak information
- Progress statistics
- User preferences

## Troubleshooting

### Common Issues

1. **Emails not sending**
   - Check SMTP credentials in `.env` file
   - Verify Gmail app password is correct
   - Check firewall/network settings

2. **Scheduler not running**
   - Check application logs for errors
   - Verify database connection
   - Ensure all required tables exist

3. **Template rendering issues**
   - Check HTML syntax in email templates
   - Verify template variables are properly defined
   - Test with simple content first

### Logging

The system includes comprehensive logging:
- Email sending attempts and results
- Scheduler status and operations
- User interaction tracking
- Error details for debugging

Check application logs for detailed information about system operation.

## Security Considerations

- **SMTP Credentials**: Never commit email credentials to version control
- **User Authentication**: All notification endpoints require user login
- **Rate Limiting**: Consider implementing rate limiting for email sending
- **Data Privacy**: Ensure compliance with email privacy regulations

## Performance

- **Background Processing**: Email sending runs in background threads
- **Database Optimization**: Efficient queries for user data and statistics
- **Template Caching**: Email templates are generated dynamically
- **Error Handling**: Graceful degradation when email service is unavailable

## Future Enhancements

Potential improvements to consider:
- **Email Analytics**: Track open rates and click-through rates
- **A/B Testing**: Test different email content and timing
- **Advanced Scheduling**: User-defined notification schedules
- **Email Preferences**: Granular control over email types and frequency
- **Integration**: Connect with external notification services

## Support

For issues or questions about the email notification system:
1. Check the application logs for error details
2. Verify environment variable configuration
3. Test individual components using the test page
4. Review database schema and data integrity

---

**Note**: This email notification system is designed to enhance user engagement and support healthy eye habits. Ensure compliance with email marketing regulations in your jurisdiction.