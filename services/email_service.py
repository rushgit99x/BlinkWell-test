import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import os
from flask import current_app, render_template_string
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the email service with Flask app configuration"""
        self.app = app
        self.smtp_server = app.config.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = app.config.get('SMTP_PORT', 587)
        self.smtp_username = app.config.get('SMTP_USERNAME')
        self.smtp_password = app.config.get('SMTP_PASSWORD')
        self.sender_email = app.config.get('SENDER_EMAIL')
        self.sender_name = app.config.get('SENDER_NAME', 'BlinkWell')
        
        # Validate required configuration
        if not all([self.smtp_username, self.smtp_password, self.sender_email]):
            logger.warning("Email service not fully configured. Some email features may not work.")
    
    def send_email(self, to_email, subject, html_content, text_content=None, attachments=None):
        """Send an email with HTML and optional text content and attachments"""
        if not self.smtp_username or not self.smtp_password:
            logger.error("SMTP credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.sender_name} <{self.sender_email}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    with open(attachment['path'], 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    def send_welcome_email(self, user_email, username):
        """Send welcome email for new account creation"""
        subject = "Welcome to BlinkWell - Your Eye Health Journey Begins!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to BlinkWell</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>👁️ Welcome to BlinkWell!</h1>
                    <p>Your journey to better eye health starts now</p>
                </div>
                <div class="content">
                    <h2>Hello {username}!</h2>
                    <p>Thank you for joining BlinkWell, your comprehensive eye health companion. We're excited to help you develop healthy eye habits and maintain optimal vision.</p>
                    
                    <h3>🚀 What you can do with BlinkWell:</h3>
                    <ul>
                        <li><strong>Eye Disease Detection:</strong> Upload images for AI-powered analysis</li>
                        <li><strong>Habit Tracking:</strong> Build healthy eye care routines</li>
                        <li><strong>Personalized Recommendations:</strong> Get tips based on your habits</li>
                        <li><strong>Progress Monitoring:</strong> Track your eye health journey</li>
                    </ul>
                    
                    <p>Ready to get started? Click the button below to access your dashboard:</p>
                    
                    <a href="{current_app.config.get('BASE_URL', 'http://localhost:5000')}/dashboard" class="button">Go to Dashboard</a>
                    
                    <h3>💡 Quick Tips to Get Started:</h3>
                    <ol>
                        <li>Complete your profile and preferences</li>
                        <li>Choose eye habits that fit your lifestyle</li>
                        <li>Set up reminders for your selected habits</li>
                        <li>Try our eye disease detection feature</li>
                    </ol>
                    
                    <p>If you have any questions or need assistance, don't hesitate to reach out to our support team.</p>
                </div>
                <div class="footer">
                    <p>© 2024 BlinkWell. All rights reserved.</p>
                    <p>This email was sent to {user_email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to BlinkWell!
        
        Hello {username}!
        
        Thank you for joining BlinkWell, your comprehensive eye health companion. We're excited to help you develop healthy eye habits and maintain optimal vision.
        
        What you can do with BlinkWell:
        - Eye Disease Detection: Upload images for AI-powered analysis
        - Habit Tracking: Build healthy eye care routines
        - Personalized Recommendations: Get tips based on your habits
        - Progress Monitoring: Track your eye health journey
        
        Ready to get started? Visit your dashboard at: {current_app.config.get('BASE_URL', 'http://localhost:5000')}/dashboard
        
        Quick Tips to Get Started:
        1. Complete your profile and preferences
        2. Choose eye habits that fit your lifestyle
        3. Set up reminders for your selected habits
        4. Try our eye disease detection feature
        
        If you have any questions or need assistance, don't hesitate to reach out to our support team.
        
        © 2024 BlinkWell. All rights reserved.
        """
        
        return self.send_email(user_email, subject, html_content, text_content)
    
    def send_habit_reminder_email(self, user_email, username, habit_name, reminder_time, streak_days=0):
        """Send eye habit reminder email"""
        subject = f"👁️ Time for your {habit_name} habit!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Habit Reminder - {habit_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .streak {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; text-align: center; }}
                .button {{ display: inline-block; background: #ff6b6b; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>👁️ Time for {habit_name}!</h1>
                    <p>Your eye health reminder at {reminder_time}</p>
                </div>
                <div class="content">
                    <h2>Hello {username}!</h2>
                    <p>It's time to take care of your eyes! Don't forget to complete your <strong>{habit_name}</strong> habit.</p>
                    
                    {f'<div class="streak"><h3>🔥 Current Streak: {streak_days} days!</h3><p>Keep up the great work!</p></div>' if streak_days > 0 else ''}
                    
                    <h3>💪 Why this habit matters:</h3>
                    <p>Regular eye care habits help maintain good vision and prevent eye strain. Every small step counts towards better eye health!</p>
                    
                    <p>Click the button below to mark your habit as complete:</p>
                    
                    <a href="{current_app.config.get('BASE_URL', 'http://localhost:5000')}/habits" class="button">Complete Habit</a>
                    
                    <h3>💡 Quick Tips:</h3>
                    <ul>
                        <li>Take breaks every 20 minutes when working on screens</li>
                        <li>Maintain proper lighting in your workspace</li>
                        <li>Keep your eyes hydrated</li>
                        <li>Practice the 20-20-20 rule: Look 20 feet away for 20 seconds every 20 minutes</li>
                    </ul>
                    
                    <p>Your eyes will thank you! 🌟</p>
                </div>
                <div class="footer">
                    <p>© 2024 BlinkWell. All rights reserved.</p>
                    <p>This email was sent to {user_email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Time for {habit_name}!
        
        Hello {username}!
        
        It's time to take care of your eyes! Don't forget to complete your {habit_name} habit.
        
        {f'Current Streak: {streak_days} days! Keep up the great work!' if streak_days > 0 else ''}
        
        Why this habit matters:
        Regular eye care habits help maintain good vision and prevent eye strain. Every small step counts towards better eye health!
        
        Complete your habit at: {current_app.config.get('BASE_URL', 'http://localhost:5000')}/habits
        
        Quick Tips:
        - Take breaks every 20 minutes when working on screens
        - Maintain proper lighting in your workspace
        - Keep your eyes hydrated
        - Practice the 20-20-20 rule: Look 20 feet away for 20 seconds every 20 minutes
        
        Your eyes will thank you!
        
        © 2024 BlinkWell. All rights reserved.
        """
        
        return self.send_email(user_email, subject, html_content, text_content)
    
    def send_recommendation_email(self, user_email, username, recommendations, user_habits=None):
        """Send personalized eye health recommendations email"""
        subject = "👁️ Your Personalized Eye Health Recommendations"
        
        # Generate recommendations HTML
        recommendations_html = ""
        for i, rec in enumerate(recommendations, 1):
            recommendations_html += f"""
            <div style="background: #e8f5e8; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #2e7d32;">{i}. {rec['title']}</h4>
                <p style="margin: 0; color: #1b5e20;">{rec['description']}</p>
                {f'<p style="margin: 10px 0 0 0; font-size: 14px; color: #666;"><em>Based on your {rec.get("based_on", "eye health profile")}</em></p>' if rec.get('based_on') else ''}
            </div>
            """
        
        # Generate current habits summary if provided
        habits_summary = ""
        if user_habits:
            habits_summary = f"""
            <h3>📊 Your Current Eye Habits Summary:</h3>
            <div style="background: #f0f8ff; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <p><strong>Active Habits:</strong> {len(user_habits)}</p>
                <ul>
            """
            for habit in user_habits[:5]:  # Show top 5 habits
                habits_summary += f"<li>{habit['name']} - {habit.get('streak_days', 0)} day streak</li>"
            habits_summary += "</ul></div>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Your Eye Health Recommendations</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #4caf50; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>👁️ Your Eye Health Recommendations</h1>
                    <p>Personalized tips for better vision</p>
                </div>
                <div class="content">
                    <h2>Hello {username}!</h2>
                    <p>Based on your eye health profile and habits, we've prepared some personalized recommendations to help you maintain optimal vision.</p>
                    
                    {habits_summary}
                    
                    <h3>💡 Your Personalized Recommendations:</h3>
                    {recommendations_html}
                    
                    <p>Ready to implement these recommendations? Visit your dashboard to track your progress:</p>
                    
                    <a href="{current_app.config.get('BASE_URL', 'http://localhost:5000')}/dashboard" class="button">View Dashboard</a>
                    
                    <h3>🌟 Remember:</h3>
                    <ul>
                        <li>Consistency is key to building healthy eye habits</li>
                        <li>Small changes can make a big difference</li>
                        <li>Listen to your eyes and take breaks when needed</li>
                        <li>Regular eye check-ups are important</li>
                    </ul>
                    
                    <p>Keep up the great work on your eye health journey!</p>
                </div>
                <div class="footer">
                    <p>© 2024 BlinkWell. All rights reserved.</p>
                    <p>This email was sent to {user_email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Generate text version
        text_content = f"""
        Your Eye Health Recommendations
        
        Hello {username}!
        
        Based on your eye health profile and habits, we've prepared some personalized recommendations to help you maintain optimal vision.
        
        Your Personalized Recommendations:
        """
        
        for i, rec in enumerate(recommendations, 1):
            text_content += f"\n{i}. {rec['title']}\n{rec['description']}\n"
        
        text_content += f"""
        Ready to implement these recommendations? Visit your dashboard at: {current_app.config.get('BASE_URL', 'http://localhost:5000')}/dashboard
        
        Remember:
        - Consistency is key to building healthy eye habits
        - Small changes can make a big difference
        - Listen to your eyes and take breaks when needed
        - Regular eye check-ups are important
        
        Keep up the great work on your eye health journey!
        
        © 2024 BlinkWell. All rights reserved.
        """
        
        return self.send_email(user_email, subject, html_content, text_content)
    
    def send_weekly_progress_email(self, user_email, username, weekly_stats):
        """Send weekly progress summary email"""
        subject = "📊 Your Weekly Eye Health Progress Report"
        
        # Generate progress stats HTML
        progress_html = ""
        if weekly_stats.get('habits_completed'):
            progress_html += f"""
            <div style="background: #e3f2fd; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <h3 style="margin: 0 0 15px 0; color: #1565c0;">Weekly Habit Completion</h3>
                <p><strong>Total Habits Completed:</strong> {weekly_stats['habits_completed']}</p>
                <p><strong>Completion Rate:</strong> {weekly_stats.get('completion_rate', 0):.1f}%</p>
                <p><strong>Best Day:</strong> {weekly_stats.get('best_day', 'N/A')}</p>
            </div>
            """
        
        if weekly_stats.get('streak_info'):
            progress_html += f"""
            <div style="background: #fff3e0; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <h3 style="margin: 0 0 15px 0; color: #ef6c00;">Streak Highlights</h3>
                <p><strong>Longest Current Streak:</strong> {weekly_stats['streak_info'].get('longest_streak', 0)} days</p>
                <p><strong>Total Streak Days:</strong> {weekly_stats['streak_info'].get('total_streak_days', 0)} days</p>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weekly Progress Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #2196f3; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Weekly Progress Report</h1>
                    <p>Your eye health journey this week</p>
                </div>
                <div class="content">
                    <h2>Hello {username}!</h2>
                    <p>Here's a summary of your eye health progress for this week. Keep up the great work!</p>
                    
                    {progress_html}
                    
                    <h3>🎯 Next Week's Goals:</h3>
                    <ul>
                        <li>Maintain your current habit completion rate</li>
                        <li>Try to increase your streak on your best-performing habit</li>
                        <li>Consider adding a new eye health habit</li>
                        <li>Schedule a break reminder if you haven't already</li>
                    </ul>
                    
                    <p>View your detailed progress and set new goals:</p>
                    
                    <a href="{current_app.config.get('BASE_URL', 'http://localhost:5000')}/habits" class="button">View Progress</a>
                    
                    <h3>💪 Keep Going!</h3>
                    <p>Every day you practice good eye habits, you're investing in your long-term vision health. You're doing amazing!</p>
                </div>
                <div class="footer">
                    <p>© 2024 BlinkWell. All rights reserved.</p>
                    <p>This email was sent to {user_email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Generate text version
        text_content = f"""
        Weekly Progress Report
        
        Hello {username}!
        
        Here's a summary of your eye health progress for this week. Keep up the great work!
        
        Weekly Habit Completion:
        - Total Habits Completed: {weekly_stats.get('habits_completed', 0)}
        - Completion Rate: {weekly_stats.get('completion_rate', 0):.1f}%
        - Best Day: {weekly_stats.get('best_day', 'N/A')}
        
        Next Week's Goals:
        - Maintain your current habit completion rate
        - Try to increase your streak on your best-performing habit
        - Consider adding a new eye health habit
        - Schedule a break reminder if you haven't already
        
        View your detailed progress at: {current_app.config.get('BASE_URL', 'http://localhost:5000')}/habits
        
        Keep Going! Every day you practice good eye habits, you're investing in your long-term vision health. You're doing amazing!
        
        © 2024 BlinkWell. All rights reserved.
        """
        
        return self.send_email(user_email, subject, html_content, text_content)

# Create a global instance
email_service = EmailService()