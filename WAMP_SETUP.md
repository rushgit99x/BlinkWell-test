# 🚀 WAMP Server Setup Guide for BlinkWell AI Chatbot

This guide will help you set up the AI chatbot system on your WAMP server (Windows + Apache + MySQL + PHP).

## 📋 Prerequisites

- **WAMP Server** installed and running
- **Python 3.7+** installed
- **pip** package manager
- **Git** (optional, for version control)

## 🔧 Step 1: Start WAMP Server

1. **Launch WAMP Server**
   - Double-click the WAMP icon in your system tray
   - Wait for the icon to turn green (indicating all services are running)

2. **Verify Services**
   - Apache should be running on port 80
   - MySQL should be running on port 3306
   - Check by clicking the WAMP icon → Apache → Service → Status

## 🗄️ Step 2: Database Setup

### Option A: Using phpMyAdmin (Recommended)

1. **Open phpMyAdmin**
   - Go to `http://localhost/phpmyadmin` in your browser
   - Login with username: `root` (password is usually empty in WAMP)

2. **Create Database**
   - Click "New" on the left sidebar
   - Enter database name: `flask_auth_db`
   - Select collation: `utf8mb4_unicode_ci`
   - Click "Create"

3. **Import Schema**
   - Select the `flask_auth_db` database
   - Click "Import" tab
   - Click "Choose File" and select `database.sql`
   - Click "Go" to execute

### Option B: Using MySQL Command Line

1. **Open MySQL Command Line**
   - Click WAMP icon → MySQL → MySQL Console
   - Press Enter when prompted for password (usually empty)

2. **Execute Commands**
   ```sql
   -- Copy and paste the contents of database.sql
   -- Or run this file directly:
   source C:/path/to/your/project/database.sql;
   ```

## 🐍 Step 3: Python Environment Setup

1. **Create Virtual Environment** (Recommended)
   ```bash
   # Navigate to your project directory
   cd C:\wamp64\www\your-project-folder
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Environment File**
   ```bash
   python config_wamp.py
   ```

## ⚙️ Step 4: Configuration

1. **Update .env File**
   - Open the `.env` file created in the previous step
   - Update MySQL credentials if needed:
   ```env
   MYSQL_HOST=localhost
   MYSQL_USER=root
   MYSQL_PASSWORD=
   MYSQL_DB=flask_auth_db
   MYSQL_PORT=3306
   ```

2. **Verify Database Connection**
   - Test the connection using the test script:
   ```bash
   python test_chatbot.py
   ```

## 🚀 Step 5: Run the Application

1. **Start Flask App**
   ```bash
   python app.py
   ```

2. **Access the Application**
   - Main app: `http://localhost:5000`
   - Chatbot: `http://localhost:5000/chat`
   - FAQ page: `http://localhost:5000/chat/faq`
   - Admin panel: `http://localhost:5000/admin/knowledge-base`

## 🔍 Step 6: Testing

### Test Database Connection
```bash
python test_chatbot.py
```

### Test Chatbot Functionality
1. Go to `http://localhost:5000/chat`
2. Try asking questions like:
   - "What is this web app about?"
   - "How do I use the eye disease detection?"
   - "Hello there!"

### Test Admin Panel
1. Go to `http://localhost:5000/admin/knowledge-base`
2. Try adding a new FAQ item
3. Test search and filtering

## 🛠️ Troubleshooting

### Common WAMP Issues

#### 1. **Port Conflicts**
- **Apache Port 80**: If port 80 is busy, change it in `httpd.conf`
- **MySQL Port 3306**: If port 3306 is busy, change it in `my.ini`

#### 2. **MySQL Connection Issues**
```bash
# Check if MySQL is running
netstat -an | findstr 3306

# Restart MySQL service
# Right-click WAMP icon → MySQL → Service → Restart Service
```

#### 3. **Permission Issues**
- Ensure your project folder has read/write permissions
- Check if antivirus is blocking Python or MySQL

#### 4. **Python Path Issues**
```bash
# Verify Python installation
python --version
pip --version

# If not found, add Python to PATH environment variable
```

### Database Issues

#### 1. **Table Creation Fails**
```sql
-- Check if database exists
SHOW DATABASES;

-- Check if user has privileges
SHOW GRANTS FOR 'root'@'localhost';

-- Grant privileges if needed
GRANT ALL PRIVILEGES ON flask_auth_db.* TO 'root'@'localhost';
FLUSH PRIVILEGES;
```

#### 2. **Character Encoding Issues**
```sql
-- Check current character set
SHOW VARIABLES LIKE 'character_set%';

-- Set proper character set
SET NAMES utf8mb4;
```

## 📁 File Structure

After setup, your project should look like this:
```
your-project-folder/
├── app.py
├── config.py
├── config_wamp.py
├── requirements.txt
├── database.sql
├── .env
├── models/
│   └── chatbot.py
├── routes/
│   ├── chatbot.py
│   └── admin.py
├── templates/
│   ├── chatbot/
│   │   ├── chat.html
│   │   └── faq.html
│   └── admin/
│       └── knowledge_base.html
└── static/
    └── css/
```

## 🔒 Security Considerations

1. **Change Default Passwords**
   - Update MySQL root password
   - Change Flask secret key in `.env`

2. **Environment Variables**
   - Never commit `.env` file to version control
   - Use strong, unique secret keys

3. **Database Security**
   - Create dedicated database user (not root)
   - Limit user privileges to necessary operations

## 📊 Performance Optimization

1. **MySQL Configuration**
   - Adjust `my.ini` settings for your hardware
   - Enable query cache
   - Optimize buffer sizes

2. **Python Optimization**
   - Use production WSGI server (Gunicorn/uWSGI)
   - Enable Flask caching
   - Use connection pooling

## 🚀 Production Deployment

When moving to production:

1. **Disable Debug Mode**
   ```python
   DEBUG = False
   ```

2. **Use Production WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Set Up Reverse Proxy**
   - Configure Apache/Nginx to proxy to Flask
   - Use SSL certificates for HTTPS

4. **Database Optimization**
   - Use dedicated MySQL server
   - Implement proper backup strategies
   - Monitor performance metrics

## 📞 Support

If you encounter issues:

1. **Check WAMP Logs**
   - Apache logs: `C:\wamp64\logs\apache_error.log`
   - MySQL logs: `C:\wamp64\logs\mysql.log`

2. **Python Debugging**
   - Enable Flask debug mode
   - Check console output for errors

3. **Database Debugging**
   - Use phpMyAdmin to verify table structure
   - Test queries directly in MySQL console

## 🎯 Next Steps

After successful setup:

1. **Customize Knowledge Base**
   - Add your own FAQ items
   - Organize by relevant categories
   - Import existing FAQ data

2. **Integrate with Existing App**
   - Add chatbot links to your navigation
   - Customize styling to match your theme
   - Test with real users

3. **Monitor and Improve**
   - Track user questions and responses
   - Update knowledge base based on usage
   - Optimize response accuracy

---

**🎉 Congratulations!** Your AI chatbot is now running on WAMP server and ready to provide intelligent support to your users.