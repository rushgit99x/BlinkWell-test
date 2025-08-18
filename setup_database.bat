@echo off
REM BlinkWell Database Setup Script for Windows
REM This script sets up the complete database with admin panel support

echo.
echo ============================================================
echo 🚀 BlinkWell Database Setup for Windows
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if required files exist
if not exist "config.py" (
    echo ❌ config.py not found
    echo Please ensure config.py exists with correct database settings
    pause
    exit /b 1
)

if not exist "database_setup.sql" (
    echo ❌ database_setup.sql not found
    echo Please ensure database_setup.sql exists
    pause
    exit /b 1
)

echo ✅ Required files found

REM Install required packages if needed
echo.
echo 📦 Checking/Installing required packages...
pip install Flask-MySQLdb PyMySQL python-dotenv

REM Run the database setup
echo.
echo 🔧 Running database setup...
python setup_database.py

REM Check if setup was successful
if errorlevel 1 (
    echo.
    echo ❌ Database setup failed!
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo 🎉 Database setup completed successfully!
echo.
echo 📋 Next steps:
echo    1. Start your Flask application: python app.py
echo    2. Navigate to /admin in your browser
echo    3. Login with admin/admin123
echo    4. Change the default password immediately
echo.
pause