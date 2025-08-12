@echo off
echo ========================================
echo    BlinkWell AI Chatbot WAMP Setup
echo ========================================
echo.

echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and add it to PATH
    pause
    exit /b 1
)
echo ✓ Python found: 
python --version

echo.
echo [2/6] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo ✓ Virtual environment created
)

echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated

echo.
echo [4/6] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

echo.
echo [5/6] Creating configuration files...
python config_wamp.py
echo ✓ Configuration files created

echo.
echo [6/6] Testing setup...
python test_chatbot.py
if errorlevel 1 (
    echo.
    echo ⚠️  Some tests failed. Please check the output above.
    echo.
    echo Next steps:
    echo 1. Make sure WAMP server is running
    echo 2. Create database 'flask_auth_db' in phpMyAdmin
    echo 3. Import database.sql file
    echo 4. Update .env file with your MySQL credentials
    echo 5. Run: python app.py
) else (
    echo.
    echo 🎉 Setup completed successfully!
    echo.
    echo Next steps:
    echo 1. Make sure WAMP server is running
    echo 2. Create database 'flask_auth_db' in phpMyAdmin
    echo 3. Import database.sql file
    echo 4. Update .env file with your MySQL credentials
    echo 5. Run: python app.py
    echo 6. Access chatbot at: http://localhost:5000/chat
)

echo.
echo Press any key to exit...
pause >nul