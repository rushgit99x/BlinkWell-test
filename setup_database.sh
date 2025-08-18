#!/bin/bash

# BlinkWell Database Setup Script for Linux/Mac
# This script sets up the complete database with admin panel support

echo
echo "============================================================"
echo "🚀 BlinkWell Database Setup for Linux/Mac"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

echo "✅ Python found"
python3 --version

# Check if required files exist
if [ ! -f "config.py" ]; then
    echo "❌ config.py not found"
    echo "Please ensure config.py exists with correct database settings"
    exit 1
fi

if [ ! -f "database_setup.sql" ]; then
    echo "❌ database_setup.sql not found"
    echo "Please ensure database_setup.sql exists"
    exit 1
fi

echo "✅ Required files found"

# Install required packages if needed
echo
echo "📦 Checking/Installing required packages..."
pip3 install Flask-MySQLdb PyMySQL python-dotenv

# Make the setup script executable
chmod +x setup_database.py

# Run the database setup
echo
echo "🔧 Running database setup..."
python3 setup_database.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo
    echo "🎉 Database setup completed successfully!"
    echo
    echo "📋 Next steps:"
    echo "   1. Start your Flask application: python3 app.py"
    echo "   2. Navigate to /admin in your browser"
    echo "   3. Login with admin/admin123"
    echo "   4. Change the default password immediately"
    echo
else
    echo
    echo "❌ Database setup failed!"
    echo "Please check the error messages above"
    exit 1
fi