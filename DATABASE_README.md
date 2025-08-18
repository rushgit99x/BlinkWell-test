# BlinkWell Database Setup Guide

This guide will help you set up the complete BlinkWell database with admin panel support.

## 🗄️ Database Overview

The BlinkWell database includes:

### **Admin Panel Tables**
- `admin_users` - Admin user accounts and roles
- `admin_activity_logs` - Audit trail of all admin actions
- `admin_permissions` - Role-based access control permissions

### **Application Tables**
- `users` - Regular user accounts
- `eye_habits` - Eye health habits and exercises
- `user_habits` - User-specific habit assignments
- `habit_tracking` - Daily habit completion tracking
- `user_eye_health_data` - User health and risk assessment data
- `habit_achievements` - User achievements and badges
- `habit_summaries` - Weekly/monthly habit summaries
- `user_notification_preferences` - User notification settings
- `user_privacy_settings` - User privacy preferences
- `user_recommendations` - Personalized health recommendations

## 🚀 Quick Setup

### **Option 1: Automated Setup (Recommended)**

#### **Windows Users**
```bash
# Double-click the batch file
setup_database.bat
```

#### **Linux/Mac Users**
```bash
# Make executable and run
chmod +x setup_database.sh
./setup_database.sh
```

#### **Manual Python Execution**
```bash
# Install dependencies
pip install Flask-MySQLdb PyMySQL python-dotenv

# Run setup
python setup_database.py
```

### **Option 2: Manual SQL Execution**

1. **Connect to MySQL**
```bash
mysql -u your_username -p
```

2. **Execute the SQL file**
```sql
source database_setup.sql;
```

## ⚙️ Prerequisites

### **Required Software**
- **Python 3.7+** with pip
- **MySQL 5.7+** or **MariaDB 10.2+**
- **MySQL Connector** for Python

### **Required Python Packages**
```bash
pip install Flask-MySQLdb PyMySQL python-dotenv
```

### **Database Configuration**
Ensure your `config.py` has the correct database settings:

```python
class Config:
    MYSQL_HOST = 'localhost'  # or your MySQL host
    MYSQL_USER = 'your_username'
    MYSQL_PASSWORD = 'your_password'
    MYSQL_DB = 'b_test9'  # or your preferred database name
```

## 📋 Setup Process

### **Step 1: Database Connection**
The setup script will:
- Verify your database configuration
- Create the database if it doesn't exist
- Establish a connection

### **Step 2: Table Creation**
Creates all necessary tables with:
- Proper data types and constraints
- Indexes for performance
- Foreign key relationships
- Timestamp fields for tracking

### **Step 3: Default Data**
Inserts:
- **Default admin user**: `admin` / `admin123`
- **Sample eye habits** (10 pre-configured habits)
- **Permission system** for role-based access

### **Step 4: Database Objects**
Creates:
- **Views** for common queries
- **Stored procedures** for statistics
- **Triggers** for automatic updates
- **Indexes** for performance optimization

## 🔐 Default Admin Access

After setup, you'll have access to:

- **Username**: `admin`
- **Password**: `admin123`
- **Role**: `super_admin`
- **Access**: Full system access

**⚠️ IMPORTANT**: Change this password immediately after first login!

## 📊 Database Features

### **Performance Optimizations**
- **Indexes** on frequently queried columns
- **Composite indexes** for complex queries
- **Optimized data types** for storage efficiency

### **Data Integrity**
- **Foreign key constraints** where appropriate
- **Unique constraints** on critical fields
- **Check constraints** for data validation

### **Monitoring & Maintenance**
- **Activity logging** for all admin actions
- **Audit trails** with IP addresses and timestamps
- **Automatic timestamp updates** via triggers

## 🛠️ Customization

### **Adding New Tables**
1. Add table creation SQL to `database_setup.sql`
2. Update the verification in `setup_database.py`
3. Add any necessary indexes or constraints

### **Modifying Existing Tables**
1. Use `ALTER TABLE` statements
2. Consider data migration for existing data
3. Test changes in development first

### **Adding Sample Data**
1. Insert statements in the "INSERT DEFAULT DATA" section
2. Ensure data consistency with existing records
3. Use appropriate foreign key references

## 🔍 Verification

After setup, verify:

### **Table Count**
```sql
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'b_test9';
```
Expected: 15+ tables

### **Admin User**
```sql
SELECT username, email, role FROM admin_users 
WHERE username = 'admin';
```
Expected: admin user with super_admin role

### **Sample Habits**
```sql
SELECT COUNT(*) FROM eye_habits;
```
Expected: 10+ sample habits

### **Permissions**
```sql
SELECT COUNT(*) FROM admin_permissions;
```
Expected: 20+ permission entries

## 🚨 Troubleshooting

### **Common Issues**

#### **Connection Errors**
```
❌ Database connection failed: [Errno 2003] Can't connect to MySQL server
```
**Solution**: Check MySQL service is running and credentials are correct

#### **Permission Errors**
```
❌ Error creating database: (1044, "Access denied for user...")
```
**Solution**: Ensure MySQL user has CREATE DATABASE privileges

#### **Table Creation Errors**
```
❌ Error executing statement: (1050, "Table '...' already exists")
```
**Solution**: Drop existing tables or use DROP TABLE IF EXISTS

#### **Character Set Issues**
```
❌ Error: Unknown character set 'utf8mb4'
```
**Solution**: Upgrade to MySQL 5.5+ or use 'utf8' instead

### **Debug Mode**
Enable detailed error reporting:

```python
# In setup_database.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Manual Verification**
Connect to MySQL and run:
```sql
USE b_test9;
SHOW TABLES;
DESCRIBE admin_users;
SELECT * FROM admin_users;
```

## 📈 Performance Tuning

### **Index Optimization**
The setup creates indexes for:
- User lookups by email/username
- Habit tracking by user and date
- Activity logs by timestamp
- Category-based habit queries

### **Query Optimization**
Use the provided views:
- `user_habit_summary` - User statistics
- `habit_performance` - Habit analytics
- `daily_activity_summary` - Activity trends

### **Maintenance Procedures**
```sql
-- Clean up old logs (keep last 90 days)
CALL CleanupOldLogs(90);

-- Get user statistics
CALL GetUserStats(1);

-- Get habit statistics
CALL GetHabitStats(1);
```

## 🔒 Security Considerations

### **Database Security**
- Use strong passwords for MySQL users
- Limit database user privileges
- Enable SSL connections in production
- Regular security updates

### **Application Security**
- Change default admin password
- Use environment variables for sensitive data
- Implement rate limiting for admin access
- Regular security audits

## 📚 Additional Resources

### **MySQL Documentation**
- [MySQL 8.0 Reference Manual](https://dev.mysql.com/doc/refman/8.0/en/)
- [MySQL Security Best Practices](https://dev.mysql.com/doc/refman/8.0/en/security-best-practices.html)

### **Python MySQL Connectors**
- [PyMySQL Documentation](https://pymysql.readthedocs.io/)
- [Flask-MySQLdb Documentation](https://flask-mysqldb.readthedocs.io/)

### **Database Design**
- [Database Normalization](https://en.wikipedia.org/wiki/Database_normalization)
- [Index Optimization](https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html)

## 🆘 Support

If you encounter issues:

1. **Check the error messages** carefully
2. **Verify prerequisites** are met
3. **Check MySQL logs** for detailed errors
4. **Review the troubleshooting section** above
5. **Enable debug mode** for more information

## 📝 Changelog

### **Version 1.0**
- Initial database setup
- Admin panel tables
- Sample data and permissions
- Performance optimizations
- Comprehensive documentation

---

**Happy coding! 🎉**

For questions or issues, please refer to the main project documentation or create an issue in the project repository.