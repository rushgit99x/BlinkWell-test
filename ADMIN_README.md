# BlinkWell Admin Panel

A comprehensive admin panel for the BlinkWell eye health application, providing system monitoring, user management, analytics, and administrative controls.

## Features

### 🎯 Dashboard
- **System Overview**: Real-time statistics and key metrics
- **Quick Actions**: Easy access to common admin functions
- **Recent Activity**: Live feed of system and user activities
- **System Health**: Real-time monitoring of application status

### 👥 User Management
- **User List**: View all users with pagination and search
- **User Details**: Edit user information, permissions, and status
- **Admin Controls**: Grant/revoke admin privileges
- **User Actions**: Activate/deactivate accounts, delete users

### 📊 Analytics & Reporting
- **User Growth**: Visual charts showing user acquisition over time
- **Activity Metrics**: User engagement and feature usage statistics
- **Demographics**: Age groups and geographic distribution
- **Export Reports**: Download analytics data in various formats

### 🖥️ System Monitoring
- **Resource Usage**: CPU, memory, and disk utilization
- **Database Status**: Connection health and performance metrics
- **Network Monitoring**: I/O statistics and connection status
- **Performance Metrics**: Response times and error rates

### 📝 Log Management
- **System Logs**: View and filter application logs
- **Log Levels**: Filter by DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Source Filtering**: Filter by application, auth, database, API, system
- **Search & Export**: Advanced log search and export functionality

## Installation & Setup

### 1. Database Setup

First, update your database schema to include admin fields:

```sql
-- Run this SQL to update your existing users table
ALTER TABLE users 
ADD COLUMN is_admin BOOLEAN DEFAULT FALSE,
ADD COLUMN is_active BOOLEAN DEFAULT TRUE,
ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN last_login TIMESTAMP NULL;
```

Or use the updated `database.sql` file provided.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Admin User

Run the admin user creation script:

```bash
python create_admin.py
```

Follow the prompts to create your first admin user.

### 4. Access Admin Panel

Once logged in as an admin user, navigate to:
```
http://your-domain/admin
```

## Usage

### Accessing the Admin Panel

1. **Login**: Use your admin credentials to log into the main application
2. **Navigate**: Go to `/admin` in your browser
3. **Dashboard**: View system overview and quick actions

### User Management

1. **View Users**: Navigate to Users section to see all registered users
2. **Edit User**: Click the edit button to modify user details
3. **Admin Rights**: Toggle admin privileges for users
4. **User Status**: Activate/deactivate user accounts

### System Monitoring

1. **Health Check**: Monitor system status in real-time
2. **Resource Usage**: Track CPU, memory, and disk utilization
3. **Performance**: Monitor response times and error rates
4. **Logs**: View and filter system logs for troubleshooting

### Analytics

1. **User Growth**: Analyze user acquisition trends
2. **Feature Usage**: Track which features are most popular
3. **Demographics**: Understand your user base
4. **Export Data**: Download reports for external analysis

## Security Features

### Admin Authentication
- **Role-based Access**: Only users with `is_admin=True` can access the panel
- **Session Management**: Secure session handling with Flask-Login
- **Permission Checks**: Decorator-based admin verification

### Data Protection
- **Input Validation**: All user inputs are validated and sanitized
- **SQL Injection Prevention**: Parameterized queries for database operations
- **XSS Protection**: Template escaping and secure output handling

### Audit Trail
- **Action Logging**: All admin actions are logged for audit purposes
- **User Tracking**: Monitor who made changes and when
- **Change History**: Track modifications to user accounts and system settings

## Configuration

### Environment Variables

Ensure these environment variables are set in your `.env` file:

```env
MYSQL_HOST=localhost
MYSQL_USER=your_db_user
MYSQL_PASSWORD=your_db_password
MYSQL_DB=blinkwell
SECRET_KEY=your_secret_key
```

### Database Connection

The admin panel uses the same database connection as your main application. Ensure your database connection function is properly configured in `config.py`.

## Customization

### Adding New Admin Features

1. **Create Route**: Add new routes in `routes/admin.py`
2. **Add Template**: Create corresponding HTML templates in `templates/admin/`
3. **Update Navigation**: Add menu items to the admin sidebar
4. **Style**: Use the existing CSS classes for consistent styling

### Modifying Existing Features

- **Templates**: Edit HTML files in `templates/admin/`
- **Styles**: Modify `static/css/admin.css`
- **Logic**: Update functions in `routes/admin.py`
- **Database**: Add new queries and data processing

## Troubleshooting

### Common Issues

1. **Access Denied**: Ensure user has `is_admin=True` in database
2. **Database Errors**: Check database connection and schema
3. **Template Errors**: Verify all template files exist and are properly formatted
4. **CSS Issues**: Ensure admin.css is properly linked

### Debug Mode

Enable Flask debug mode to see detailed error messages:

```python
app.run(debug=True)
```

### Log Files

Check your application logs for detailed error information. The admin panel logs all actions and errors for debugging purposes.

## API Endpoints

### Admin Routes

- `GET /admin` - Admin dashboard
- `GET /admin/users` - User management
- `GET /admin/users/<id>` - Edit user
- `POST /admin/users/<id>` - Update user
- `POST /admin/users/<id>/delete` - Delete user
- `GET /admin/analytics` - Analytics dashboard
- `GET /admin/system` - System status
- `GET /admin/logs` - System logs
- `GET /admin/api/stats` - API statistics

### Authentication Required

All admin routes require:
- User to be logged in (`@login_required`)
- User to have admin privileges (`@admin_required`)

## Contributing

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add docstrings to all functions
- Include type hints where appropriate

### Testing

- Test all admin functions thoroughly
- Verify security measures work correctly
- Test with different user permission levels
- Validate all form inputs and outputs

## License

This admin panel is part of the BlinkWell project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review application logs for error details
3. Verify database schema and connections
4. Ensure all dependencies are properly installed

---

**Note**: This admin panel provides powerful system management capabilities. Use responsibly and ensure proper security measures are in place for production deployments.