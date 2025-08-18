# BlinkWell Admin Panel

A comprehensive admin panel for managing the BlinkWell eye health application, built with Flask and modern web technologies.

## Features

### 🔐 Authentication & Security
- **Role-based access control** with three levels:
  - `super_admin`: Full access to all features
  - `admin`: Access to most features (read/write)
  - `moderator`: Read-only access to basic features
- **Secure login system** with password hashing
- **Activity logging** for all admin actions
- **Permission-based access** to different resources

### 📊 Dashboard
- **Real-time statistics** showing:
  - Total users, habits, tracking records, and health data
  - Recent user registrations and habit completions
  - Top performing habits with completion rates
  - Daily user activity charts
  - Recent admin activity logs
- **Interactive charts** using Chart.js
- **Auto-refresh** every 30 seconds

### 👥 User Management
- **User listing** with pagination (20 users per page)
- **User search** by username or email
- **User details** including:
  - Profile information
  - Assigned habits
  - Habit tracking history
  - Eye health data
  - Achievements
- **User actions**: View details, send messages, suspend users

### 🏃‍♂️ Eye Habits Management
- **Habit listing** with usage statistics
- **Create new habits** with comprehensive forms:
  - Basic information (name, category, icon)
  - Target settings (frequency, count, unit)
  - Difficulty and time estimates
  - Instructions and benefits
  - Live preview functionality
- **Edit existing habits**
- **Toggle habit status** (active/inactive)
- **Delete habits** with safety checks
- **Category-based filtering**

### 📈 Analytics & Reporting
- **User registration trends** over time
- **Habit completion trends**
- **Category performance** analysis
- **User engagement metrics**
- **Customizable date ranges** (7, 30, 90 days)

### 👨‍💼 Admin User Management
- **Create new admin users**
- **Edit admin user details**
- **Manage admin roles and permissions**
- **Soft delete** admin users
- **Activity logging** for all admin actions

### 📝 Activity Logs
- **Comprehensive logging** of all admin actions
- **Detailed information** including:
  - Admin user who performed the action
  - Action type and target resource
  - IP address and user agent
  - Timestamp
- **Pagination** for large log files

## Installation & Setup

### 1. Database Setup
The admin panel will automatically create the necessary tables on first run. Visit `/admin/setup` to initialize the admin system.

### 2. Default Admin User
After setup, a default admin user is created:
- **Username**: `admin`
- **Password**: `admin123`
- **Role**: `super_admin`

**⚠️ Important**: Change the default password immediately after first login!

### 3. Access the Admin Panel
Navigate to `/admin` in your browser. You'll be redirected to the login page if not authenticated.

## Database Schema

### Admin Tables

#### `admin_users`
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `role`: User role (super_admin, admin, moderator)
- `is_active`: Account status
- `last_login`: Last login timestamp
- `created_at`, `updated_at`: Timestamps

#### `admin_activity_logs`
- `id`: Primary key
- `admin_id`: Reference to admin user
- `action`: Action performed
- `table_name`: Affected table
- `record_id`: Affected record ID
- `details`: JSON details of the action
- `ip_address`: IP address of the admin
- `user_agent`: Browser/user agent info
- `created_at`: Timestamp

#### `admin_permissions`
- `id`: Primary key
- `role`: Admin role
- `resource`: Resource being accessed
- `action`: Action allowed (read, write, delete)

## API Endpoints

### Authentication
- `GET /admin/login` - Login page
- `POST /admin/login` - Process login
- `GET /admin/logout` - Logout

### Dashboard
- `GET /admin/dashboard` - Main dashboard
- `GET /admin/api/stats` - Real-time statistics API

### User Management
- `GET /admin/users` - User listing
- `GET /admin/users/<id>` - User details

### Habit Management
- `GET /admin/habits` - Habit listing
- `GET /admin/habits/create` - Create habit form
- `POST /admin/habits/create` - Process habit creation
- `GET /admin/habits/<id>/edit` - Edit habit form
- `POST /admin/habits/<id>/edit` - Process habit updates
- `POST /admin/habits/<id>/delete` - Delete habit

### Analytics
- `GET /admin/analytics` - Analytics dashboard

### Admin Users
- `GET /admin/admin-users` - Admin user listing
- `GET /admin/admin-users/create` - Create admin user form
- `POST /admin/admin-users/create` - Process admin user creation
- `GET /admin/admin-users/<id>/edit` - Edit admin user form
- `POST /admin/admin-users/<id>/edit` - Process admin user updates
- `POST /admin/admin-users/<id>/delete` - Delete admin user

### Activity Logs
- `GET /admin/activity-logs` - Activity log listing

### Setup
- `GET /admin/setup` - Initial setup page

## Security Features

### Permission System
The admin panel uses a granular permission system:

```python
@permission_required('users', 'read')  # Can read user data
@permission_required('users', 'write') # Can modify user data
@permission_required('users', 'delete') # Can delete user data
```

### Role Hierarchy
1. **super_admin**: Full access to all resources
2. **admin**: Read/write access to most resources
3. **moderator**: Read-only access to basic resources

### Activity Logging
All admin actions are logged with:
- Admin user ID
- Action performed
- Resource affected
- IP address
- User agent
- Timestamp

## Customization

### Adding New Resources
To add new resources to the admin panel:

1. **Create the model** in `models/admin.py`
2. **Add routes** in `routes/admin.py`
3. **Create templates** in `templates/admin/`
4. **Update permissions** in the setup function
5. **Add navigation** to the sidebar

### Styling
The admin panel uses:
- **Bootstrap 5** for responsive design
- **Font Awesome** for icons
- **Chart.js** for data visualization
- **Custom CSS** for branding and layout

### JavaScript Features
- **Real-time updates** via AJAX
- **Form validation** and auto-save
- **Interactive modals** and confirmations
- **Search and filtering** functionality
- **Responsive charts** and data visualization

## Best Practices

### Security
- Always use HTTPS in production
- Regularly rotate admin passwords
- Monitor activity logs for suspicious activity
- Limit admin user access to necessary roles only

### Performance
- Use pagination for large datasets
- Implement caching for frequently accessed data
- Optimize database queries
- Use AJAX for real-time updates

### Maintenance
- Regularly backup admin activity logs
- Monitor database size and performance
- Update admin panel dependencies
- Review and clean up old activity logs

## Troubleshooting

### Common Issues

#### Admin Tables Not Created
- Visit `/admin/setup` to initialize the admin system
- Check database connection settings
- Ensure MySQL user has CREATE TABLE permissions

#### Login Issues
- Verify admin user exists in database
- Check password hash is correct
- Ensure user account is active

#### Permission Errors
- Verify admin user has correct role
- Check permissions table has correct entries
- Ensure role-based access control is working

### Debug Mode
Enable Flask debug mode to see detailed error messages:

```python
app.run(debug=True)
```

## Support

For issues or questions about the admin panel:
1. Check the activity logs for error details
2. Verify database connectivity and permissions
3. Review the permission system configuration
4. Check browser console for JavaScript errors

## License

This admin panel is part of the BlinkWell project and follows the same licensing terms.