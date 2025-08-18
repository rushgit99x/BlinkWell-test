// Profile form submission
document.getElementById('profileForm').addEventListener('submit', function(e) {
    e.preventDefault();
    updateProfile();
});

// Notification form submission
document.getElementById('notificationForm').addEventListener('submit', function(e) {
    e.preventDefault();
    updateNotifications();
});

// Password form submission
document.getElementById('passwordForm').addEventListener('submit', function(e) {
    e.preventDefault();
    updatePassword();
});

// Privacy form submission
document.getElementById('privacyForm').addEventListener('submit', function(e) {
    e.preventDefault();
    updatePrivacy();
});

// Update profile
function updateProfile() {
    const form = document.getElementById('profileForm');
    const submitBtn = form.querySelector('button[type="submit"]');
    
    form.classList.add('loading');
    submitBtn.disabled = true;
    
    const data = {
        username: document.getElementById('username').value,
        email: document.getElementById('email').value
    };
    
    fetch('/api/settings/profile', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage(data.message, 'success');
        } else {
            showMessage(data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showMessage('Failed to update profile. Please try again.', 'error');
    })
    .finally(() => {
        form.classList.remove('loading');
        submitBtn.disabled = false;
    });
}

// Update notifications
function updateNotifications() {
    const form = document.getElementById('notificationForm');
    const submitBtn = form.querySelector('button[type="submit"]');
    
    form.classList.add('loading');
    submitBtn.disabled = true;
    
    const data = {
        eye_exercise_reminders: document.getElementById('eye_exercise_reminders').checked,
        daily_habit_tracking: document.getElementById('daily_habit_tracking').checked,
        weekly_progress_reports: document.getElementById('weekly_progress_reports').checked,
        risk_assessment_updates: document.getElementById('risk_assessment_updates').checked,
        email_frequency: document.getElementById('email_frequency').value
    };
    
    fetch('/api/settings/notifications', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage(data.message, 'success');
        } else {
            showMessage(data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showMessage('Failed to update notifications. Please try again.', 'error');
    })
    .finally(() => {
        form.classList.remove('loading');
        submitBtn.disabled = false;
    });
}

// Update password
function updatePassword() {
    const form = document.getElementById('passwordForm');
    const submitBtn = form.querySelector('button[type="submit"]');
    
    form.classList.add('loading');
    submitBtn.disabled = true;
    
    const data = {
        current_password: document.getElementById('current_password').value,
        new_password: document.getElementById('new_password').value,
        confirm_password: document.getElementById('confirm_password').value
    };
    
    fetch('/api/settings/password', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage(data.message, 'success');
            form.reset();
        } else {
            showMessage(data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showMessage('Failed to update password. Please try again.', 'error');
    })
    .finally(() => {
        form.classList.remove('loading');
        submitBtn.disabled = false;
    });
}

// Update privacy settings
function updatePrivacy() {
    const form = document.getElementById('privacyForm');
    const submitBtn = form.querySelector('button[type="submit"]');
    
    form.classList.add('loading');
    submitBtn.disabled = true;
    
    const data = {
        share_data_research: document.getElementById('share_data_research').checked
    };
    
    fetch('/api/settings/privacy', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage(data.message, 'success');
        } else {
            showMessage(data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showMessage('Failed to update privacy settings. Please try again.', 'error');
    })
    .finally(() => {
        form.classList.remove('loading');
        submitBtn.disabled = false;
    });
}

// Show success/error messages
function showMessage(message, type) {
    const messageContainer = document.getElementById('messageContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = type === 'success' ? 'success-message' : 'error-message';
    messageDiv.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i> ${message}`;
    
    messageContainer.innerHTML = '';
    messageContainer.appendChild(messageDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// Export data function
function exportData() {
    fetch('/api/settings/export-data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage(data.message, 'success');
            // Create download link
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else {
            showMessage(data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showMessage('Failed to export data. Please try again.', 'error');
    });
}

// Clear cache function
function clearCache() {
    if (confirm('Are you sure you want to clear the cache? This may slow down the app temporarily.')) {
        fetch('/api/settings/clear-cache', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage(data.message, 'success');
            } else {
                showMessage(data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage('Failed to clear cache. Please try again.', 'error');
        });
    }
}

// Reset all data function
function resetAllData() {
    if (confirm('Are you sure you want to reset all your data? This action cannot be undone.')) {
        // Send request to the existing route
        fetch('/start-new-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('All data has been reset successfully. You can start fresh!', 'success');
            } else {
                showMessage('Error resetting data: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage('Error resetting data. Please try again.', 'error');
        });
    }
}

// Delete account function
function deleteAccount() {
    const confirmation = prompt('Are you absolutely sure? This will permanently delete your account and all data. Type "DELETE" to confirm:');
    if (confirmation === 'DELETE') {
        fetch('/api/settings/delete-account', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ confirmation: 'DELETE' })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('Account deleted successfully. Redirecting to login page...', 'success');
                setTimeout(() => {
                    window.location.href = '/login';
                }, 2000);
            } else {
                showMessage(data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage('Failed to delete account. Please try again.', 'error');
        });
    }
}

// Add smooth animations
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.settings-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'all 0.6s ease';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});