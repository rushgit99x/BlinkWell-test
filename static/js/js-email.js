// Load configuration info on page load
document.addEventListener('DOMContentLoaded', function() {
    getStatus();
});

function sendTestEmail(emailType) {
    const statusDiv = document.getElementById('test-email-status');
    statusDiv.innerHTML = '<div class="status info">Sending test email...</div>';
    
    fetch('/api/notifications/test-email', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_type: emailType })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}

function sendImmediateReminder() {
    const statusDiv = document.getElementById('manual-notification-status');
    statusDiv.innerHTML = '<div class="status info">Sending reminder...</div>';
    
    fetch('/api/notifications/send-reminder', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            habit_name: '20-20-20 Rule',
            reminder_time: 'Now'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}

function sendRecommendations() {
    const statusDiv = document.getElementById('manual-notification-status');
    statusDiv.innerHTML = '<div class="status info">Sending recommendations...</div>';
    
    fetch('/api/notifications/send-recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}

function sendWeeklyReport() {
    const statusDiv = document.getElementById('manual-notification-status');
    statusDiv.innerHTML = '<div class="status info">Sending weekly report...</div>';
    
    fetch('/api/notifications/send-weekly-report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}

function startScheduler() {
    const statusDiv = document.getElementById('scheduler-status-display');
    statusDiv.innerHTML = '<div class="status info">Starting scheduler...</div>';
    
    fetch('/api/notifications/start-scheduler', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
            setTimeout(getStatus, 1000);
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}

function stopScheduler() {
    const statusDiv = document.getElementById('scheduler-status-display');
    statusDiv.innerHTML = '<div class="status info">Stopping scheduler...</div>';
    
    fetch('/api/notifications/stop-scheduler', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
            setTimeout(getStatus, 1000);
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}

function getStatus() {
    fetch('/api/notifications/status')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const status = data.status;
            
            // Update config display
            document.getElementById('scheduler-status').textContent = status.scheduler_running ? 'Running' : 'Stopped';
            document.getElementById('smtp-server').textContent = 'smtp.gmail.com';
            document.getElementById('smtp-port').textContent = '587';
            document.getElementById('sender-email').textContent = status.email_service_configured ? 'Configured' : 'Not Configured';
            
            // Update scheduler status display
            const statusDiv = document.getElementById('scheduler-status-display');
            if (status.scheduler_running) {
                statusDiv.innerHTML = '<div class="status success">Scheduler is running</div>';
            } else {
                statusDiv.innerHTML = '<div class="status info">Scheduler is stopped</div>';
            }
        }
    })
    .catch(error => {
        console.error('Error getting status:', error);
    });
}

function getPreferences() {
    const statusDiv = document.getElementById('preferences-display');
    statusDiv.innerHTML = '<div class="status info">Loading preferences...</div>';
    
    fetch('/api/notifications/preferences')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (data.preferences && data.preferences.length > 0) {
                let html = '<div class="status success"><h4>Your Notification Preferences:</h4><ul>';
                data.preferences.forEach(pref => {
                    html += `<li><strong>${pref.habit_name}</strong>: ${pref.reminder_enabled ? 'Enabled' : 'Disabled'} at ${pref.reminder_time || 'Default'}</li>`;
                });
                html += '</ul></div>';
                statusDiv.innerHTML = html;
            } else {
                statusDiv.innerHTML = '<div class="status info">No notification preferences found</div>';
            }
        } else {
            statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
    });
}