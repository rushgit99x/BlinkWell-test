// Add this to a new file: static/js/habit-notifications.js

class HabitNotificationManager {
    constructor() {
        this.notificationPermission = 'default';
        this.reminderTimers = new Map();
        this.init();
    }

    async init() {
        // Request notification permission
        if ('Notification' in window) {
            this.notificationPermission = await Notification.requestPermission();
        }

        // Check for service worker support
        if ('serviceWorker' in navigator) {
            this.registerServiceWorker();
        }

        // Setup reminder system
        this.setupHabitReminders();
    }

    async registerServiceWorker() {
        try {
            const registration = await navigator.serviceWorker.register('/static/js/sw.js');
            console.log('Service Worker registered:', registration);
        } catch (error) {
            console.log('Service Worker registration failed:', error);
        }
    }

    async setupHabitReminders() {
        try {
            const response = await fetch('/api/habits/user-habits');
            const data = await response.json();

            if (data.success) {
                this.scheduleReminders(data.habits);
            }
        } catch (error) {
            console.error('Error setting up reminders:', error);
        }
    }

    scheduleReminders(habits) {
        // Clear existing timers
        this.reminderTimers.forEach(timer => clearTimeout(timer));
        this.reminderTimers.clear();

        habits.forEach(habit => {
            if (habit.reminder_enabled && habit.reminder_time) {
                this.scheduleHabitReminder(habit);
            }
        });

        // Schedule 20-20-20 reminders every 20 minutes
        this.schedule20_20_20Reminders();
    }

    scheduleHabitReminder(habit) {
        const now = new Date();
        const reminderTime = new Date();
        const [hours, minutes] = habit.reminder_time.split(':');
        
        reminderTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);

        // If reminder time has passed today, schedule for tomorrow
        if (reminderTime <= now) {
            reminderTime.setDate(reminderTime.getDate() + 1);
        }

        const timeUntilReminder = reminderTime.getTime() - now.getTime();

        const timer = setTimeout(() => {
            this.showHabitReminder(habit);
            // Reschedule for next day
            this.scheduleHabitReminder(habit);
        }, timeUntilReminder);

        this.reminderTimers.set(`habit_${habit.user_habit_id}`, timer);
    }

    schedule20_20_20Reminders() {
        const timer = setInterval(() => {
            this.show20_20_20Reminder();
        }, 20 * 60 * 1000); // Every 20 minutes

        this.reminderTimers.set('20_20_20', timer);
    }

    showHabitReminder(habit) {
        if (this.notificationPermission === 'granted') {
            const notification = new Notification(`Time for ${habit.name}!`, {
                body: habit.description,
                icon: '/static/img/logo.png',
                badge: '/static/img/badge.png',
                tag: `habit_${habit.user_habit_id}`,
                requireInteraction: true,
                actions: [
                    { action: 'complete', title: 'Mark Complete' },
                    { action: 'snooze', title: 'Remind in 10min' }
                ]
            });

            notification.onclick = () => {
                window.focus();
                window.location.href = '/habits';
                notification.close();
            };

            // Auto-close after 10 seconds
            setTimeout(() => notification.close(), 10000);
        }
    }

    show20_20_20Reminder() {
        if (this.notificationPermission === 'granted') {
            const notification = new Notification('20-20-20 Break Time!', {
                body: 'Look at something 20 feet away for 20 seconds.',
                icon: '/static/img/logo.png',
                tag: '20_20_20_reminder',
                requireInteraction: false
            });

            // Auto-close after 5 seconds
            setTimeout(() => notification.close(), 5000);
        }

        // Also show in-page notification
        this.showInPageNotification('Time for your 20-20-20 break! Look away from the screen.', 'info');
    }

    showInPageNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            z-index: 1003;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            max-width: 300px;
        `;
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-bell"></i>
                <div>${message}</div>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: white; cursor: pointer; margin-left: auto;">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.parentElement.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
}

// Habit Analytics and Insights
class HabitAnalytics {
    constructor() {
        this.charts = {};
    }

    async generateWeeklyReport() {
        try {
            const response = await fetch('/api/habits/analytics/7');
            const data = await response.json();

            if (data.success) {
                return this.formatWeeklyReport(data);
            }
        } catch (error) {
            console.error('Error generating weekly report:', error);
        }
        return null;
    }

    formatWeeklyReport(data) {
        const report = {
            summary: {
                totalDays: data.period_days,
                averageCompletion: 0,
                bestDay: null,
                improvementTrend: 'stable'
            },
            categories: {},
            recommendations: []
        };

        // Calculate summary stats
        if (data.daily_analytics.length > 0) {
            const completions = data.daily_analytics.map(day => day.avg_completion || 0);
            report.summary.averageCompletion = Math.round(
                completions.reduce((a, b) => a + b, 0) / completions.length
            );

            // Find best day
            const bestDayData = data.daily_analytics.reduce((best, current) => 
                (current.avg_completion || 0) > (best.avg_completion || 0) ? current : best
            );
            report.summary.bestDay = bestDayData.date;

            // Determine trend
            const firstHalf = completions.slice(0, Math.floor(completions.length / 2));
            const secondHalf = completions.slice(Math.floor(completions.length / 2));
            const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
            const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

            if (secondAvg > firstAvg + 5) {
                report.summary.improvementTrend = 'improving';
            } else if (secondAvg < firstAvg - 5) {
                report.summary.improvementTrend = 'declining';
            }
        }

        // Process category data
        data.category_analytics.forEach(category => {
            report.categories[category.category] = {
                completion: Math.round(category.avg_completion || 0),
                totalEntries: category.total_tracking_entries,
                completedCount: category.completed_count
            };
        });

        // Generate improvement recommendations
        report.recommendations = this.generateImprovementRecommendations(report);

        return report;
    }

    generateImprovementRecommendations(report) {
        const recommendations = [];

        // Low completion rate recommendations
        if (report.summary.averageCompletion < 70) {
            recommendations.push({
                type: 'improvement',
                title: 'Focus on Consistency',
                message: 'Try to complete at least one habit fully each day rather than partially completing many.',
                priority: 'high'
            });
        }

        // Category-specific recommendations
        Object.entries(report.categories).forEach(([category, stats]) => {
            if (stats.completion < 50) {
                const categoryNames = {
                    'screen_health': 'Screen Health',
                    'exercise': 'Eye Exercises',
                    'hydration': 'Hydration',
                    'sleep': 'Sleep Quality',
                    'environment': 'Environment',
                    'nutrition': 'Nutrition'
                };

                recommendations.push({
                    type: 'category_focus',
                    title: `Improve ${categoryNames[category]} Habits`,
                    message: `Your ${categoryNames[category].toLowerCase()} habits need attention. Consider setting reminders or reducing difficulty.`,
                    priority: 'medium'
                });
            }
        });

        // Trend-based recommendations
        if (report.summary.improvementTrend === 'declining') {
            recommendations.push({
                type: 'motivation',
                title: 'Stay Motivated',
                message: 'Your completion rate has been declining. Consider reviewing your goals or adjusting habit difficulty.',
                priority: 'high'
            });
        } else if (report.summary.improvementTrend === 'improving') {
            recommendations.push({
                type: 'celebration',
                title: 'Great Progress!',
                message: 'Your habit completion is improving. Consider adding a new challenging habit.',
                priority: 'low'
            });
        }

        return recommendations;
    }

    async createProgressChart(elementId, days = 7) {
        try {
            const response = await fetch(`/api/habits/analytics/${days}`);
            const data = await response.json();

            if (data.success && data.daily_analytics.length > 0) {
                this.renderProgressChart(elementId, data.daily_analytics);
            }
        } catch (error) {
            console.error('Error creating progress chart:', error);
        }
    }

    renderProgressChart(elementId, analyticsData) {
        const element = document.getElementById(elementId);
        if (!element) return;

        // Simple chart implementation using CSS
        const maxCompletion = Math.max(...analyticsData.map(d => d.avg_completion || 0));
        
        let chartHTML = '<div style="display: flex; align-items: end; gap: 8px; height: 120px; padding: 10px 0;">';
        
        analyticsData.forEach((day, index) => {
            const completion = day.avg_completion || 0;
            const height = (completion / maxCompletion) * 100;
            const dayName = new Date(day.date).toLocaleDateString('en-US', { weekday: 'short' });
            
            chartHTML += `
                <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                    <div style="
                        width: 100%; 
                        background: linear-gradient(135deg, #667eea, #764ba2); 
                        border-radius: 4px 4px 0 0;
                        height: ${height}%;
                        min-height: 2px;
                        transition: height 0.5s ease;
                    "></div>
                    <div style="font-size: 0.75rem; color: #666; margin-top: 8px;">${dayName}</div>
                    <div style="font-size: 0.7rem; color: #999;">${Math.round(completion)}%</div>
                </div>
            `;
        });
        
        chartHTML += '</div>';
        element.innerHTML = chartHTML;
    }
}

// Habit Gamification System
class HabitGamification {
    constructor() {
        this.points = 0;
        this.level = 1;
        this.badges = [];
        this.loadUserProgress();
    }

    async loadUserProgress() {
        try {
            const response = await fetch('/api/habits/gamification-stats');
            const data = await response.json();

            if (data.success) {
                this.points = data.total_points;
                this.level = data.level;
                this.badges = data.badges;
                this.updateProgressDisplay();
            }
        } catch (error) {
            console.error('Error loading gamification data:', error);
        }
    }

    calculatePointsForCompletion(habit, streakDay) {
        let points = 10; // Base points

        // Difficulty multiplier
        const difficultyMultiplier = {
            'easy': 1,
            'medium': 1.5,
            'hard': 2
        };
        points *= difficultyMultiplier[habit.difficulty_level] || 1;

        // Streak bonus
        if (streakDay > 1) {
            points += Math.min(streakDay * 2, 50); // Max 50 bonus points
        }

        // Perfect completion bonus
        if (habit.completion_percentage >= 100) {
            points += 5;
        }

        return Math.round(points);
    }

    checkLevelUp(newPoints) {
        const newLevel = Math.floor(newPoints / 1000) + 1; // Level up every 1000 points
        
        if (newLevel > this.level) {
            this.level = newLevel;
            this.showLevelUpNotification(newLevel);
            return true;
        }
        return false;
    }

    showLevelUpNotification(level) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #ffd700, #ffed4e);
            color: #7c2d12;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(255, 215, 0, 0.4);
            z-index: 1004;
            text-align: center;
            animation: levelUpPulse 0.6s ease-in-out;
        `;

        notification.innerHTML = `
            <div style="font-size: 3rem; margin-bottom: 15px;">🏆</div>
            <h2 style="margin-bottom: 10px;">Level Up!</h2>
            <p style="font-size: 1.2rem; font-weight: 600;">You've reached Level ${level}!</p>
            <button onclick="this.parentElement.remove()" style="
                margin-top: 20px; 
                padding: 10px 20px; 
                background: #7c2d12; 
                color: #ffd700; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
                font-weight: 600;
            ">Awesome!</button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.parentElement.removeChild(notification);
            }
        }, 5000);
    }

    updateProgressDisplay() {
        // Update level and points display if elements exist
        const levelElement = document.getElementById('userLevel');
        const pointsElement = document.getElementById('userPoints');
        
        if (levelElement) levelElement.textContent = this.level;
        if (pointsElement) pointsElement.textContent = this.points.toLocaleString();
    }
}

// Habit Session Timer with Guided Practice
class HabitSessionManager {
    constructor() {
        this.currentSession = null;
        this.sessionTimer = null;
        this.isSessionActive = false;
    }

    startGuidedSession(habit) {
        this.currentSession = {
            habitId: habit.user_habit_id,
            habitName: habit.name,
            duration: habit.estimated_time_minutes * 60, // Convert to seconds
            startTime: Date.now(),
            steps: this.getHabitSteps(habit.name)
        };

        this.showSessionInterface();
        this.startSessionTimer();
    }

    getHabitSteps(habitName) {
        const habitSteps = {
            '20-20-20 Rule': [
                { step: 1, instruction: 'Set your timer for 20 minutes', duration: 5 },
                { step: 2, instruction: 'Continue your screen work normally', duration: 1195 },
                { step: 3, instruction: 'Look at something 20 feet away', duration: 20 },
                { step: 4, instruction: 'Return to work and reset timer', duration: 5 }
            ],
            'Blinking Exercises': [
                { step: 1, instruction: 'Sit comfortably and relax your face', duration: 10 },
                { step: 2, instruction: 'Blink slowly and deliberately 10 times', duration: 30 },
                { step: 3, instruction: 'Close eyes and hold for 2 seconds', duration: 20 },
                { step: 4, instruction: 'Repeat the cycle 3 times', duration: 120 }
            ],
            'Eye Massage': [
                { step: 1, instruction: 'Wash your hands thoroughly', duration: 20 },
                { step: 2, instruction: 'Gently massage temples in circular motions', duration: 60 },
                { step: 3, instruction: 'Massage around the eye area (avoid direct eye contact)', duration: 90 },
                { step: 4, instruction: 'Finish with gentle pressure on closed eyelids', duration: 30 }
            ],
            'Water Intake': [
                { step: 1, instruction: 'Fill a glass with clean water', duration: 10 },
                { step: 2, instruction: 'Drink slowly and mindfully', duration: 30 },
                { step: 3, instruction: 'Log this glass in your tracker', duration: 10 }
            ]
        };

        return habitSteps[habitName] || [
            { step: 1, instruction: `Practice ${habitName} according to instructions`, duration: 300 }
        ];
    }

    showSessionInterface() {
        // Create session overlay
        const overlay = document.createElement('div');
        overlay.id = 'habitSessionOverlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 2000;
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        overlay.innerHTML = `
            <div style="
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 500px;
                width: 90%;
                text-align: center;
                box-shadow: 0 25px 60px rgba(0, 0, 0, 0.3);
            ">
                <h2 style="margin-bottom: 20px; color: #333;">${this.currentSession.habitName} Session</h2>
                
                <div id="sessionProgress" style="margin-bottom: 30px;">
                    <div style="
                        width: 100px;
                        height: 100px;
                        border-radius: 50%;
                        border: 8px solid #e5e7eb;
                        border-top: 8px solid #667eea;
                        margin: 0 auto 20px;
                        animation: spin 2s linear infinite;
                    "></div>
                    <div id="sessionTimer" style="font-size: 2rem; font-weight: bold; color: #667eea; margin-bottom: 15px;">
                        ${this.formatTime(this.currentSession.duration)}
                    </div>
                    <div id="sessionStep" style="color: #666; font-size: 1.1rem; line-height: 1.6;">
                        Get ready to start your session...
                    </div>
                </div>

                <div style="display: flex; gap: 15px; justify-content: center;">
                    <button onclick="habitSessionManager.pauseSession()" class="btn btn-outline" id="pauseSessionBtn">
                        <i class="fas fa-pause"></i> Pause
                    </button>
                    <button onclick="habitSessionManager.endSession()" class="btn btn-outline">
                        <i class="fas fa-times"></i> End Session
                    </button>
                    <button onclick="habitSessionManager.completeSession()" class="btn btn-success" id="completeSessionBtn" style="display: none;">
                        <i class="fas fa-check"></i> Complete
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);
        this.runSessionSteps();
    }

    async runSessionSteps() {
        const steps = this.currentSession.steps;
        let currentStepIndex = 0;
        let stepTimeRemaining = steps[0].duration;

        this.isSessionActive = true;

        this.sessionTimer = setInterval(() => {
            if (!this.isSessionActive) return;

            stepTimeRemaining--;
            const totalTimeRemaining = this.currentSession.duration - 
                (Date.now() - this.currentSession.startTime) / 1000;

            // Update displays
            document.getElementById('sessionTimer').textContent = this.formatTime(Math.max(0, totalTimeRemaining));
            document.getElementById('sessionStep').textContent = steps[currentStepIndex].instruction;

            // Check if current step is complete
            if (stepTimeRemaining <= 0) {
                currentStepIndex++;
                
                if (currentStepIndex >= steps.length) {
                    // Session complete
                    this.sessionComplete();
                    return;
                }
                
                stepTimeRemaining = steps[currentStepIndex].duration;
                
                // Show step transition notification
                this.showStepTransition(steps[currentStepIndex]);
            }

            // Check if total session time is up
            if (totalTimeRemaining <= 0) {
                this.sessionComplete();
            }
        }, 1000);
    }

    showStepTransition(nextStep) {
        // Brief pause and highlight for step change
        const stepElement = document.getElementById('sessionStep');
        stepElement.style.background = '#f0f4ff';
        stepElement.style.padding = '15px';
        stepElement.style.borderRadius = '10px';
        stepElement.style.border = '2px solid #667eea';

        setTimeout(() => {
            stepElement.style.background = 'none';
            stepElement.style.padding = '0';
            stepElement.style.borderRadius = '0';
            stepElement.style.border = 'none';
        }, 2000);
    }

    pauseSession() {
        this.isSessionActive = !this.isSessionActive;
        const pauseBtn = document.getElementById('pauseSessionBtn');
        
        if (this.isSessionActive) {
            pauseBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
        } else {
            pauseBtn.innerHTML = '<i class="fas fa-play"></i> Resume';
        }
    }

    sessionComplete() {
        clearInterval(this.sessionTimer);
        
        // Show completion interface
        document.getElementById('sessionProgress').innerHTML = `
            <div style="font-size: 4rem; color: #10b981; margin-bottom: 20px;">
                <i class="fas fa-check-circle"></i>
            </div>
            <h3 style="color: #10b981; margin-bottom: 15px;">Session Complete!</h3>
            <p style="color: #666;">Great job completing your ${this.currentSession.habitName} session.</p>
        `;
        
        document.getElementById('completeSessionBtn').style.display = 'inline-flex';
        document.getElementById('pauseSessionBtn').style.display = 'none';
    }

    async completeSession() {
        try {
            // Mark habit as completed
            const response = await fetch('/api/habits/track-progress', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_habit_id: this.currentSession.habitId,
                    completed_count: 1, // Mark as completed
                    target_count: 1
                })
            });

            const data = await response.json();

            if (data.success) {
                this.endSession();
                
                // Show achievements if any
                if (data.achievements && data.achievements.length > 0) {
                    setTimeout(() => {
                        habitGamification.showAchievement(data.achievements[0]);
                    }, 500);
                }
                
                // Reload habits page
                if (typeof loadUserHabits === 'function') {
                    await loadUserHabits();
                }
            }
        } catch (error) {
            console.error('Error completing session:', error);
        }
    }

    endSession() {
        clearInterval(this.sessionTimer);
        this.isSessionActive = false;
        this.currentSession = null;
        
        const overlay = document.getElementById('habitSessionOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(Math.abs(seconds) / 60);
        const secs = Math.floor(Math.abs(seconds) % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
}

// Offline Support
class OfflineHabitManager {
    constructor() {
        this.offlineQueue = [];
        this.isOnline = navigator.onLine;
        this.setupOfflineHandling();
    }

    setupOfflineHandling() {
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.syncOfflineData();
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.showOfflineNotification();
        });
    }

    async logHabitOffline(habitData) {
        // Store in local queue for later sync
        const offlineEntry = {
            ...habitData,
            timestamp: Date.now(),
            id: Date.now() + Math.random()
        };

        this.offlineQueue.push(offlineEntry);
        
        // Store in localStorage for persistence
        try {
            localStorage.setItem('habitOfflineQueue', JSON.stringify(this.offlineQueue));
        } catch (error) {
            console.warn('Could not save offline data:', error);
        }

        this.showOfflineLoggedNotification();
    }

    async syncOfflineData() {
        if (this.offlineQueue.length === 0) return;

        try {
            // Load from localStorage if available
            const stored = localStorage.getItem('habitOfflineQueue');
            if (stored) {
                this.offlineQueue = JSON.parse(stored);
            }

            const syncPromises = this.offlineQueue.map(entry => 
                fetch('/api/habits/track-progress', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(entry)
                })
            );

            await Promise.all(syncPromises);
            
            // Clear offline queue
            this.offlineQueue = [];
            localStorage.removeItem('habitOfflineQueue');
            
            this.showSyncCompleteNotification();
            
            // Reload habits data
            if (typeof loadUserHabits === 'function') {
                await loadUserHabits();
            }
        } catch (error) {
            console.error('Error syncing offline data:', error);
        }
    }

    showOfflineNotification() {
        this.showNotification('You\'re offline. Habit tracking will be saved locally and synced when you\'re back online.', 'warning');
    }

    showOfflineLoggedNotification() {
        this.showNotification('Habit logged offline. Will sync when connection is restored.', 'info');
    }

    showSyncCompleteNotification() {
        this.showNotification('Offline data synced successfully!', 'success');
    }

    showNotification(message, type) {
        // Reuse the notification system from the main habits page
        if (typeof showNotification === 'function') {
            showNotification(message, type);
        }
    }
}

// CSS animations
const additionalCSS = `
    @keyframes levelUpPulse {
        0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
        50% { transform: translate(-50%, -50%) scale(1.1); }
        100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .habit-card.completing {
        animation: completionPulse 0.6s ease-in-out;
    }

    @keyframes completionPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); box-shadow: 0 20px 60px rgba(16, 185, 129, 0.3); }
        100% { transform: scale(1); }
    }
`;

// Initialize all systems when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add additional CSS
    const style = document.createElement('style');
    style.textContent = additionalCSS;
    document.head.appendChild(style);

    // Initialize managers
    window.habitNotificationManager = new HabitNotificationManager();
    window.habitAnalytics = new HabitAnalytics();
    window.habitGamification = new HabitGamification();
    window.habitSessionManager = new HabitSessionManager();
    window.offlineHabitManager = new OfflineHabitManager();
});