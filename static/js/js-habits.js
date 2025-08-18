// Global variables
let userHabits = [];
let availableHabits = [];
let selectedHabits = [];
let timer20_20_20 = null;
let timerSeconds = 1200; // 20 minutes in seconds
let isTimerRunning = false;

// Session timer variables
let sessionTimer = null;
let sessionSeconds = 0;
let isSessionRunning = false;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    loadUserHabits();
    setupCategoryFilters();
    setupTimer();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Close modals when clicking outside
    window.onclick = function(event) {
        const addModal = document.getElementById('addHabitModal');
        const detailModal = document.getElementById('habitDetailModal');
        
        if (event.target === addModal) {
            closeAddHabitModal();
        }
        if (event.target === detailModal) {
            closeHabitDetailModal();
        }
    };

    // Close modals with Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeAddHabitModal();
            closeHabitDetailModal();
            closeBulkDeleteModal();
            closeDeleteConfirmModal();
        }
    });

    // Close habit menus when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.habit-menu')) {
            const allMenus = document.querySelectorAll('.habit-menu-dropdown');
            allMenus.forEach(menu => menu.classList.remove('show'));
        }
    });

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

// Load user habits
async function loadUserHabits() {
    try {
        document.getElementById('loadingState').style.display = 'block';
        document.getElementById('habitsGrid').style.display = 'none';
        document.getElementById('emptyState').style.display = 'none';

        const response = await fetch('/api/habits/user-habits');
        const data = await response.json();

        if (data.success) {
            userHabits = data.habits;
            updateStats(data.weekly_stats);
            
            if (userHabits.length > 0) {
                displayHabits(userHabits);
                updateTodayGoals(userHabits);
                updateStreakInfo(userHabits);
                document.getElementById('habitsGrid').style.display = 'grid';
            } else {
                document.getElementById('emptyState').style.display = 'block';
            }
        } else {
            console.error('Failed to load habits:', data.error);
            document.getElementById('emptyState').style.display = 'block';
        }

        document.getElementById('loadingState').style.display = 'none';
    } catch (error) {
        console.error('Error loading habits:', error);
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('emptyState').style.display = 'block';
    }
}

// Update statistics display
function updateStats(stats) {
    document.getElementById('todayCompleted').textContent = stats.completed_today || 0;
    document.getElementById('weeklyAverage').textContent = Math.round(stats.avg_completion || 0) + '%';
    document.getElementById('totalHabits').textContent = stats.active_habits || 0;
    
    // Calculate longest streak from habits
    const longestStreak = Math.max(...userHabits.map(h => h.streak_days || 0), 0);
    document.getElementById('longestStreak').textContent = longestStreak;
}

// Display habits in the grid
function displayHabits(habits) {
    const grid = document.getElementById('habitsGrid');
    grid.innerHTML = '';

    habits.forEach(habit => {
        const habitCard = createHabitCard(habit);
        grid.appendChild(habitCard);
    });
}

// Create individual habit card
function createHabitCard(habit) {
    const card = document.createElement('div');
    const isCompleted = habit.is_completed;
    const completionPercentage = habit.completion_percentage || 0;
    
    card.className = `habit-card ${isCompleted ? 'completed' : (completionPercentage > 0 ? 'in-progress' : '')}`;
    card.dataset.category = habit.category;

    card.innerHTML = `
        <div class="habit-header">
            <div class="habit-icon">
                <i class="${habit.icon || 'fas fa-eye'}"></i>
            </div>
            <div class="habit-info">
                <h3>${habit.name}</h3>
                <div class="habit-meta">
                    <span><i class="fas fa-clock"></i> ${habit.estimated_time_minutes}min</span>
                    <span><i class="fas fa-signal"></i> ${habit.difficulty_level}</span>
                    <span><i class="fas fa-fire"></i> ${habit.streak_days || 0} days</span>
                </div>
            </div>
            <div class="habit-menu">
                <button class="habit-menu-btn" onclick="toggleHabitMenu(${habit.user_habit_id})">
                    <i class="fas fa-ellipsis-v"></i>
                </button>
                <div class="habit-menu-dropdown" id="habitMenu_${habit.user_habit_id}">
                    <button onclick="showHabitSettings(${habit.user_habit_id})" class="menu-item">
                        <i class="fas fa-cog"></i> Settings
                    </button>
                    <button onclick="confirmDeleteHabit(${habit.user_habit_id}, '${habit.name.replace(/'/g, "\\'")}')" class="menu-item delete">
                        <i class="fas fa-trash"></i> Remove Habit
                    </button>
                </div>
            </div>
        </div>
        
        <div class="habit-description">
            ${habit.description}
        </div>
        
        <div class="habit-progress">
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${completionPercentage}%"></div>
            </div>
            <div class="progress-text">
                <span>${habit.today_completed || 0}/${habit.target_count} ${habit.target_unit}</span>
                <span>${Math.round(completionPercentage)}%</span>
            </div>
        </div>

        <div class="habit-streak-visual">
            ${generateStreakDots(habit.streak_days || 0)}
        </div>
        
        <div class="habit-actions">
            ${!isCompleted ? `
                <button class="quick-action-btn" onclick="quickIncrement(${habit.user_habit_id})" title="Quick +1">
                    <i class="fas fa-plus"></i>
                </button>
                <button class="btn btn-primary" onclick="startHabitSession(${habit.user_habit_id})">
                    <i class="fas fa-play"></i>
                    Start Session
                </button>
            ` : `
                <button class="btn btn-success" disabled>
                    <i class="fas fa-check"></i>
                    Completed
                </button>
            `}
            <button class="btn btn-outline" onclick="showHabitDetails(${habit.habit_id})">
                <i class="fas fa-info-circle"></i>
                Details
            </button>
        </div>
    `;

    return card;
}

// Generate streak dots visualization
function generateStreakDots(streakDays) {
    const maxDots = 7; // Show last 7 days
    let dots = '';
    
    for (let i = maxDots - 1; i >= 0; i--) {
        const isToday = i === 0;
        const isCompleted = i < streakDays;
        const classes = `streak-dot ${isCompleted ? 'completed' : ''} ${isToday ? 'today' : ''}`;
        dots += `<div class="${classes}"></div>`;
    }
    
    return dots;
}

// Quick increment habit progress
async function quickIncrement(userHabitId) {
    try {
        const response = await fetch('/api/habits/quick-log', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_habit_id: userHabitId,
                increment: 1
            })
        });

        const data = await response.json();

        if (data.success) {
            // Show achievement if any
            if (data.achievements && data.achievements.length > 0) {
                showAchievement(data.achievements[0]);
            }
            
            // Reload habits to show updated progress
            await loadUserHabits();
        } else {
            showNotification('Failed to update progress: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error updating habit:', error);
        showNotification('Network error. Please try again.', 'error');
    }
}

// Start habit session
async function startHabitSession(userHabitId) {
    // Find the habit
    const habit = userHabits.find(h => h.user_habit_id === userHabitId);
    if (!habit) return;

    // Show session modal or start guided session
    const sessionData = {
        habitName: habit.name,
        instructions: habit.instructions,
        estimatedTime: habit.estimated_time_minutes,
        userHabitId: userHabitId
    };

    showHabitSessionModal(sessionData);
}

// Show habit session modal
function showHabitSessionModal(sessionData) {
    // Create session modal content
    const modalContent = `
        <div style="text-align: center;">
            <h3>${sessionData.habitName} Session</h3>
            <p style="color: #666; margin-bottom: 20px;">Estimated time: ${sessionData.estimatedTime} minutes</p>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 25px; text-align: left;">
                <h4 style="margin-bottom: 15px; color: #333;">Instructions:</h4>
                <p style="line-height: 1.6; color: #666;">${sessionData.instructions}</p>
            </div>

            <div style="background: linear-gradient(135deg, #f0f4ff, #e0e7ff); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
                <div style="font-size: 2rem; font-weight: bold; color: #667eea; margin-bottom: 10px;" id="sessionTimer">
                    ${sessionData.estimatedTime}:00
                </div>
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <button class="timer-btn play" onclick="startSessionTimer(${sessionData.estimatedTime * 60})">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="timer-btn pause" onclick="pauseSessionTimer()" style="display: none;" id="sessionPause">
                        <i class="fas fa-pause"></i>
                    </button>
                    <button class="timer-btn reset" onclick="resetSessionTimer(${sessionData.estimatedTime * 60})">
                        <i class="fas fa-redo"></i>
                    </button>
                </div>
            </div>

            <div style="display: flex; gap: 15px; justify-content: center;">
                <button class="btn btn-outline" onclick="closeHabitDetailModal()">
                    Cancel
                </button>
                <button class="btn btn-success" onclick="completeHabitSession(${sessionData.userHabitId})">
                    <i class="fas fa-check"></i>
                    Mark Complete
                </button>
            </div>
        </div>
    `;

    document.getElementById('habitDetailContent').innerHTML = modalContent;
    document.getElementById('habitDetailModal').style.display = 'block';
}

// Complete habit session
async function completeHabitSession(userHabitId) {
    try {
        const habit = userHabits.find(h => h.user_habit_id === userHabitId);
        
        const response = await fetch('/api/habits/track-progress', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_habit_id: userHabitId,
                completed_count: habit.target_count, // Mark as fully completed
                target_count: habit.target_count
            })
        });

        const data = await response.json();

        if (data.success) {
            closeHabitDetailModal();
            
            if (data.achievements && data.achievements.length > 0) {
                showAchievement(data.achievements[0]);
            }
            
            await loadUserHabits();
            showNotification('Session completed successfully!', 'success');
        } else {
            showNotification('Failed to complete session: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error completing session:', error);
        showNotification('Network error. Please try again.', 'error');
    }
}

// Update today's goals in sidebar
function updateTodayGoals(habits) {
    const goalsContainer = document.getElementById('todayGoals');
    goalsContainer.innerHTML = '';

    habits.forEach(habit => {
        const goalItem = document.createElement('li');
        goalItem.className = 'goal-item';
        
        const isCompleted = habit.is_completed;
        const progress = `${habit.today_completed || 0}/${habit.target_count}`;
        
        goalItem.innerHTML = `
            <div class="goal-checkbox ${isCompleted ? 'checked' : ''}" onclick="quickIncrement(${habit.user_habit_id})">
                ${isCompleted ? '<i class="fas fa-check"></i>' : ''}
            </div>
            <div class="goal-text">${habit.name}</div>
            <div class="goal-progress">${progress}</div>
        `;
        
        goalsContainer.appendChild(goalItem);
    });
}

// Update streak information in sidebar
function updateStreakInfo(habits) {
    const streakContainer = document.getElementById('streakCalendar');
    streakContainer.innerHTML = '';

    const topStreaks = habits
        .filter(h => h.streak_days > 0)
        .sort((a, b) => b.streak_days - a.streak_days)
        .slice(0, 3);

    if (topStreaks.length === 0) {
        streakContainer.innerHTML = '<p style="color: #666; text-align: center; font-style: italic;">No active streaks yet. Start your first habit!</p>';
        return;
    }

    topStreaks.forEach(habit => {
        const streakItem = document.createElement('div');
        streakItem.style.cssText = 'margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #f3f4f6;';
        
        streakItem.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #333;">${habit.name}</span>
                <span style="background: #fff5b4; color: #92400e; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    ${habit.streak_days} days
                </span>
            </div>
            <div class="habit-streak-visual">
                ${generateStreakDots(habit.streak_days)}
            </div>
        `;
        
        streakContainer.appendChild(streakItem);
    });
}

// Show add habit modal
async function showAddHabitModal() {
    try {
        document.getElementById('addHabitModal').style.display = 'block';
        document.getElementById('habitBrowserLoading').style.display = 'block';
        document.getElementById('habitBrowser').innerHTML = '';
        document.getElementById('habitBrowserError').style.display = 'none';
        
        // Reset selected habits
        selectedHabits = [];
        updateSelectedCount();
        
        // Load available habits
        const response = await fetch('/api/habits/available');
        const data = await response.json();

        document.getElementById('habitBrowserLoading').style.display = 'none';

        if (data.success) {
            console.log('Habits API response:', data); // Debug log
            availableHabits = data.habits;
            displayAvailableHabits(data.habits);
        } else {
            document.getElementById('habitBrowserError').style.display = 'block';
            console.error('Failed to load available habits:', data.error);
        }
    } catch (error) {
        console.error('Error loading available habits:', error);
        document.getElementById('habitBrowserLoading').style.display = 'none';
        document.getElementById('habitBrowserError').style.display = 'block';
    }
}

// Display available habits in browser
function displayAvailableHabits(habits) {
    const browser = document.getElementById('habitBrowser');
    browser.innerHTML = '';

    console.log('Available habits data:', habits); // Debug log

    const unselectedHabits = habits.filter(habit => !habit.is_selected);

    if (unselectedHabits.length === 0) {
        browser.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: #666;">
                <i class="fas fa-check-circle" style="font-size: 3rem; margin-bottom: 15px; color: #10b981;"></i>
                <h3>All Available Habits Added!</h3>
                <p>You've added all available eye health habits to your routine.</p>
            </div>
        `;
        return;
    }

    unselectedHabits.forEach(habit => {
        console.log('Processing habit:', habit); // Debug log
        
        const habitDiv = document.createElement('div');
        habitDiv.className = 'available-habit';
        habitDiv.dataset.habitId = habit.id;
        
        // Handle different data access methods - sometimes it's an array, sometimes an object
        const habitData = {
            id: habit.id || habit[0],
            name: habit.name || habit[1],
            description: habit.description || habit[2], 
            category: habit.category || habit[3],
            icon: habit.icon || habit[4],
            target_frequency: habit.target_frequency || habit[5],
            target_count: habit.target_count || habit[6],
            target_unit: habit.target_unit || habit[7],
            estimated_time_minutes: habit.estimated_time_minutes || habit[11],
            is_selected: habit.is_selected || habit[habit.length - 1] // Usually last field
        };
        
        habitDiv.innerHTML = `
            <div style="margin-bottom: 15px;">
                <i class="${habitData.icon || 'fas fa-eye'}" style="font-size: 2rem; color: #667eea; margin-bottom: 10px;"></i>
                <h4 style="color: #333; margin-bottom: 8px;">${habitData.name || 'Unknown Habit'}</h4>
                <p style="color: #666; font-size: 0.85rem; line-height: 1.4;">${habitData.description || 'No description available'}</p>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #888;">
                <span><i class="fas fa-clock"></i> ${habitData.estimated_time_minutes || 5}min</span>
                <span><i class="fas fa-target"></i> ${habitData.target_count || 1} ${habitData.target_unit || 'times'}</span>
            </div>
        `;

        habitDiv.onclick = () => toggleHabitSelection(habitData.id, habitDiv);
        browser.appendChild(habitDiv);
    });
}

// Toggle habit selection
function toggleHabitSelection(habitId, element) {
    const index = selectedHabits.indexOf(habitId);
    
    if (index === -1) {
        selectedHabits.push(habitId);
        element.classList.add('selected');
    } else {
        selectedHabits.splice(index, 1);
        element.classList.remove('selected');
    }

    updateSelectedCount();
}

// Update selected habits count
function updateSelectedCount() {
    document.getElementById('selectedCount').textContent = selectedHabits.length;
    document.getElementById('addHabitsBtn').disabled = selectedHabits.length === 0;
}

// Add selected habits
async function addSelectedHabits() {
    if (selectedHabits.length === 0) {
        showNotification('Please select at least one habit to add.', 'error');
        return;
    }

    try {
        // Disable button to prevent double-clicking
        const addBtn = document.getElementById('addHabitsBtn');
        const originalText = addBtn.innerHTML;
        addBtn.disabled = true;
        addBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding Habits...';

        let successCount = 0;
        let errors = [];

        // Add habits one by one to better handle errors
        for (const habitId of selectedHabits) {
            try {
                const response = await fetch('/api/habits/add-habit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        habit_id: habitId
                    })
                });

                const data = await response.json();

                if (data.success) {
                    successCount++;
                } else {
                    const habitName = availableHabits.find(h => h.id === habitId)?.name || `Habit ${habitId}`;
                    errors.push(`${habitName}: ${data.error}`);
                }
            } catch (error) {
                const habitName = availableHabits.find(h => h.id === habitId)?.name || `Habit ${habitId}`;
                errors.push(`${habitName}: Network error`);
            }
        }

        // Reset button first (before any async operations)
        addBtn.disabled = false;
        addBtn.innerHTML = originalText;

        // Show results
        if (successCount > 0) {
            showNotification(`${successCount} habit${successCount > 1 ? 's' : ''} added successfully!`, 'success');
            closeAddHabitModal();
            await loadUserHabits();
        }

        if (errors.length > 0) {
            console.error('Errors adding habits:', errors);
            showNotification(`Some habits could not be added. Check console for details.`, 'error');
        }

        // Only show the generic error if there were no successes at all
        if (successCount === 0) {
            showNotification('Failed to add habits. Please try again.', 'error');
        }

    } catch (error) {
        console.error('Error adding habits:', error);
        
        // Reset button in case of outer try-catch error
        const addBtn = document.getElementById('addHabitsBtn');
        addBtn.disabled = false;
        addBtn.innerHTML = '<i class="fas fa-check"></i> Add Selected Habits (<span id="selectedCount">0</span>)';
        
        showNotification('Failed to add habits. Please try again.', 'error');
    }
}

// Close add habit modal
function closeAddHabitModal() {
    document.getElementById('addHabitModal').style.display = 'none';
    selectedHabits = [];
    updateSelectedCount();
}

// Close habit detail modal
function closeHabitDetailModal() {
    document.getElementById('habitDetailModal').style.display = 'none';
}

// Show habit details
async function showHabitDetails(habitId) {
    try {
        const response = await fetch(`/api/habits/habit-details/${habitId}`);
        const data = await response.json();

        console.log('Habit details response:', data); // Debug log

        if (data.success) {
            displayHabitDetails(data.habit, data.recent_progress);
            document.getElementById('habitDetailModal').style.display = 'block';
        } else {
            showNotification('Failed to load habit details: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error loading habit details:', error);
        showNotification('Network error loading habit details.', 'error');
    }
}

// Display habit details in modal
function displayHabitDetails(habit, recentProgress) {
    console.log('Displaying habit details:', habit); // Debug log
    
    // Handle both array and object data formats
    const habitData = {
        id: habit.id || habit[0],
        name: habit.name || habit[1] || 'Unknown Habit',
        description: habit.description || habit[2] || 'No description available',
        category: habit.category || habit[3],
        icon: habit.icon || habit[4] || 'fas fa-eye',
        target_frequency: habit.target_frequency || habit[5],
        target_count: habit.target_count || habit[6] || 1,
        target_unit: habit.target_unit || habit[7] || 'times',
        instructions: habit.instructions || habit[8] || 'Follow the general guidance for this habit.',
        benefits: habit.benefits || habit[9] || 'Supports overall eye health and wellness.',
        difficulty_level: habit.difficulty_level || habit[10] || 'easy',
        estimated_time_minutes: habit.estimated_time_minutes || habit[11] || 5,
        user_habit_id: habit.user_habit_id || habit[12],
        custom_target_count: habit.custom_target_count || habit[13],
        reminder_time: habit.reminder_time || habit[14],
        start_date: habit.start_date || habit[15],
        avg_completion_30d: habit.avg_completion_30d || habit[16] || 0,
        completed_days_30d: habit.completed_days_30d || habit[17] || 0,
        best_streak: habit.best_streak || habit[18] || 0
    };

    document.getElementById('habitDetailTitle').textContent = habitData.name;
    
    const content = `
        <div style="text-align: center; margin-bottom: 25px;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #667eea, #764ba2); color: white; display: flex; align-items: center; justify-content: center; font-size: 2rem; margin: 0 auto 15px;">
                <i class="${habitData.icon}"></i>
            </div>
            <h3 style="color: #333; margin-bottom: 10px;">${habitData.name}</h3>
            <p style="color: #666;">${habitData.description}</p>
        </div>

        <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <h4 style="color: #333; margin-bottom: 15px;">Instructions:</h4>
            <p style="line-height: 1.6; color: #666;">${habitData.instructions}</p>
        </div>

        <div style="background: #e0f2fe; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <h4 style="color: #0277bd; margin-bottom: 15px;"><i class="fas fa-lightbulb"></i> Benefits:</h4>
            <p style="line-height: 1.6; color: #01579b;">${habitData.benefits}</p>
        </div>

        <div style="background: #f0f4ff; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <h4 style="color: #333; margin-bottom: 15px;"><i class="fas fa-info-circle"></i> Habit Details:</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem;">
                <div><strong>Difficulty:</strong> ${habitData.difficulty_level}</div>
                <div><strong>Time:</strong> ${habitData.estimated_time_minutes} minutes</div>
                <div><strong>Target:</strong> ${habitData.target_count} ${habitData.target_unit}</div>
                <div><strong>Category:</strong> ${habitData.category?.replace('_', ' ') || 'General'}</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 25px;">
            <div style="text-align: center; padding: 15px; background: #f0f4ff; border-radius: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">${Math.round(habitData.avg_completion_30d) || 0}%</div>
                <div style="font-size: 0.85rem; color: #666;">30-Day Average</div>
            </div>
            <div style="text-align: center; padding: 15px; background: #fff5f5; border-radius: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #ef4444;">${habitData.best_streak || 0}</div>
                <div style="font-size: 0.85rem; color: #666;">Best Streak</div>
            </div>
            <div style="text-align: center; padding: 15px; background: #f0fdf4; border-radius: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">${habitData.completed_days_30d || 0}</div>
                <div style="font-size: 0.85rem; color: #666;">Days Completed</div>
            </div>
        </div>

        <div style="text-align: center;">
            <button class="btn btn-outline" onclick="closeHabitDetailModal()">
                <i class="fas fa-times"></i>
                Close
            </button>
        </div>
    `;

    document.getElementById('habitDetailContent').innerHTML = content;
}

// Setup category filters
function setupCategoryFilters() {
    const filters = document.querySelectorAll('.category-filter');
    filters.forEach(filter => {
        filter.addEventListener('click', function() {
            // Update active filter
            filters.forEach(f => f.classList.remove('active'));
            this.classList.add('active');

            // Filter habits
            const category = this.dataset.category;
            filterHabitsByCategory(category);
        });
    });
}

// Filter habits by category
function filterHabitsByCategory(category) {
    const habitCards = document.querySelectorAll('.habit-card');
    
    habitCards.forEach(card => {
        if (category === 'all' || card.dataset.category === category) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

// 20-20-20 Timer functionality
function setupTimer() {
    updateTimerDisplay();
}

function startTimer() {
    if (isTimerRunning) return;
    
    isTimerRunning = true;
    document.getElementById('timerPlay').style.display = 'none';
    document.getElementById('timerPause').style.display = 'flex';

    timer20_20_20 = setInterval(() => {
        timerSeconds--;
        updateTimerDisplay();

        if (timerSeconds <= 0) {
            timerAlert();
            resetTimer();
        }
    }, 1000);
}

function pauseTimer() {
    isTimerRunning = false;
    clearInterval(timer20_20_20);
    document.getElementById('timerPlay').style.display = 'flex';
    document.getElementById('timerPause').style.display = 'none';
}

function resetTimer() {
    isTimerRunning = false;
    clearInterval(timer20_20_20);
    timerSeconds = 1200; // 20 minutes
    updateTimerDisplay();
    document.getElementById('timerPlay').style.display = 'flex';
    document.getElementById('timerPause').style.display = 'none';
}

function updateTimerDisplay() {
    const minutes = Math.floor(timerSeconds / 60);
    const seconds = timerSeconds % 60;
    const display = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    document.getElementById('timerDisplay').textContent = display;
    document.getElementById('nextBreak').textContent = display;
}

function timerAlert() {
    // Show break reminder
    showNotification('Time for a 20-second break! Look at something 20 feet away.', 'info');
    
    // Play notification sound (if available)
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('BlinkWell - Time for your eye break!', {
            body: 'Look at something 20 feet away for 20 seconds.',
            icon: '/static/img/logo.png'
        });
    }
}

// Session timer functions
function startSessionTimer(duration) {
    if (isSessionRunning) return;
    
    sessionSeconds = duration;
    isSessionRunning = true;
    
    const playBtn = document.querySelector('#sessionTimer').parentElement.querySelector('.play');
    const pauseBtn = document.getElementById('sessionPause');
    
    if (playBtn) playBtn.style.display = 'none';
    if (pauseBtn) pauseBtn.style.display = 'flex';

    sessionTimer = setInterval(() => {
        sessionSeconds--;
        updateSessionTimerDisplay();

        if (sessionSeconds <= 0) {
            sessionTimerComplete();
        }
    }, 1000);
}

function pauseSessionTimer() {
    isSessionRunning = false;
    clearInterval(sessionTimer);
    
    const playBtn = document.querySelector('#sessionTimer').parentElement.querySelector('.play');
    const pauseBtn = document.getElementById('sessionPause');
    
    if (playBtn) playBtn.style.display = 'flex';
    if (pauseBtn) pauseBtn.style.display = 'none';
}

function resetSessionTimer(duration) {
    isSessionRunning = false;
    clearInterval(sessionTimer);
    sessionSeconds = duration;
    updateSessionTimerDisplay();
    
    const playBtn = document.querySelector('#sessionTimer').parentElement.querySelector('.play');
    const pauseBtn = document.getElementById('sessionPause');
    
    if (playBtn) playBtn.style.display = 'flex';
    if (pauseBtn) pauseBtn.style.display = 'none';
}

function updateSessionTimerDisplay() {
    const minutes = Math.floor(sessionSeconds / 60);
    const seconds = sessionSeconds % 60;
    const display = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const timerElement = document.getElementById('sessionTimer');
    if (timerElement) {
        timerElement.textContent = display;
    }
}

function sessionTimerComplete() {
    resetSessionTimer(sessionSeconds);
    showNotification('Session complete! Great job on your eye health practice.', 'success');
}

// Habit menu functions
function toggleHabitMenu(userHabitId) {
    const menu = document.getElementById(`habitMenu_${userHabitId}`);
    const allMenus = document.querySelectorAll('.habit-menu-dropdown');
    
    // Close all other menus
    allMenus.forEach(m => {
        if (m.id !== `habitMenu_${userHabitId}`) {
            m.classList.remove('show');
        }
    });
    
    // Toggle current menu
    menu.classList.toggle('show');
}

// Show habit settings (placeholder for future implementation)
function showHabitSettings(userHabitId) {
    const habit = userHabits.find(h => h.user_habit_id === userHabitId);
    if (!habit) return;
    
    // Close the menu
    const menu = document.getElementById(`habitMenu_${userHabitId}`);
    if (menu) menu.classList.remove('show');
    
    // For now, just show a placeholder
    showNotification('Habit settings coming soon!', 'info');
}

// Delete habit confirmation
function confirmDeleteHabit(userHabitId, habitName) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.id = 'deleteConfirmModal';
    modal.style.display = 'block';
    
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 480px;">
            <div class="modal-header">
                <h2 style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> Remove Habit</h2>
                <button class="close-btn" onclick="closeDeleteConfirmModal()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div style="text-align: center; padding: 20px 0;">
                    <div style="font-size: 4rem; color: #fbbf24; margin-bottom: 20px;">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <h3 style="margin-bottom: 15px;">Remove "${habitName}" from your routine?</h3>
                    <p style="color: #666; line-height: 1.6; margin-bottom: 25px;">
                        This will remove the habit from your active routine, but your progress history will be preserved. 
                        You can add this habit back anytime from the habit library.
                    </p>
                    
                    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin-bottom: 25px;">
                        <div style="display: flex; align-items: center; gap: 10px; color: #92400e;">
                            <i class="fas fa-info-circle"></i>
                            <span style="font-weight: 600;">Your streak and progress data will be kept</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 15px; justify-content: center;">
                        <button class="btn btn-outline" onclick="closeDeleteConfirmModal()">
                            <i class="fas fa-times"></i>
                            Cancel
                        </button>
                        <button class="btn btn-danger" onclick="deleteHabit(${userHabitId})" id="confirmDeleteBtn">
                            <i class="fas fa-trash"></i>
                            Remove Habit
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close menu dropdown
    const menu = document.getElementById(`habitMenu_${userHabitId}`);
    if (menu) menu.classList.remove('show');
}

function closeDeleteConfirmModal() {
    const modal = document.getElementById('deleteConfirmModal');
    if (modal) {
        document.body.removeChild(modal);
    }
}

// Delete habit function
async function deleteHabit(userHabitId) {
    try {
        const confirmBtn = document.getElementById('confirmDeleteBtn');
        const originalText = confirmBtn.innerHTML;
        
        // Show loading state
        confirmBtn.disabled = true;
        confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Removing...';
        
        const response = await fetch('/api/habits/remove-habit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_habit_id: userHabitId
            })
        });

        const data = await response.json();

        if (data.success) {
            closeDeleteConfirmModal();
            showNotification('Habit removed successfully!', 'success');
            
            // Add fade out animation to the habit card
            const habitCards = document.querySelectorAll('.habit-card');
            const targetCard = Array.from(habitCards).find(card => {
                const actionsDiv = card.querySelector('.habit-actions');
                return actionsDiv && actionsDiv.innerHTML.includes(`quickIncrement(${userHabitId})`);
            });
            
            if (targetCard) {
                targetCard.style.transition = 'all 0.3s ease-out';
                targetCard.style.transform = 'scale(0.95)';
                targetCard.style.opacity = '0.5';
                
                setTimeout(() => {
                    loadUserHabits(); // Reload the habits
                }, 300);
            } else {
                loadUserHabits(); // Reload immediately if card not found
            }
        } else {
            // Reset button on error
            confirmBtn.disabled = false;
            confirmBtn.innerHTML = originalText;
            showNotification('Failed to remove habit: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error removing habit:', error);
        
        // Reset button on error
        const confirmBtn = document.getElementById('confirmDeleteBtn');
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.innerHTML = '<i class="fas fa-trash"></i> Remove Habit';
        }
        
        showNotification('Network error. Please try again.', 'error');
    }
}

// Bulk delete functionality
function showBulkDeleteOptions() {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.id = 'bulkDeleteModal';
    modal.style.display = 'block';
    
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-trash-alt"></i> Manage Habits</h2>
                <button class="close-btn" onclick="closeBulkDeleteModal()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <p style="color: #666; margin-bottom: 20px;">Select habits to remove from your routine:</p>
                
                <div id="bulkDeleteList" class="bulk-delete-list">
                    <!-- Habits will be populated here -->
                </div>
                
                <div style="text-align: center; margin-top: 25px;">
                    <button class="btn btn-outline" onclick="closeBulkDeleteModal()">
                        Cancel
                    </button>
                    <button class="btn btn-danger" onclick="bulkDeleteHabits()" id="bulkDeleteBtn" disabled>
                        <i class="fas fa-trash"></i>
                        Remove Selected (<span id="bulkSelectedCount">0</span>)
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    populateBulkDeleteList();
}

function populateBulkDeleteList() {
    const list = document.getElementById('bulkDeleteList');
    if (!list) return;
    
    list.innerHTML = '';
    
    userHabits.forEach(habit => {
        const item = document.createElement('div');
        item.className = 'bulk-delete-item';
        
        item.innerHTML = `
            <label style="display: flex; align-items: center; gap: 15px; padding: 15px; cursor: pointer;">
                <input type="checkbox" value="${habit.user_habit_id}" onchange="updateBulkDeleteCount()">
                <div class="habit-icon" style="width: 40px; height: 40px; font-size: 1.2rem;">
                    <i class="${habit.icon || 'fas fa-eye'}"></i>
                </div>
                <div>
                    <div style="font-weight: 600; color: #333;">${habit.name}</div>
                    <div style="font-size: 0.85rem; color: #666;">
                        ${habit.streak_days || 0} day streak • ${Math.round(habit.completion_percentage || 0)}% today
                    </div>
                </div>
            </label>
        `;
        
        list.appendChild(item);
    });
}

function updateBulkDeleteCount() {
    const checkboxes = document.querySelectorAll('#bulkDeleteList input[type="checkbox"]:checked');
    const count = checkboxes.length;
    
    document.getElementById('bulkSelectedCount').textContent = count;
    document.getElementById('bulkDeleteBtn').disabled = count === 0;
}

function closeBulkDeleteModal() {
    const modal = document.getElementById('bulkDeleteModal');
    if (modal) {
        document.body.removeChild(modal);
    }
}

async function bulkDeleteHabits() {
    const checkboxes = document.querySelectorAll('#bulkDeleteList input[type="checkbox"]:checked');
    const userHabitIds = Array.from(checkboxes).map(cb => parseInt(cb.value));
    
    if (userHabitIds.length === 0) return;
    
    const confirmMsg = `Are you sure you want to remove ${userHabitIds.length} habit${userHabitIds.length > 1 ? 's' : ''} from your routine?`;
    
    if (!confirm(confirmMsg)) return;
    
    try {
        const deleteBtn = document.getElementById('bulkDeleteBtn');
        deleteBtn.disabled = true;
        deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Removing...';
        
        let successCount = 0;
        let errors = [];
        
        for (const userHabitId of userHabitIds) {
            try {
                const response = await fetch('/api/habits/remove-habit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        user_habit_id: userHabitId
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    successCount++;
                } else {
                    errors.push(data.error);
                }
            } catch (error) {
                errors.push('Network error');
            }
        }
        
        closeBulkDeleteModal();
        
        if (successCount > 0) {
            showNotification(`${successCount} habit${successCount > 1 ? 's' : ''} removed successfully!`, 'success');
            await loadUserHabits();
        }
        
        if (errors.length > 0) {
            showNotification(`Some habits could not be removed. Please try again.`, 'error');
        }
        
    } catch (error) {
        console.error('Error in bulk delete:', error);
        showNotification('Failed to remove habits. Please try again.', 'error');
    }
}

// Achievement display
function showAchievement(achievement) {
    const toast = document.getElementById('achievementToast');
    document.getElementById('achievementTitle').textContent = achievement.name;
    document.getElementById('achievementDesc').textContent = achievement.description;
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 5000);
}

// Notification system
function showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 1002;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 300px;
        word-wrap: break-word;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Auto-refresh every 5 minutes
setInterval(loadUserHabits, 5 * 60 * 1000);