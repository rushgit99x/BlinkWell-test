// Toggle sidebar
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');
    
    if (window.innerWidth > 1024) {
        sidebar.classList.toggle('collapsed');
        mainContent.classList.toggle('expanded');
    } else {
        sidebar.classList.toggle('active');
    }
}

// Handle responsive behavior
window.addEventListener('resize', function() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');
    
    if (window.innerWidth > 1024) {
        sidebar.classList.remove('active');
        if (sidebar.classList.contains('collapsed')) {
            mainContent.classList.add('expanded');
        } else {
            mainContent.classList.remove('expanded');
        }
    }
});

// Initialize dynamic progress bars
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill[data-width]');
    progressBars.forEach(bar => {
        const width = bar.getAttribute('data-width');
        if (width) {
            // Set initial width to 0
            bar.style.width = '0%';
            // Animate to target width
            setTimeout(() => {
                bar.style.width = width + '%';
            }, 500);
        }
    });
}

// Animate progress bars on load
window.addEventListener('load', function() {
    // Handle traditional progress bars with inline styles
    const traditionalProgressBars = document.querySelectorAll('.progress-fill:not([data-width])');
    traditionalProgressBars.forEach(bar => {
        const width = bar.style.width;
        if (width) {
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 500);
        }
    });
    
    // Handle dynamic progress bars with data attributes
    initializeProgressBars();
});

// Simulate habit completion
document.querySelectorAll('.habit-day.today').forEach(day => {
    day.addEventListener('click', function() {
        this.classList.remove('today');
        this.classList.add('completed');
    });
});

// Add entrance animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.fade-in-up').forEach(item => {
    item.style.opacity = '0';
    item.style.transform = 'translateY(20px)';
    item.style.transition = 'all 0.6s ease-out';
    observer.observe(item);
});

// Update progress bars when data changes (for dynamic updates)
function updateProgressBar(selector, newWidth) {
    const bar = document.querySelector(selector);
    if (bar) {
        bar.setAttribute('data-width', newWidth);
        bar.style.width = newWidth + '%';
    }
}

// Export function for external use
window.updateProgressBar = updateProgressBar;

// Logout Popup Functions
function showLogoutPopup() {
    const modal = document.getElementById('logoutModal');
    modal.style.display = 'flex';
    
    // Trigger animation after a brief delay
    setTimeout(() => {
        modal.classList.add('show');
    }, 10);
    
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
    
    // Add escape key listener
    document.addEventListener('keydown', handleEscapeKey);
    
    // Add click outside to close
    modal.addEventListener('click', handleOutsideClick);
}

function hideLogoutPopup() {
    const modal = document.getElementById('logoutModal');
    modal.classList.remove('show');
    
    // Wait for animation to complete before hiding
    setTimeout(() => {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }, 300);
    
    // Remove event listeners
    document.removeEventListener('keydown', handleEscapeKey);
    modal.removeEventListener('click', handleOutsideClick);
}

function confirmLogout() {
    // Show loading state
    const logoutBtn = document.querySelector('.btn-logout');
    const originalText = logoutBtn.innerHTML;
    logoutBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging out...';
    logoutBtn.disabled = true;
    
    // Redirect to logout route
    setTimeout(() => {
        window.location.href = '/logout';
    }, 500);
}

function handleEscapeKey(event) {
    if (event.key === 'Escape') {
        hideLogoutPopup();
    }
}

function handleOutsideClick(event) {
    if (event.target === event.currentTarget) {
        hideLogoutPopup();
    }
}

// Close modal when clicking outside
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('logoutModal');
    if (modal) {
        modal.addEventListener('click', function(event) {
            if (event.target === modal) {
                hideLogoutPopup();
            }
        });
    }
});