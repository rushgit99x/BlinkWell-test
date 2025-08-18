let currentRecommendations = {};
let userStats = {};
let debugMode = false;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Add debug mode toggle (double-click header to enable)
    document.querySelector('.header h1').addEventListener('dblclick', function() {
        debugMode = !debugMode;
        document.getElementById('debugButton').style.display = debugMode ? 'block' : 'none';
        if (debugMode) {
            loadDebugInfo();
        }
    });
    
    loadRecommendations();
});

async function loadRecommendations() {
    try {
        document.getElementById('loadingState').style.display = 'block';
        updateLoadingMessage('Loading your recommendations...');
        
        // Clear any existing display first
        hideAllSections();
        
        const response = await fetch('/my-recommendations', {
            method: 'GET',
            cache: 'no-cache', // Ensure we get fresh data
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        });
        
        const data = await response.json();
        
        document.getElementById('loadingState').style.display = 'none';

        if (debugMode) {
            console.log('Loaded recommendations data:', data);
            updateDebugInfo('Recommendations loaded', data);
        }

        if (data.success && data.stats.total_recommendations > 0) {
            currentRecommendations = data.recommendations;
            userStats = data.stats;
            
            displayStats(data.stats);
            displayRecommendations(data.recommendations);
            
            document.getElementById('actionButtons').style.display = 'flex';
        } else {
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading recommendations:', error);
        document.getElementById('loadingState').style.display = 'none';
        showEmptyState();
        
        if (debugMode) {
            updateDebugInfo('Error loading recommendations', error);
        }
    }
}

async function loadDebugInfo() {
    if (!debugMode) return;
    
    try {
        const response = await fetch('/debug-user-data');
        const data = await response.json();
        updateDebugInfo('Debug data loaded', data);
    } catch (error) {
        updateDebugInfo('Debug error', error);
    }
}

function updateDebugInfo(action, data) {
    if (!debugMode) return;
    
    const debugDiv = document.getElementById('debugInfo');
    debugDiv.style.display = 'block';
    debugDiv.innerHTML = `
        <strong>Debug: ${action}</strong><br>
        <small>${new Date().toLocaleTimeString()}</small><br>
        <pre>${JSON.stringify(data, null, 2)}</pre>
    `;
}

function toggleDebugInfo() {
    const debugDiv = document.getElementById('debugInfo');
    debugDiv.style.display = debugDiv.style.display === 'none' ? 'block' : 'none';
}

function updateLoadingMessage(message) {
    const loadingText = document.querySelector('#loadingState p');
    if (loadingText) {
        loadingText.textContent = message;
    }
}

function displayStats(stats) {
    document.getElementById('completedCount').textContent = stats.completed_count;
    document.getElementById('pendingCount').textContent = stats.pending_count;
    document.getElementById('inProgressCount').textContent = stats.in_progress_count;
    document.getElementById('riskScore').textContent = stats.risk_score.toFixed(1);
}

function displayRecommendations(recommendations) {
    const categories = [
        { key: 'immediate_actions', sectionId: 'immediateSection', listId: 'immediateList' },
        { key: 'medical_advice', sectionId: 'medicalSection', listId: 'medicalList' },
        { key: 'lifestyle_changes', sectionId: 'lifestyleSection', listId: 'lifestyleList' },
        { key: 'monitoring', sectionId: 'monitoringSection', listId: 'monitoringList' }
    ];

    let hasAnyRecommendations = false;

    categories.forEach(category => {
        const categoryRecs = recommendations[category.key] || [];
        const section = document.getElementById(category.sectionId);
        const list = document.getElementById(category.listId);

        if (categoryRecs.length > 0) {
            hasAnyRecommendations = true;
            section.style.display = 'block';
            list.innerHTML = '';

            categoryRecs.forEach(rec => {
                const recElement = createRecommendationElement(rec);
                list.appendChild(recElement);
            });
        }
    });

    if (!hasAnyRecommendations) {
        showEmptyState();
    }
}

function createRecommendationElement(rec) {
    const recDiv = document.createElement('div');
    recDiv.className = `recommendation-item ${rec.priority}-priority ${rec.status}`;
    
    const isCompleted = rec.status === 'completed';
    
    recDiv.innerHTML = `
        <div class="recommendation-checkbox ${isCompleted ? 'checked' : ''}" 
             onclick="toggleRecommendation(${rec.id}, '${rec.status}')">
            ${isCompleted ? '<i class="fas fa-check"></i>' : ''}
        </div>
        <div class="recommendation-content">
            <div class="recommendation-text">${rec.text}</div>
            <div class="recommendation-meta">
                <span class="priority-badge ${rec.priority}">${rec.priority} Priority</span>
                <span class="status-badge ${rec.status}">${rec.status.replace('_', ' ')}</span>
                <span><i class="fas fa-calendar"></i> ${formatDate(rec.created_at)}</span>
            </div>
        </div>
    `;

    return recDiv;
}

async function toggleRecommendation(recommendationId, currentStatus) {
    let newStatus;
    
    if (currentStatus === 'pending') {
        newStatus = 'in_progress';
    } else if (currentStatus === 'in_progress') {
        newStatus = 'completed';
    } else if (currentStatus === 'completed') {
        newStatus = 'pending';
    } else {
        newStatus = 'in_progress';
    }

    try {
        const response = await fetch('/update-recommendation-status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recommendation_id: recommendationId,
                status: newStatus
            })
        });

        const data = await response.json();

        if (data.success) {
            // Reload recommendations to reflect changes
            await loadRecommendations();
        } else {
            alert('Failed to update recommendation status: ' + data.error);
        }
    } catch (error) {
        console.error('Error updating recommendation:', error);
        alert('Network error. Please try again.');
    }
}

function showEmptyState() {
    hideAllSections();
    document.getElementById('emptyState').style.display = 'block';
    document.getElementById('actionButtons').style.display = 'none';
}

function hideAllSections() {
    const sections = ['immediateSection', 'medicalSection', 'lifestyleSection', 'monitoringSection', 'emptyState'];
    sections.forEach(sectionId => {
        document.getElementById(sectionId).style.display = 'none';
    });
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

function showClearConfirmModal() {
    document.getElementById('clearConfirmModal').style.display = 'block';
}

function closeClearConfirmModal() {
    document.getElementById('clearConfirmModal').style.display = 'none';
}

async function clearAndStartNewAnalysis() {
    try {
        // Show enhanced loading state
        document.getElementById('loadingState').style.display = 'block';
        updateLoadingMessage('Clearing all previous data...');
        
        // Hide modal and current content immediately
        closeClearConfirmModal();
        document.getElementById('actionButtons').style.display = 'none';
        hideAllSections();

        if (debugMode) {
            updateDebugInfo('Starting data clear', { action: 'clear_and_new_analysis' });
        }

        // Call the enhanced endpoint to completely clear data
        const response = await fetch('/start-new-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (debugMode) {
            updateDebugInfo('Clear response received', data);
        }

        if (data.success) {
            // Show success message with more details
            updateLoadingMessage('All data cleared successfully! Redirecting to new analysis...');
            
            // Brief pause to show success message
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Redirect to eye analysis page
            window.location.href = '{{ url_for("main.eye_analysis") }}';
        } else {
            // Show error
            console.error('Failed to clear data:', data.error);
            alert('Failed to clear recommendations: ' + data.error);
            
            // Try to reload current state
            await loadRecommendations();
        }
    } catch (error) {
        console.error('Error clearing recommendations:', error);
        alert('Network error. Please try again.');
        
        if (debugMode) {
            updateDebugInfo('Clear error', error);
        }
        
        // Try to reload current state
        document.getElementById('loadingState').style.display = 'none';
        await loadRecommendations();
    }
}

function exportRecommendations() {
    // Create a simple text export of recommendations
    let exportText = "Eye Health Recommendations Report\n";
    exportText += "====================================\n\n";
    
    exportText += `Generated: ${new Date().toLocaleString()}\n`;
    exportText += `Total Recommendations: ${userStats.total_recommendations}\n`;
    exportText += `Completed: ${userStats.completed_count}\n`;
    exportText += `Pending: ${userStats.pending_count}\n`;
    exportText += `In Progress: ${userStats.in_progress_count}\n`;
    exportText += `Risk Score: ${userStats.risk_score}\n\n`;

    const categories = [
        { key: 'immediate_actions', title: 'Immediate Actions' },
        { key: 'medical_advice', title: 'Medical Advice' },
        { key: 'lifestyle_changes', title: 'Lifestyle Changes' },
        { key: 'monitoring', title: 'Monitoring' }
    ];

    categories.forEach(category => {
        const categoryRecs = currentRecommendations[category.key] || [];
        if (categoryRecs.length > 0) {
            exportText += `${category.title}:\n`;
            exportText += "".padStart(category.title.length + 1, "-") + "\n";
            categoryRecs.forEach((rec, index) => {
                exportText += `${index + 1}. ${rec.text} [${rec.status.toUpperCase()}]\n`;
                exportText += `   Priority: ${rec.priority.toUpperCase()}\n`;
                exportText += `   Created: ${formatDate(rec.created_at)}\n`;
                if (rec.completed_at) {
                    exportText += `   Completed: ${formatDate(rec.completed_at)}\n`;
                }
                exportText += "\n";
            });
            exportText += "\n";
        }
    });

    // Add footer
    exportText += "\n====================================\n";
    exportText += "This report was generated by BlinkWell Eye Health Analysis System\n";
    exportText += "Please consult with a healthcare professional for personalized medical advice.\n";

    // Download as text file
    const blob = new Blob([exportText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `eye-health-recommendations-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // ESC key to close modal
    if (event.key === 'Escape') {
        closeClearConfirmModal();
    }
    
    // Ctrl+R to refresh recommendations (prevent default browser refresh)
    if (event.ctrlKey && event.key === 'r') {
        event.preventDefault();
        loadRecommendations();
    }
});

// Auto-refresh every 30 seconds if page is visible
let refreshInterval;

document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        if (refreshInterval) {
            clearInterval(refreshInterval);
        }
    } else {
        // Refresh when page becomes visible again
        loadRecommendations();
        
        // Set up auto-refresh
        refreshInterval = setInterval(loadRecommendations, 30000);
    }
});

// Initial auto-refresh setup
if (!document.hidden) {
    refreshInterval = setInterval(loadRecommendations, 30000);
}

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = document.getElementById('clearConfirmModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}