const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');
const charCounter = document.getElementById('charCounter');

let isWaitingForResponse = false;
const MAX_CHARS = 500;

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
    updateCharCounter();
});

// Update character counter
function updateCharCounter() {
    const currentLength = chatInput.value.length;
    const remaining = MAX_CHARS - currentLength;
    
    charCounter.textContent = `${currentLength}/${MAX_CHARS}`;
    charCounter.className = 'char-counter';
    
    if (remaining <= 50 && remaining > 20) {
        charCounter.classList.add('warning');
    } else if (remaining <= 20) {
        charCounter.classList.add('error');
    }
}

// Send message on Enter (but allow Shift+Enter for new lines)
chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const avatar = isUser ? 
        '<div class="message-avatar"><i class="fas fa-user"></i></div>' :
        '<div class="message-avatar"><i class="fas fa-robot"></i></div>';
    
    messageDiv.innerHTML = `
        ${avatar}
        <div class="message-content">
            <div>${content}</div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;
    
    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage && isUser) {
        welcomeMessage.remove();
    }
    
    // Insert before typing indicator
    chatMessages.insertBefore(messageDiv, typingIndicator);
    scrollToBottom();
}

function showTyping() {
    typingIndicator.classList.add('show');
    scrollToBottom();
}

function hideTyping() {
    typingIndicator.classList.remove('show');
}

function scrollToBottom() {
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    chatMessages.insertBefore(errorDiv, typingIndicator);
    scrollToBottom();
    
    // Remove error after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 5000);
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isWaitingForResponse) return;

    // Validate character limit
    if (message.length > MAX_CHARS) {
        showError(`Message too long. Maximum ${MAX_CHARS} characters allowed.`);
        return;
    }

    // Add user message
    addMessage(message, true);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    updateCharCounter();

    // Disable input and show typing
    isWaitingForResponse = true;
    sendButton.disabled = true;
    chatInput.disabled = true;
    showTyping();

    try {
        const response = await fetch('/api/chatbot/message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        if (data.success) {
            // Add bot response after a slight delay for realism
            setTimeout(() => {
                hideTyping();
                addMessage(data.response);
            }, 1000);
        } else {
            hideTyping();
            showError(data.error || 'Sorry, I encountered an error. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        hideTyping();
        showError('Connection error. Please check your internet and try again.');
    } finally {
        // Re-enable input
        setTimeout(() => {
            isWaitingForResponse = false;
            sendButton.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
        }, 1500);
    }
}

function sendQuickQuestion(question) {
    chatInput.value = question;
    updateCharCounter();
    sendMessage();
}

// Check chatbot health on page load
async function checkChatbotHealth() {
    try {
        const response = await fetch('/api/chatbot/health');
        const data = await response.json();
        
        if (!data.success) {
            showError('Chatbot is currently unavailable. Please try again later.');
        }
    } catch (error) {
        showError('Unable to connect to chatbot service.');
    }
}

// Initialize
window.addEventListener('load', () => {
    chatInput.focus();
    checkChatbotHealth();
    updateCharCounter();
});

// Add some interactive elements
document.addEventListener('DOMContentLoaded', function() {
    // Add subtle animations to the interface
    const container = document.querySelector('.chatbot-container');
    container.style.opacity = '0';
    container.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        container.style.transition = 'all 0.6s ease';
        container.style.opacity = '1';
        container.style.transform = 'translateY(0)';
    }, 100);
});