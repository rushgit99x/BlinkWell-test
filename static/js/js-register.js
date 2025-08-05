// Password strength checker
const passwordInput = document.getElementById('password');
const strengthIndicator = document.getElementById('passwordStrength');
const strengthText = document.getElementById('strengthText');
const registerBtn = document.getElementById('registerBtn');
const termsCheckbox = document.getElementById('terms');

function checkPasswordStrength(password) {
    let strength = 0;
    let feedback = [];

    if (password.length >= 8) strength += 1;
    else feedback.push('at least 8 characters');

    if (/[a-z]/.test(password)) strength += 1;
    else feedback.push('lowercase letters');

    if (/[A-Z]/.test(password)) strength += 1;
    else feedback.push('uppercase letters');

    if (/[0-9]/.test(password)) strength += 1;
    else feedback.push('numbers');

    if (/[^A-Za-z0-9]/.test(password)) strength += 1;
    else feedback.push('special characters');

    return { strength, feedback };
}

passwordInput.addEventListener('input', function() {
    const password = this.value;
    const { strength, feedback } = checkPasswordStrength(password);
    
    if (password.length > 0) {
        strengthIndicator.style.display = 'block';
        
        // Remove all strength classes
        strengthIndicator.className = 'password-strength';
        
        // Add appropriate strength class
        if (strength <= 2) {
            strengthIndicator.classList.add('strength-weak');
            strengthText.textContent = 'Weak - Add ' + feedback.slice(0, 2).join(', ');
        } else if (strength === 3) {
            strengthIndicator.classList.add('strength-fair');
            strengthText.textContent = 'Fair - Add ' + feedback.join(', ');
        } else if (strength === 4) {
            strengthIndicator.classList.add('strength-good');
            strengthText.textContent = 'Good - Add ' + feedback.join(', ');
        } else {
            strengthIndicator.classList.add('strength-strong');
            strengthText.textContent = 'Strong password!';
        }
    } else {
        strengthIndicator.style.display = 'none';
    }
    
    updateRegisterButton();
});

// Enable/disable register button based on form validity
function updateRegisterButton() {
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const termsAccepted = termsCheckbox.checked;
    const passwordStrong = checkPasswordStrength(password).strength >= 3;
    
    if (username && email && password && termsAccepted && passwordStrong) {
        registerBtn.disabled = false;
    } else {
        registerBtn.disabled = true;
    }
}

// Add event listeners to form inputs
document.querySelectorAll('#registerForm input').forEach(input => {
    input.addEventListener('input', updateRegisterButton);
    input.addEventListener('change', updateRegisterButton);
});

// Form submission with loading state
document.getElementById('registerForm').addEventListener('submit', function(e) {
    const submitBtn = document.getElementById('registerBtn');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.innerHTML = '<span class="loading"></span> Creating Account...';
    submitBtn.disabled = true;
    
    // Re-enable after a short delay if form validation fails
    setTimeout(() => {
        if (submitBtn.disabled) {
            submitBtn.innerHTML = originalText;
            updateRegisterButton();
        }
    }, 5000);
});

// Input focus effects
document.querySelectorAll('.form-group input').forEach(input => {
    input.addEventListener('focus', function() {
        this.parentElement.querySelector('.input-icon').style.color = '#667eea';
    });
    
    input.addEventListener('blur', function() {
        if (!this.value) {
            this.parentElement.querySelector('.input-icon').style.color = '#999';
        }
    });
});

// Auto-hide alerts after 5 seconds
document.querySelectorAll('.alert').forEach(alert => {
    setTimeout(() => {
        alert.style.opacity = '0';
        alert.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            alert.remove();
        }, 300);
    }, 5000);
});

// Navbar scroll effect (if needed)
window.addEventListener('scroll', function() {
    const navbar = document.getElementById('navbar');
    if (navbar) {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    }
});

// Initialize form on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set initial state of register button
    updateRegisterButton();
});