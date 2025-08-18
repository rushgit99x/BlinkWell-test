// Global variables
let currentStep = 1;
let imageAnalysisData = null;
let questionnaireData = {};

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    initializeSliders();
});

function setupEventListeners() {
    // File upload handling
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('dragleave', handleDragLeave);

    // Form submission
    document.getElementById('questionnaireForm').addEventListener('submit', handleQuestionnaireSubmit);
}

function initializeSliders() {
    const sliders = document.querySelectorAll('.slider');
    sliders.forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + 'Value');
        slider.addEventListener('input', function() {
            valueDisplay.textContent = this.value;
        });
    });
}

// Image upload functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        validateAndPreviewFile(files[0]);
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('dragover');
}

function validateAndPreviewFile(file) {
    hideMessages();

    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        showMessage('error', 'Please upload a valid image file (PNG, JPG, JPEG)', 'imageMessage');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        showMessage('error', 'File size must be less than 5MB', 'imageMessage');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('previewContainer').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function analyzeImage() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) {
        showMessage('error', 'Please select an image file first', 'imageMessage');
        return;
    }

    showLoading();
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<div class="loading-spinner"></div> Analyzing...';
    hideMessages();

    const formData = new FormData();
    formData.append('eye_image', file);

    fetch('/analyze-eye-image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';

        if (data.success) {
            // Store the complete image analysis data
            imageAnalysisData = {
                prediction: data.prediction,
                confidence: data.confidence / 100, // Convert percentage to decimal
                dry_eye_probability: data.prediction === 'Dry Eyes' ? 1.0 : 0.0,
                is_valid_eye: data.is_valid_eye || true
            };
            
            displayImageResult(data);
            showMessage('success', 'Image analysis completed successfully!', 'imageMessage');
        } else {
            imageAnalysisData = null;
            showMessage('error', data.error || 'Analysis failed. Please try again.', 'imageMessage');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        imageAnalysisData = null;
        hideLoading();
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';
        showMessage('error', 'Network error. Please check your connection and try again.', 'imageMessage');
    });
}

function displayImageResult(data) {
    const resultDiv = document.getElementById('imageResult');
    const resultText = document.getElementById('imageResultText');
    
    resultText.innerHTML = `
        <strong>Prediction:</strong> ${data.prediction}<br>
        <strong>Confidence:</strong> ${data.confidence}%<br>
        <strong>Status:</strong> ${data.message}
    `;
    
    resultDiv.style.display = 'block';
}

function proceedToQuestionnaire() {
    updateStep(2);
    document.getElementById('imageAnalysisCard').style.display = 'none';
    document.getElementById('questionnaireCard').style.display = 'block';
    
    // Scroll to questionnaire
    document.getElementById('questionnaireCard').scrollIntoView({ behavior: 'smooth' });
}

function handleQuestionnaireSubmit(e) {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(e.target);
    const questionnaireData = {};
    
    for (let [key, value] of formData.entries()) {
        questionnaireData[key] = value;
    }

    // Add image analysis data if available
    if (imageAnalysisData && imageAnalysisData.is_valid_eye) {
        questionnaireData.image_analysis = {
            dry_eye_probability: imageAnalysisData.dry_eye_probability,
            confidence: imageAnalysisData.confidence,
            prediction: imageAnalysisData.prediction
        };
    }

    submitQuestionnaire(questionnaireData);
}

function submitQuestionnaire(data) {
    showLoading();
    hideMessages();

    fetch('/submit-questionnaire', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();

        if (result.success) {
            displayComprehensiveResults(result);
            updateStep(3);
            document.getElementById('questionnaireCard').style.display = 'none';
            document.getElementById('resultsCard').style.display = 'block';
            document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth' });
        } else {
            showMessage('error', result.error || 'Analysis failed. Please try again.', 'questionnaireMessage');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        showMessage('error', 'Network error. Please check your connection and try again.', 'questionnaireMessage');
    });
}

function displayComprehensiveResults(data) {
    console.log('Displaying results:', data); // Debug log

    // Update risk level and description
    const riskLevel = document.getElementById('riskLevel');
    const riskDescription = document.getElementById('riskDescription');
    
    riskLevel.textContent = data.combined_analysis.risk_level;
    riskLevel.className = `risk-level ${data.combined_analysis.risk_level.toLowerCase().replace(' ', '-')}`;
    
    riskDescription.textContent = data.combined_analysis.has_dry_eyes ? 
        'Analysis indicates potential dry eye disease. Please review recommendations below.' :
        'Analysis shows low risk for dry eye disease. Continue healthy eye practices.';

    // Update individual analysis results
    if (data.individual_predictions.image_analysis) {
        const imageProb = data.individual_predictions.image_analysis.probability * 100;
        document.getElementById('imageProbability').textContent = `${imageProb.toFixed(1)}%`;
        updateProgressBar('imageBar', imageProb, imageProb);
        
        // Show that image was analyzed
        console.log('Image analysis available:', data.individual_predictions.image_analysis);
    } else {
        document.getElementById('imageProbability').textContent = 'Questionnaire only';
        updateProgressBar('imageBar', 0, 0);
        console.log('No image analysis data available');
    }

    const textProb = data.individual_predictions.text_analysis.probability * 100;
    document.getElementById('questionnaireProbability').textContent = `${textProb.toFixed(1)}%`;
    updateProgressBar('questionnaireBar', textProb, textProb);

    // Update combined analysis
    const combinedProb = data.combined_analysis.dry_eye_probability * 100;
    document.getElementById('finalRiskScore').textContent = data.combined_analysis.risk_score;
    document.getElementById('confidenceLevel').textContent = `${(data.combined_analysis.confidence * 100).toFixed(1)}%`;
    updateProgressBar('combinedBar', combinedProb, combinedProb);

    // Display risk factors
    displayRiskFactors(data.risk_factors);

    // Display recommendations
    displayRecommendations(data.recommendations);
}

function updateProgressBar(barId, percentage, value) {
    const bar = document.getElementById(barId);
    bar.style.width = `${percentage}%`;
    
    // Set color based on risk level
    if (value >= 70) {
        bar.className = 'probability-fill high';
    } else if (value >= 40) {
        bar.className = 'probability-fill medium';
    } else {
        bar.className = 'probability-fill low';
    }
}

function displayRiskFactors(riskFactors) {
    const grid = document.getElementById('riskFactorsGrid');
    grid.innerHTML = '';

    if (riskFactors.length === 0) {
        grid.innerHTML = '<p style="text-align: center; color: #666;">No significant risk factors identified.</p>';
        return;
    }

    riskFactors.forEach(factor => {
        const factorDiv = document.createElement('div');
        factorDiv.className = `risk-factor-item ${factor.impact}`;
        factorDiv.innerHTML = `
            <h5>${factor.factor}</h5>
            <p>${factor.value}</p>
            <small>Impact: ${factor.impact}</small>
        `;
        grid.appendChild(factorDiv);
    });
}

function displayRecommendations(recommendations) {
    const content = document.getElementById('recommendationsContent');
    content.innerHTML = '';

    const categories = [
        { key: 'immediate_actions', title: 'Immediate Actions', icon: 'fas fa-exclamation-circle' },
        { key: 'medical_advice', title: 'Medical Advice', icon: 'fas fa-user-md' },
        { key: 'lifestyle_changes', title: 'Lifestyle Changes', icon: 'fas fa-life-ring' },
        { key: 'monitoring', title: 'Ongoing Monitoring', icon: 'fas fa-chart-line' }
    ];

    categories.forEach(category => {
        if (recommendations[category.key] && recommendations[category.key].length > 0) {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'recommendation-category';
            
            const categoryHtml = `
                <h4><i class="${category.icon}"></i> ${category.title}</h4>
                <ul class="recommendation-list">
                    ${recommendations[category.key].map((rec, index) => `
                        <li class="recommendation-item">
                            <div class="recommendation-icon">
                                <i class="fas fa-${getRecommendationIcon(category.key, index)}"></i>
                            </div>
                            <div>${rec}</div>
                        </li>
                    `).join('')}
                </ul>
            `;
            
            categoryDiv.innerHTML = categoryHtml;
            content.appendChild(categoryDiv);
        }
    });
}

function getRecommendationIcon(category, index) {
    const icons = {
        'immediate_actions': ['eye-dropper', 'compress', 'wind'],
        'medical_advice': ['stethoscope', 'calendar-check', 'prescription'],
        'lifestyle_changes': ['mobile-alt', 'moon', 'dumbbell'],
        'monitoring': ['chart-bar', 'clock', 'calendar']
    };
    
    return icons[category] ? icons[category][index % icons[category].length] : 'lightbulb';
}

function updateStep(step) {
    currentStep = step;
    
    // Update progress bar
    const progress = (step / 3) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
    
    // Update step indicators
    for (let i = 1; i <= 3; i++) {
        const stepEl = document.getElementById(`step${i}`);
        stepEl.classList.remove('active', 'completed');
        
        if (i < step) {
            stepEl.classList.add('completed');
        } else if (i === step) {
            stepEl.classList.add('active');
        }
    }
}

function showHistory() {
    const historySection = document.getElementById('historySection');
    const historyContent = document.getElementById('historyContent');

    if (historySection.style.display === 'none') {
        historySection.style.display = 'block';
        loadHistory();
    } else {
        historySection.style.display = 'none';
    }
}

function loadHistory() {
    const historyContent = document.getElementById('historyContent');
    historyContent.innerHTML = '<div style="text-align: center; padding: 20px;"><div class="loading-spinner" style="margin: 0 auto;"></div><p>Loading history...</p></div>';

    fetch('/eye-analysis-history')
    .then(response => response.json())
    .then(data => {
        if (data.success && data.history.length > 0) {
            historyContent.innerHTML = '';
            data.history.forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'analysis-item';
                historyItem.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <strong>${item.prediction}</strong>
                        <span style="color: #666; font-size: 0.9rem;">${new Date(item.updated_date).toLocaleString()}</span>
                    </div>
                    <div>Risk Score: ${item.risk_score}/100</div>
                    <div style="margin-top: 10px;">
                        <small>Risk Factors: ${item.risk_factors.length} identified</small>
                    </div>
                `;
                historyContent.appendChild(historyItem);
            });
        } else {
            historyContent.innerHTML = '<p style="text-align: center; color: #666;">No analysis history found.</p>';
        }
    })
    .catch(error => {
        console.error('Error loading history:', error);
        historyContent.innerHTML = '<p style="text-align: center; color: #dc2626;">Failed to load history.</p>';
    });
}

// Utility functions
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showMessage(type, message, containerId) {
    const messageEl = document.getElementById(containerId);
    messageEl.className = `message ${type}`;
    messageEl.textContent = message;
    messageEl.style.display = 'block';
}

function hideMessages() {
    const messages = document.querySelectorAll('.message');
    messages.forEach(msg => {
        msg.style.display = 'none';
    });
}