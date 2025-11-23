// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Emotion to symbol mapping (no emojis)
const EMOTION_SYMBOLS = {
    'Happy': 'HAPPY',
    'Sad': 'SAD',
    'Angry': 'ANGRY',
    'Surprise': 'SURPRISE',
    'Fear': 'FEAR',
    'Disgust': 'DISGUST',
    'Neutral': 'NEUTRAL'
};

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const heatmapCheck = document.getElementById('heatmap-check');
const geminiCheck = document.getElementById('gemini-check');
const resultsSection = document.getElementById('results-section');
const loading = document.getElementById('loading');
const resultsContent = document.getElementById('results-content');

// Webcam elements
const webcam = document.getElementById('webcam');
const webcamCanvas = document.getElementById('webcam-canvas');
const startWebcamBtn = document.getElementById('start-webcam-btn');
const stopWebcamBtn = document.getElementById('stop-webcam-btn');
const captureBtn = document.getElementById('capture-btn');

let currentFile = null;
let webcamStream = null;

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        // Reset results
        resultsSection.style.display = 'none';
    });
});

// Upload area interactions
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

clearBtn.addEventListener('click', () => {
    currentFile = null;
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
});

analyzeBtn.addEventListener('click', () => {
    if (currentFile) {
        analyzeImage(currentFile);
    }
});

// Webcam controls
startWebcamBtn.addEventListener('click', startWebcam);
stopWebcamBtn.addEventListener('click', stopWebcam);
captureBtn.addEventListener('click', captureAndAnalyze);

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    currentFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    // Show loading
    resultsSection.style.display = 'block';
    loading.style.display = 'block';
    resultsContent.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('include_heatmap', heatmapCheck.checked);
    formData.append('include_gemini', geminiCheck.checked);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict?include_heatmap=${heatmapCheck.checked}&include_gemini=${geminiCheck.checked}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}\n\nMake sure the backend is running on ${API_BASE_URL}`);
        resultsSection.style.display = 'none';
    }
}

function displayResults(data) {
    loading.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Update emotion card
    const emotionIcon = document.getElementById('emotion-icon');
    const emotionLabel = document.getElementById('emotion-label');
    const confidence = document.getElementById('confidence');
    const confidenceFill = document.getElementById('confidence-fill');
    
    emotionIcon.textContent = EMOTION_SYMBOLS[data.emotion] || data.emotion.toUpperCase();
    emotionLabel.textContent = data.emotion;
    confidence.textContent = `${data.confidence.toFixed(1)}%`;
    confidenceFill.style.width = `${data.confidence}%`;
    
    // Update probabilities
    const probabilitiesList = document.getElementById('probabilities-list');
    probabilitiesList.innerHTML = '';
    
    // Sort probabilities
    const sortedProbs = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([emotion, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-label">${emotion}</span>
            <span class="prob-value">${prob.toFixed(1)}%</span>
        `;
        probabilitiesList.appendChild(item);
    });
    
    // Update Gemini insight
    const insightCard = document.getElementById('insight-card');
    const geminiInsight = document.getElementById('gemini-insight');
    
    if (data.gemini_insight) {
        insightCard.style.display = 'block';
        geminiInsight.textContent = data.gemini_insight;
    } else {
        insightCard.style.display = 'none';
    }
    
    // Update heatmap
    const heatmapCard = document.getElementById('heatmap-card');
    const heatmapImage = document.getElementById('heatmap-image');
    
    if (data.heatmap_base64) {
        heatmapCard.style.display = 'block';
        heatmapImage.src = `data:image/jpeg;base64,${data.heatmap_base64}`;
    } else {
        heatmapCard.style.display = 'none';
    }
    
    // Show warning if not confident
    if (!data.is_confident) {
        const warning = document.createElement('div');
        warning.className = 'insight-card';
        warning.style.background = '#fff3cd';
        warning.style.borderLeft = '5px solid #ffc107';
        warning.innerHTML = `
            <h3>Low Confidence Warning</h3>
            <p>The model is uncertain about this prediction. Consider using a clearer facial image with better lighting and a visible face.</p>
        `;
        resultsContent.insertBefore(warning, resultsContent.firstChild);
    }
}

// Webcam functions
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        webcam.srcObject = webcamStream;
        
        startWebcamBtn.style.display = 'none';
        stopWebcamBtn.style.display = 'inline-block';
        captureBtn.style.display = 'inline-block';
        
    } catch (error) {
        console.error('Webcam error:', error);
        alert('Could not access webcam. Please ensure you have granted camera permissions.');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        
        startWebcamBtn.style.display = 'inline-block';
        stopWebcamBtn.style.display = 'none';
        captureBtn.style.display = 'none';
    }
}

function captureAndAnalyze() {
    // Set canvas size to match video
    webcamCanvas.width = webcam.videoWidth;
    webcamCanvas.height = webcam.videoHeight;
    
    // Draw current frame to canvas
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);
    
    // Convert to blob and analyze
    webcamCanvas.toBlob(async (blob) => {
        const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
        
        // Show preview
        previewImage.src = URL.createObjectURL(blob);
        
        // Analyze
        await analyzeImage(file);
    }, 'image/jpeg', 0.95);
}

// Check API health on load
async function checkAPI() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('✅ API Status:', data);
    } catch (error) {
        console.warn('⚠️ API not reachable. Make sure backend is running.');
    }
}

checkAPI();
