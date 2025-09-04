// CNN Playground - Global JavaScript Module

/**
 * GLOBAL IMAGE MODAL FUNCTION - WORKING VERSION
 * This function must be available globally for onclick handlers
 */
function openImageModal(src) {
    console.log('[ImageModal] Opening modal for:', src);

    // Remove existing modal if present
    const existingModal = document.getElementById('imageModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal container
    const modal = document.createElement('div');
    modal.id = 'imageModal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 999999;
        cursor: pointer;
        backdrop-filter: blur(5px);
        animation: modalFadeIn 0.3s ease;
    `;

    // Create image element
    const img = document.createElement('img');
    img.src = src;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        border-radius: 10px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
    `;

    // Create close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 24px;
        font-weight: bold;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        z-index: 1000000;
        color: #333;
    `;

    // Create info text
    const infoText = document.createElement('div');
    infoText.innerHTML = 'Click anywhere to close • ESC';
    infoText.style.cssText = `
        position: absolute;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        color: rgba(255, 255, 255, 0.8);
        font-size: 14px;
        background: rgba(0, 0, 0, 0.6);
        padding: 8px 16px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        pointer-events: none;
    `;

    // Close function
    const closeModal = (e) => {
        if (e) e.stopPropagation();
        console.log('[ImageModal] Closing modal');

        modal.style.animation = 'modalFadeOut 0.3s ease';
        document.body.style.overflow = '';

        setTimeout(() => {
            if (document.body.contains(modal)) {
                document.body.removeChild(modal);
                document.removeEventListener('keydown', handleKeyDown);
            }
        }, 300);
    };

    // Keyboard handler
    const handleKeyDown = (e) => {
        if (e.key === 'Escape') {
            closeModal(e);
        }
    };

    // Event listeners
    modal.addEventListener('click', closeModal);
    closeBtn.addEventListener('click', closeModal);
    document.addEventListener('keydown', handleKeyDown);

    // Prevent image click from closing modal
    img.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // Close button hover effects
    closeBtn.addEventListener('mouseenter', () => {
        closeBtn.style.background = 'rgba(255, 255, 255, 1)';
        closeBtn.style.transform = 'scale(1.1)';
    });

    closeBtn.addEventListener('mouseleave', () => {
        closeBtn.style.background = 'rgba(255, 255, 255, 0.9)';
        closeBtn.style.transform = 'scale(1)';
    });

    // Add CSS animations if not present
    if (!document.getElementById('modal-animations')) {
        const style = document.createElement('style');
        style.id = 'modal-animations';
        style.textContent = `
            @keyframes modalFadeIn {
                from { 
                    opacity: 0; 
                    transform: scale(0.8); 
                }
                to { 
                    opacity: 1; 
                    transform: scale(1); 
                }
            }
            @keyframes modalFadeOut {
                from { 
                    opacity: 1; 
                    transform: scale(1); 
                }
                to { 
                    opacity: 0; 
                    transform: scale(0.8); 
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Assemble modal
    modal.appendChild(img);
    modal.appendChild(closeBtn);
    modal.appendChild(infoText);

    // Add to DOM
    document.body.style.overflow = 'hidden';
    document.body.appendChild(modal);

    console.log('[ImageModal] Modal created and displayed successfully');
}

/**
 * Theme Management Functions
 */
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.querySelector('.theme-toggle');

    if (body.getAttribute('data-theme') === 'dark') {
        body.removeAttribute('data-theme');
        themeToggle.textContent = '🌙';
        localStorage.setItem('theme', 'light');
    } else {
        body.setAttribute('data-theme', 'dark');
        themeToggle.textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    }
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    const themeToggle = document.querySelector('.theme-toggle');

    if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        if (themeToggle) themeToggle.textContent = '☀️';
    }
}

/**
 * Status Message System
 */
function showMessage(message, type = 'info') {
    const container = document.querySelector('.container');
    if (!container) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `status-message status-${type}`;
    messageDiv.textContent = message;

    container.insertBefore(messageDiv, container.firstChild);

    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.remove();
        }
    }, 5000);
}

function showError(message) {
    showMessage(message, 'error');
}

function showSuccess(message) {
    showMessage(message, 'success');
}

function showWarning(message) {
    showMessage(message, 'warning');
}

/**
 * Animation System
 */
function initializeAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.fade-in').forEach(el => {
        observer.observe(el);
    });
}

/**
 * API Communication
 */
async function makeRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Request failed:', error);
        throw error;
    }
}

/**
 * File Validation
 */
function validateImageFile(file, maxSizeMB = 10) {
    if (!file.type.startsWith('image/')) {
        throw new Error('Please select a valid image file (JPG, PNG)');
    }

    if (file.size > maxSizeMB * 1024 * 1024) {
        throw new Error(`File size must be less than ${maxSizeMB}MB`);
    }

    return true;
}

/**
 * UI Utility Functions
 */
function updateProgressBar(element, percentage) {
    if (element) {
        element.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
    }
}

function setLoadingState(element, loading = true) {
    if (!element) return;

    if (loading) {
        element.disabled = true;
        element.classList.add('loading');

        if (!element.dataset.originalText) {
            element.dataset.originalText = element.textContent;
        }

        element.innerHTML = '<span class="loading-spinner"></span> Loading...';
    } else {
        element.disabled = false;
        element.classList.remove('loading');

        if (element.dataset.originalText) {
            element.textContent = element.dataset.originalText;
        }
    }
}

/**
 * Number Formatting
 */
function formatNumber(num, decimals = 2) {
    if (typeof num !== 'number') return 'N/A';
    return num.toFixed(decimals);
}

function formatPercentage(num, decimals = 1) {
    if (typeof num !== 'number') return 'N/A';
    return `${(num * 100).toFixed(decimals)}%`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Performance Optimization
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Navigation Helpers
 */
function navigateTo(url) {
    window.location.href = url;
}

function setActiveNavTab(currentPage) {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });

    const activeTab = document.querySelector(`.nav-tab[href="/${currentPage}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
}

/**
 * Image Processing
 */
function resizeImage(file, maxWidth = 800, maxHeight = 800, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = () => {
            let { width, height } = img;

            if (width > height) {
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }
            } else {
                if (height > maxHeight) {
                    width = (width * maxHeight) / height;
                    height = maxHeight;
                }
            }

            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);

            canvas.toBlob(resolve, file.type, quality);
        };

        img.src = URL.createObjectURL(file);
    });
}

/**
 * Modal System
 */
function createModal(title, content, actions = []) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        backdrop-filter: blur(5px);
    `;

    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: var(--bg-primary);
        border-radius: 15px;
        padding: 30px;
        max-width: 500px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    `;

    modalContent.innerHTML = `
        <h2 style="margin-bottom: 20px; color: var(--text-primary);">${title}</h2>
        <div style="margin-bottom: 20px; color: var(--text-secondary);">${content}</div>
        <div style="display: flex; gap: 10px; justify-content: flex-end;">
            ${actions.map(action => `
                <button class="btn ${action.class || ''}" onclick="${action.onclick}">${action.text}</button>
            `).join('')}
        </div>
    `;

    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    modal.onclick = (e) => {
        if (e.target === modal) {
            closeModal(modal);
        }
    };

    return modal;
}

function closeModal(modal) {
    if (modal && modal.parentNode) {
        modal.parentNode.removeChild(modal);
    }
}

/**
 * Application Initialization
 */
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    initializeAnimations();

    // Delayed animation trigger
    setTimeout(() => {
        document.querySelectorAll('.fade-in').forEach(el => {
            el.classList.add('visible');
        });
    }, 300);

    console.log('[CNNPlayground] Application initialized');
});

/**
 * Global API Exports
 */
window.CNNPlayground = {
    // Theme management
    toggleTheme,
    loadTheme,

    // Messaging system
    showMessage,
    showError,
    showSuccess,
    showWarning,

    // Network communication
    makeRequest,

    // File handling
    validateImageFile,
    resizeImage,

    // UI utilities
    updateProgressBar,
    setLoadingState,
    formatNumber,
    formatPercentage,
    formatFileSize,

    // Performance optimization
    debounce,
    throttle,

    // Navigation
    navigateTo,
    setActiveNavTab,

    // Modal system
    createModal,
    closeModal
};

/**
 * CRITICAL: Make image modal function available globally
 * This ensures onclick="openImageModal(src)" works from any HTML
 */
window.openImageModal = openImageModal;

console.log('[CNNPlayground] Core module loaded with working image modal');