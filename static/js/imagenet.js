// ImageNet Classification Interface

class ImageNetClassifier {
    constructor() {
        this.currentFeatureMaps = null;
        this.layerDescriptions = {
            'overview': {
                'description': 'Overview showing original image and representative feature maps from each ResNet50 layer.',
                'size': 'Multiple',
                'channels': 'Various'
            },
            'conv1': {
                'description': 'First convolutional layer - detects basic edges and textures.',
                'size': '112×112',
                'channels': '64'
            },
            'conv2_1': {
                'description': 'Conv2_1 stage - combines edges into simple shapes.',
                'size': '56×56',
                'channels': '256'
            },
            'conv3_1': {
                'description': 'Conv3_1 stage - recognizes object parts.',
                'size': '28×28',
                'channels': '512'
            },
            'conv4_1': {
                'description': 'Conv4_1 stage - complex object features.',
                'size': '14×14',
                'channels': '1024'
            },
            'conv5_1': {
                'description': 'Conv5_1 stage - high-level features for classification.',
                'size': '7×7',
                'channels': '2048'
            }
        };

        this.initialize();
    }

    initialize() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.initializeLayerTabs();
        this.setupImageModalDelegation(); // NEW - Event delegation for images
        console.log('ImageNet Classifier initialized');
    }

    // NEW - Event delegation for all clickable images
    setupImageModalDelegation() {
        console.log('[ImageModal] Setting up event delegation');

        // Listen for clicks on the entire document
        document.addEventListener('click', (e) => {
            const target = e.target;

            // Check if clicked element is an image that should open modal
            if (target.tagName === 'IMG' &&
                (target.classList.contains('gradcam-image') ||
                 target.classList.contains('layer-image') ||
                 target.hasAttribute('data-modal-src'))) {

                e.preventDefault();
                e.stopPropagation();

                const src = target.getAttribute('data-modal-src') || target.src;
                console.log('[ImageModal] Delegated click detected, opening modal for:', src);

                if (window.openImageModal) {
                    window.openImageModal(src);
                } else {
                    console.error('[ImageModal] openImageModal function not found!');
                }
            }
        });

        console.log('[ImageModal] Event delegation setup complete');
    }

    setupEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const uploadSection = document.getElementById('uploadSection');

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFile(e.target.files[0]);
                }
            });
        }

        if (uploadSection) {
            uploadSection.addEventListener('click', () => {
                if (fileInput) fileInput.click();
            });
        }
    }

    setupDragAndDrop() {
        const uploadSection = document.getElementById('uploadSection');
        if (!uploadSection) return;

        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });
    }

    async handleFile(file) {
        try {
            // Validate file
            window.CNNPlayground.validateImageFile(file, 10);

            // Get model selection
            const modelSelect = document.getElementById('modelSelect');
            const selectedModel = modelSelect ? modelSelect.value : 'resnet50';

            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('dataset', 'imagenet');
            formData.append('model', selectedModel);

            // Show loading state
            this.showLoading();

            // Simulate progress
            const progressInterval = this.simulateProgress();

            // Send request
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            // Complete progress
            this.completeProgress();
            clearInterval(progressInterval);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Received analysis data:', data);

            if (data.status === 'success') {
                setTimeout(() => this.displayResults(data, file), 500);
                window.CNNPlayground.showSuccess('Image analysis completed successfully!');
            } else {
                throw new Error(data.message || 'Analysis failed');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            this.hideLoading();
            window.CNNPlayground.showError('Analysis failed: ' + error.message);
        }
    }

    showLoading() {
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        if (loading) loading.style.display = 'block';
        if (results) {
            results.style.display = 'none';
            results.classList.remove('show');
        }
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        const progressContainer = document.getElementById('progressContainer');

        if (loading) loading.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'none';
    }

    simulateProgress() {
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');

        if (progressContainer) progressContainer.style.display = 'block';

        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;

            window.CNNPlayground.updateProgressBar(progressBar, progress);
        }, 200);

        return interval;
    }

    completeProgress() {
        const progressBar = document.getElementById('progressBar');
        window.CNNPlayground.updateProgressBar(progressBar, 100);
    }

    displayResults(data, originalFile) {
        console.log('=== DISPLAY RESULTS DEBUG ===');
        console.log('Full data received:', data);
        console.log('Feature maps data:', data.feature_maps);

        // Display uploaded image
        this.displayUploadedImage(originalFile, data);

        // Display predictions
        this.displayPredictions(data.predictions);

        // Display Grad-CAM
        this.displayGradCAM(data.gradcam_url);

        // Display feature maps
        this.displayFeatureMaps(data);

        // Display network info
        this.displayNetworkInfo(data.model_info);

        // Show results with animation
        const results = document.getElementById('results');
        if (results) {
            results.style.display = 'block';
            setTimeout(() => {
                results.classList.add('show');
            }, 100);
        }

        // Hide loading
        this.hideLoading();
    }

    displayUploadedImage(file, data) {
        const container = document.getElementById('uploadedImageContainer');
        if (!container) return;

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = 'Uploaded image';

        const infoDiv = document.createElement('div');
        infoDiv.className = 'image-info';
        infoDiv.innerHTML = `
            <strong>Image:</strong> ${file.name} •
            <strong>Size:</strong> ${window.CNNPlayground.formatFileSize(file.size)} •
            <strong>Model:</strong> ${data.model_info?.architecture || 'ResNet50'}
        `;

        container.innerHTML = '<h3>Your Image</h3>';
        container.appendChild(img);
        container.appendChild(infoDiv);
    }

    displayPredictions(predictions) {
        const container = document.getElementById('predictions');
        if (!container) return;

        container.innerHTML = '';

        if (predictions && Array.isArray(predictions)) {
            predictions.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = 'prediction-item';

                setTimeout(() => {
                    item.innerHTML = `
                        <span class="prediction-rank">${index + 1}.</span>
                        <span class="prediction-name">${pred.class_name || 'Unknown'}</span>
                        <div class="confidence-container">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${(pred.confidence || 0) * 100}%"></div>
                            </div>
                            <span class="confidence-text">${window.CNNPlayground.formatPercentage(pred.confidence || 0)}</span>
                        </div>
                    `;
                }, index * 150);

                container.appendChild(item);
            });
        }
    }

    displayGradCAM(gradcamUrl) {
        const container = document.getElementById('gradcamContainer');
        if (!container) return;

        if (gradcamUrl) {
            setTimeout(() => {
                console.log('[GradCAM] Creating image with URL:', gradcamUrl);
                container.innerHTML = `
                    <img src="${gradcamUrl}" 
                         alt="Grad-CAM visualization" 
                         class="gradcam-image" 
                         data-modal-src="${gradcamUrl}"
                         style="cursor: pointer;">
                    <p style="margin-top: 15px; color: var(--text-secondary); font-size: 0.9em;">
                        <strong>Click to enlarge</strong><br>
                        Red/yellow areas show neural network attention
                    </p>
                `;
                console.log('[GradCAM] Image created with data-modal-src attribute');
            }, 800);
        } else {
            container.innerHTML = `
                <div class="gradcam-placeholder">
                    <p>Generating attention map...</p>
                    <div class="spinner" style="width: 30px; height: 30px; margin: 15px auto;"></div>
                </div>
            `;
        }
    }

    displayFeatureMaps(data) {
        const layerProgression = document.getElementById('layerProgression');

        if (data.feature_maps) {
            console.log('Setting up feature maps section');

            // Reset previous data
            this.currentFeatureMaps = data.feature_maps;

            if (layerProgression) {
                layerProgression.style.display = 'block';

                // Initialize with overview
                setTimeout(() => {
                    this.showLayerContent('overview');
                }, 500);

                // Update tab availability
                this.updateTabAvailability(data.feature_maps);
            }
        }
    }

    displayNetworkInfo(modelInfo) {
        const container = document.getElementById('featureMaps');
        if (!container || !modelInfo) return;

        container.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div>
                    <h4 style="color: var(--text-primary); margin-bottom: 10px;">Model Details</h4>
                    <p><strong>Architecture:</strong> ${modelInfo.architecture || 'ResNet50'}</p>
                    <p><strong>Dataset:</strong> ${modelInfo.dataset || 'ImageNet'}</p>
                    <p><strong>Input Size:</strong> ${modelInfo.input_size || '224x224'}</p>
                    <p><strong>Device:</strong> ${modelInfo.device || 'CPU'}</p>
                </div>
                <div>
                    <h4 style="color: var(--text-primary); margin-bottom: 10px;">Processing Info</h4>
                    <p><strong>Original Size:</strong> ${modelInfo.original_size || 'Unknown'}</p>
                    <p><strong>Preprocessing:</strong> Aspect ratio preserved</p>
                    <p><strong>Feature Maps:</strong> <span style="color: #10B981;">✓ Generated</span></p>
                    <p><strong>Status:</strong> <span style="color: #10B981;">✓ Complete</span></p>
                </div>
            </div>
        `;
    }

    initializeLayerTabs() {
        console.log('Initializing layer tabs');

        document.querySelectorAll('.layer-tab').forEach(tab => {
            const layerName = tab.getAttribute('data-layer');
            console.log('Setting up tab for layer:', layerName);

            tab.addEventListener('click', () => {
                console.log('Tab clicked:', layerName);

                // Remove active class from all tabs
                document.querySelectorAll('.layer-tab').forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                tab.classList.add('active');

                this.showLayerContent(layerName);
            });
        });
    }

    updateTabAvailability(featureMaps) {
        document.querySelectorAll('.layer-tab').forEach(tab => {
            const layerName = tab.getAttribute('data-layer');
            console.log(`Checking tab availability for: ${layerName}`);

            if (layerName === 'overview') {
                tab.style.opacity = '1';
                tab.style.pointerEvents = 'auto';
                console.log(`Overview tab enabled`);
            } else if (featureMaps.individual_layers && featureMaps.individual_layers[layerName]) {
                tab.style.opacity = '1';
                tab.style.pointerEvents = 'auto';
                console.log(`Tab ${layerName} enabled - data available`);
            } else {
                tab.style.opacity = '0.5';
                tab.style.pointerEvents = 'none';
                console.log(`Tab ${layerName} disabled - no data available`);
            }
        });
    }

    showLayerContent(layerName) {
        console.log('Showing layer content for:', layerName);
        console.log('Current feature maps:', this.currentFeatureMaps);

        const layerContent = document.getElementById('layerContent');
        const layerInfo = document.getElementById('layerInfo');

        if (!layerContent || !layerInfo) {
            console.error('Layer content or info elements not found');
            return;
        }

        // Update description
        const desc = this.layerDescriptions[layerName];
        if (desc) {
            if (layerName !== 'overview') {
                const statsHTML = `
                    <div class="layer-stats">
                        <div class="layer-stat">
                            <span class="value">${desc.size}</span>
                            <div class="label">Feature Size</div>
                        </div>
                        <div class="layer-stat">
                            <span class="value">${desc.channels}</span>
                            <div class="label">Channels</div>
                        </div>
                    </div>
                `;
                layerInfo.innerHTML = `<h4>Layer Information</h4><p>${desc.description}</p>${statsHTML}`;
            } else {
                layerInfo.innerHTML = `<h4>Overview</h4><p>${desc.description}</p>`;
            }
        }

        // Update content based on available data
        if (layerName === 'overview' && this.currentFeatureMaps && this.currentFeatureMaps.overview_url) {
            console.log('Loading overview from:', this.currentFeatureMaps.overview_url);
            layerContent.innerHTML = `
                <img src="${this.currentFeatureMaps.overview_url}" 
                     alt="CNN overview" 
                     class="layer-image" 
                     data-modal-src="${this.currentFeatureMaps.overview_url}"
                     style="cursor: pointer;">
            `;
            console.log('[Overview] Image created with data-modal-src attribute');
        } else if (layerName !== 'overview' && this.currentFeatureMaps && this.currentFeatureMaps.individual_layers) {
            console.log('Looking for individual layer:', layerName);
            console.log('Available layers:', Object.keys(this.currentFeatureMaps.individual_layers));

            if (this.currentFeatureMaps.individual_layers[layerName]) {
                const layerData = this.currentFeatureMaps.individual_layers[layerName];
                console.log('Layer data for', layerName, ':', layerData);

                if (layerData.url) {
                    console.log('Loading layer image from:', layerData.url);
                    layerContent.innerHTML = `
                        <img src="${layerData.url}" 
                             alt="${layerName} feature maps" 
                             class="layer-image" 
                             data-modal-src="${layerData.url}"
                             style="cursor: pointer;">
                    `;
                    console.log(`[${layerName}] Image created with data-modal-src attribute`);
                } else {
                    console.log('No URL for layer:', layerName);
                    layerContent.innerHTML = '<div class="layer-placeholder"><p>Feature maps not available for this layer</p></div>';
                }
            } else {
                console.log('Layer not found in individual_layers:', layerName);
                layerContent.innerHTML = '<div class="layer-placeholder"><p>Loading feature maps...</p></div>';
            }
        } else {
            console.log('No data available for layer:', layerName);
            layerContent.innerHTML = '<div class="layer-placeholder"><p>No data available</p></div>';
        }
    }

    // Reset interface
    reset() {
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const progressContainer = document.getElementById('progressContainer');
        const fileInput = document.getElementById('fileInput');
        const layerProgression = document.getElementById('layerProgression');

        if (loading) loading.style.display = 'none';
        if (results) {
            results.style.display = 'none';
            results.classList.remove('show');
        }
        if (progressContainer) progressContainer.style.display = 'none';
        if (fileInput) fileInput.value = '';
        if (layerProgression) layerProgression.style.display = 'none';

        this.currentFeatureMaps = null;

        // Reset layer tabs
        document.querySelectorAll('.layer-tab').forEach(tab => {
            tab.classList.remove('active');
            tab.style.opacity = '1';
            tab.style.pointerEvents = 'auto';
        });

        const overviewTab = document.querySelector('.layer-tab[data-layer="overview"]');
        if (overviewTab) overviewTab.classList.add('active');

        console.log('Interface reset');
    }
}

// Make class available globally
window.ImageNetClassifier = ImageNetClassifier;