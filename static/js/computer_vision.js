/**
 * Computer Vision Analysis Interface - COMPLETE OPTIMIZED VERSION
 * Professional multi-mode analyzer with performance optimizations
 *
 * @class ComputerVisionAnalyzer
 * @version 2.1.0 - Performance Optimized
 * @author CNN Playground
 */

class ComputerVisionAnalyzer {
    /**
     * Initialize the Computer Vision Analyzer with all processing modes
     * @constructor
     */
    constructor() {
        // Current state management
        this.currentMode = 'image';
        this.currentResults = null;
        this.activeTab = 'summary';
        this.isProcessing = false;

        // Video processing state
        this.videoProcessor = null;
        this.currentVideo = null;

        // Live camera state
        this.cameraStream = null;
        this.cameraCanvas = null;
        this.cameraContext = null;
        this.detectionInterval = null;
        this.isDetecting = false;
        this.sessionStartTime = null;

        // PERFORMANCE: Add frame processing throttling
        this.isProcessingFrame = false;
        this.lastFrameTime = 0;
        this.frameSkipCounter = 0;

        // LOCAL MODELS: TensorFlow.js support
        this.localModels = {
            cocoSsd: null,
            isLoading: false,
            isLoaded: false
        };

        this.detectionStats = {
            totalFrames: 0,
            averageLatency: 0,
            objectCount: 0,
            sessionTime: 0
        };

        // Detection debouncing and stability
        this.lastDetections = [];
        this.stableDetections = [];
        this.detectionHistory = [];
        this.maxHistoryLength = 5;
        this.stabilityThreshold = 3;

        // Detection log for live mode
        this.detectionLog = [];
        this.maxLogEntries = 50;

        // Fullscreen state
        this.isFullscreen = false;

        this.initialize();
    }

    /**
     * Initialize all components and event listeners
     * @private
     */
    initialize() {
        this.setupMainTabNavigation();
        this.setupImageMode();
        this.setupVideoMode();
        this.setupLiveMode();
        this.setupAnalysisTabNavigation();

        console.log('[ComputerVision] Professional analyzer initialized with performance optimizations');
    }

    /**
     * Setup main tab navigation (Image, Video, Live)
     * @private
     */
    setupMainTabNavigation() {
        const mainTabs = document.querySelectorAll('.main-tab');

        mainTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabType = tab.getAttribute('data-tab');
                this.switchMainTab(tabType);
            });
        });
    }

    /**
     * Switch between main tabs and reset states
     * @param {string} tabType - The tab to switch to ('image', 'video', 'live')
     * @private
     */
    switchMainTab(tabType) {
        if (this.isProcessing) {
            this.showWarning('Please wait for current operation to complete');
            return;
        }

        // Stop any active processes
        this.stopAllProcesses();

        // Update active tab
        document.querySelectorAll('.main-tab').forEach(tab => {
            tab.classList.remove('active');
        });

        document.querySelector(`[data-tab="${tabType}"]`)?.classList.add('active');

        // Update content panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });

        document.getElementById(`${tabType}-panel`)?.classList.add('active');

        this.currentMode = tabType;
        this.resetState();

        console.log(`[ComputerVision] Switched to ${tabType} mode`);
    }

    /**
     * Setup Image processing mode
     * @private
     */
    setupImageMode() {
        this.setupImageUpload();
        this.setupImageConfiguration();
    }

    /**
     * Setup image upload functionality with drag and drop
     * @private
     */
    setupImageUpload() {
        const fileInput = document.getElementById('fileInput');
        const uploadSection = document.getElementById('uploadSection');

        if (fileInput) {
            fileInput.addEventListener('change', (event) => {
                if (event.target.files.length > 0) {
                    this.handleImageFile(event.target.files[0]);
                }
            });
        }

        if (uploadSection) {
            // Click to upload
            uploadSection.addEventListener('click', () => {
                if (fileInput) fileInput.click();
            });

            // Drag and drop functionality
            this.setupDragAndDrop(uploadSection);
        }
    }

    /**
     * Setup drag and drop functionality for file upload
     * @param {HTMLElement} element - The element to make droppable
     * @private
     */
    setupDragAndDrop(element) {
        element.addEventListener('dragover', (event) => {
            event.preventDefault();
            element.classList.add('dragover');
        });

        element.addEventListener('dragleave', (event) => {
            event.preventDefault();
            element.classList.remove('dragover');
        });

        element.addEventListener('drop', (event) => {
            event.preventDefault();
            element.classList.remove('dragover');

            const files = event.dataTransfer.files;
            if (files.length > 0) {
                if (this.currentMode === 'image') {
                    this.handleImageFile(files[0]);
                } else if (this.currentMode === 'video') {
                    this.handleVideoFile(files[0]);
                }
            }
        });
    }

    /**
     * Setup image processing configuration controls
     * @private
     */
    setupImageConfiguration() {
        const confidenceSlider = document.getElementById('confidenceThreshold');
        const confidenceValue = document.getElementById('confidenceValue');

        if (confidenceSlider && confidenceValue) {
            confidenceSlider.addEventListener('input', (event) => {
                confidenceValue.textContent = event.target.value;
            });
        }
    }

    /**
     * Handle image file processing
     * @param {File} file - The image file to process
     * @private
     */
    async handleImageFile(file) {
        if (this.isProcessing) {
            this.showWarning('Analysis already in progress');
            return;
        }

        try {
            // Validate image file
            this.validateImageFile(file, 10);

            // Get analysis configuration
            const config = this.getImageAnalysisConfig();

            // Show loading state
            this.showImageLoading();

            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('detection_model', config.detectionModel);
            formData.append('segmentation_model', config.segmentationModel);
            formData.append('confidence_threshold', config.confidenceThreshold);

            // Start progress simulation
            const progressInterval = this.simulateImageProgress();

            // Send request
            const response = await fetch('/api/analyze_vision', {
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
            console.log('[ComputerVision] Image analysis results:', data);

            if (data.status === 'success') {
                this.displayImageResults(data, file);
                this.showSuccess('Image analysis completed successfully');
            } else {
                throw new Error(data.message || 'Analysis failed');
            }

        } catch (error) {
            console.error('[ComputerVision] Image analysis error:', error);
            this.hideImageLoading();
            this.showError('Analysis failed: ' + error.message);
        }
    }

    /**
     * Get current image analysis configuration
     * @returns {Object} Configuration object
     * @private
     */
    getImageAnalysisConfig() {
        return {
            detectionModel: document.getElementById('detectionModel')?.value || 'yolo11m',
            segmentationModel: document.getElementById('segmentationModel')?.value || 'yolo11m-seg',
            confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold')?.value || '0.5')
        };
    }

    /**
     * Setup Video processing mode
     * @private
     */
    setupVideoMode() {
        this.setupVideoUpload();
        this.setupVideoConfiguration();
        this.setupVideoControls();
    }

    /**
     * Setup video upload functionality
     * @private
     */
    setupVideoUpload() {
        const videoInput = document.getElementById('videoInput');
        const videoUploadArea = document.querySelector('#video-panel .upload-section');

        if (videoInput) {
            videoInput.addEventListener('change', (event) => {
                if (event.target.files.length > 0) {
                    this.handleVideoFile(event.target.files[0]);
                }
            });
        }

        if (videoUploadArea) {
            videoUploadArea.addEventListener('click', () => {
                if (videoInput) videoInput.click();
            });

            this.setupDragAndDrop(videoUploadArea);
        }
    }

    /**
     * Setup video processing configuration
     * @private
     */
    setupVideoConfiguration() {
        const videoConfidenceSlider = document.getElementById('videoConfidence');
        const videoConfidenceValue = document.getElementById('videoConfidenceValue');

        if (videoConfidenceSlider && videoConfidenceValue) {
            videoConfidenceSlider.addEventListener('input', (event) => {
                videoConfidenceValue.textContent = event.target.value;
            });
        }
    }

    /**
     * Setup video processing controls
     * @private
     */
    setupVideoControls() {
        const processBtn = document.getElementById('processVideoBtn');
        const downloadBtn = document.getElementById('downloadVideoBtn');
        const cancelBtn = document.getElementById('cancelVideoBtn');

        if (processBtn) {
            processBtn.addEventListener('click', () => {
                this.processCurrentVideo();
            });
        }

        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadProcessedVideo();
            });
        }

        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
                this.cancelVideoProcessing();
            });
        }
    }

    /**
     * Handle video file selection
     * @param {File} file - The video file to process
     * @private
     */
    handleVideoFile(file) {
        try {
            this.validateVideoFile(file, 100);
            this.currentVideo = file;

            // Display video preview
            this.displayVideoPreview(file);

            this.showSuccess(`Video loaded: ${file.name}`);
            console.log('[ComputerVision] Video file loaded:', file.name);
        } catch (error) {
            this.showError('Video loading failed: ' + error.message);
        }
    }

    /**
     * Display video preview
     * @param {File} videoFile - The video file to preview
     * @private
     */
    displayVideoPreview(videoFile) {
        const originalVideo = document.getElementById('originalVideo');
        if (originalVideo) {
            originalVideo.src = URL.createObjectURL(videoFile);

            // Show video results container
            const videoResults = document.getElementById('videoResults');
            if (videoResults) {
                videoResults.style.display = 'block';
            }
        }
    }

    /**
     * Process the currently loaded video
     * @private
     */
    async processCurrentVideo() {
        if (!this.currentVideo) {
            this.showWarning('Please upload a video file first');
            return;
        }

        if (this.isProcessing) {
            this.showWarning('Video processing already in progress');
            return;
        }

        try {
            this.isProcessing = true;
            this.showVideoProgress();

            const config = this.getVideoProcessingConfig();

            // Create form data
            const formData = new FormData();
            formData.append('video', this.currentVideo);
            formData.append('detection_model', config.detectionModel);
            formData.append('frame_skip', config.frameSkip);
            formData.append('confidence_threshold', config.confidenceThreshold);

            // Simulate video processing progress
            const progressInterval = this.simulateVideoProgress();

            // Send request
            const response = await fetch('/api/process_video', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.status === 'success') {
                this.displayVideoResults(data);
                this.showSuccess('Video processing completed successfully');
            } else {
                throw new Error(data.message || 'Video processing failed');
            }

        } catch (error) {
            console.error('[ComputerVision] Video processing error:', error);
            this.showError('Video processing failed: ' + error.message);
        } finally {
            this.isProcessing = false;
            this.hideVideoProgress();
        }
    }

    /**
     * Get video processing configuration
     * @returns {Object} Video processing configuration
     * @private
     */
    getVideoProcessingConfig() {
        return {
            detectionModel: document.getElementById('videoDetectionModel')?.value || 'yolo11n',
            frameSkip: parseInt(document.getElementById('frameSkip')?.value || '2'),
            confidenceThreshold: parseFloat(document.getElementById('videoConfidence')?.value || '0.4')
        };
    }

    /**
     * Setup Live camera mode
     * @private
     */
    setupLiveMode() {
        this.setupLiveConfiguration();
        this.setupLiveControls();
        this.initializeLiveCanvas();
    }

    /**
     * Setup live camera configuration
     * @private
     */
    setupLiveConfiguration() {
        const liveConfidenceSlider = document.getElementById('liveConfidence');
        const liveConfidenceValue = document.getElementById('liveConfidenceValue');

        if (liveConfidenceSlider && liveConfidenceValue) {
            liveConfidenceSlider.addEventListener('input', (event) => {
                liveConfidenceValue.textContent = event.target.value;
            });
        }

        // Setup model type change handler
        const liveDetectionModel = document.getElementById('liveDetectionModel');
        if (liveDetectionModel) {
            // Initialize FPS options based on default model
            this.updateFpsOptions(liveDetectionModel.value);

            liveDetectionModel.addEventListener('change', (event) => {
                this.handleModelTypeChange(event.target.value);
            });
        }

        // Add FPS change indicator
        const processingFpsSelect = document.getElementById('processingFps');
        if (processingFpsSelect) {
            processingFpsSelect.addEventListener('change', (event) => {
                const fps = event.target.value;
                const isLocal = this.isUsingLocalModel();
                if (isLocal && parseInt(fps) >= 20) {
                    this.showSuccess(`Ultra-smooth ${fps} FPS mode activated! 🚀`);
                } else if (isLocal && parseInt(fps) >= 10) {
                    this.showSuccess(`High-performance ${fps} FPS mode enabled!`);
                }
            });
        }
    }

    /**
     * Handle model type change (local vs server)
     * @param {string} modelValue - Selected model value
     * @private
     */
    handleModelTypeChange(modelValue) {
        // Update FPS options based on model type
        this.updateFpsOptions(modelValue);

        if (modelValue.startsWith('local_')) {
            this.loadLocalModel(modelValue);
        }
    }

    /**
     * Update FPS options based on selected model type
     * @param {string} modelValue - Selected model value
     * @private
     */
    updateFpsOptions(modelValue) {
        const processingFpsSelect = document.getElementById('processingFps');
        if (!processingFpsSelect) return;

        const isLocal = modelValue.startsWith('local_');

        // Clear existing options
        processingFpsSelect.innerHTML = '';

        if (isLocal) {
            // High FPS options for local models
            const localOptions = [
                { value: '5', text: '5 FPS (Balanced)' },
                { value: '10', text: '10 FPS (Smooth)' },
                { value: '15', text: '15 FPS (High Performance)' },
                { value: '20', text: '20 FPS (Ultra Smooth)' },
                { value: '30', text: '30 FPS (Maximum)' }
            ];

            localOptions.forEach(option => {
                const optionEl = document.createElement('option');
                optionEl.value = option.value;
                optionEl.textContent = option.text;
                if (option.value === '15') optionEl.selected = true; // Default to 15 FPS for local
                processingFpsSelect.appendChild(optionEl);
            });

            this.showSuccess('Local model selected - High FPS modes available!');
        } else {
            // Conservative FPS options for server models
            const serverOptions = [
                { value: '1', text: '1 FPS (Battery Saving)' },
                { value: '2', text: '2 FPS (Conservative)' },
                { value: '3', text: '3 FPS (Balanced)' },
                { value: '5', text: '5 FPS (Maximum for Server)' }
            ];

            serverOptions.forEach(option => {
                const optionEl = document.createElement('option');
                optionEl.value = option.value;
                optionEl.textContent = option.text;
                if (option.value === '3') optionEl.selected = true; // Default to 3 FPS for server
                processingFpsSelect.appendChild(optionEl);
            });
        }

        console.log(`[ComputerVision] Updated FPS options for ${isLocal ? 'local' : 'server'} model`);
    }

    /**
     * Load TensorFlow.js local model
     * @param {string} modelType - Type of local model
     * @private
     */
    async loadLocalModel(modelType) {
        if (this.localModels.isLoading) {
            this.showWarning('Local model is already loading...');
            return;
        }

        if (this.localModels.isLoaded && modelType === 'local_coco_ssd') {
            console.log('[ComputerVision] Local COCO-SSD model already loaded');
            return;
        }

        try {
            this.localModels.isLoading = true;
            this.showSuccess('Loading local TensorFlow.js model...');

            // Load TensorFlow.js and COCO-SSD
            if (!window.tf) {
                const tfScript = document.createElement('script');
                tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest';
                document.head.appendChild(tfScript);
                await new Promise(resolve => tfScript.onload = resolve);
            }

            if (!window.cocoSsd) {
                const cocoScript = document.createElement('script');
                cocoScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@latest';
                document.head.appendChild(cocoScript);
                await new Promise(resolve => cocoScript.onload = resolve);
            }

            // Load the model
            console.log('[ComputerVision] Loading COCO-SSD model...');
            this.localModels.cocoSsd = await cocoSsd.load();
            this.localModels.isLoaded = true;

            this.showSuccess('Local TensorFlow.js model loaded successfully! 🚀');
            console.log('[ComputerVision] COCO-SSD model loaded successfully');

        } catch (error) {
            console.error('[ComputerVision] Error loading local model:', error);
            this.showError('Failed to load local model: ' + error.message);
        } finally {
            this.localModels.isLoading = false;
        }
    }

    /**
     * Perform local object detection using TensorFlow.js
     * @param {HTMLCanvasElement} canvas - Canvas with image data
     * @returns {Promise<Array>} Detection results
     * @private
     */
    async performLocalDetection(canvas) {
        if (!this.localModels.cocoSsd) {
            throw new Error('Local model not loaded');
        }

        try {
            const predictions = await this.localModels.cocoSsd.detect(canvas);

            // Convert to our format
            return predictions.map(prediction => ({
                class_name: prediction.class,
                confidence: prediction.score,
                bbox: [
                    prediction.bbox[0], // x1
                    prediction.bbox[1], // y1
                    prediction.bbox[0] + prediction.bbox[2], // x2
                    prediction.bbox[1] + prediction.bbox[3]  // y2
                ]
            }));

        } catch (error) {
            console.error('[ComputerVision] Local detection error:', error);
            throw error;
        }
    }

    /**
     * Check if current model is local
     * @returns {boolean} True if using local model
     * @private
     */
    isUsingLocalModel() {
        const modelValue = document.getElementById('liveDetectionModel')?.value || '';
        return modelValue.startsWith('local_');
    }

    /**
     * Setup live camera controls
     * @private
     */
    setupLiveControls() {
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const captureBtn = document.getElementById('captureFrameBtn');
        const toggleBtn = document.getElementById('toggleDetectionBtn');

        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startCamera();
            });
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                this.stopCamera();
            });
        }

        if (captureBtn) {
            captureBtn.addEventListener('click', () => {
                this.captureCurrentFrame();
            });
        }

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.toggleDetection();
            });
        }

        // Add fullscreen toggle button
        this.setupFullscreenControls();
    }

    /**
     * Setup fullscreen controls for camera
     * @private
     */
    setupFullscreenControls() {
        const cameraDisplay = document.querySelector('.camera-display');
        if (!cameraDisplay) return;

        // Add fullscreen button
        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.className = 'btn fullscreen-btn';
        fullscreenBtn.innerHTML = 'Fullscreen';
        fullscreenBtn.style.position = 'absolute';
        fullscreenBtn.style.top = '10px';
        fullscreenBtn.style.right = '10px';
        fullscreenBtn.style.display = 'none';
        fullscreenBtn.style.zIndex = '1000';

        cameraDisplay.appendChild(fullscreenBtn);

        fullscreenBtn.addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Show/hide fullscreen button when camera is active
        this.fullscreenBtn = fullscreenBtn;
    }

    /**
     * Toggle fullscreen mode for camera - FIXED VERSION
     * @private
     */
    toggleFullscreen() {
        if (!this.isFullscreen) {
            this.enterFullscreen();
        } else {
            this.exitFullscreen();
        }
    }

    /**
     * Enter fullscreen mode with proper overlay
     * @private
     */
    enterFullscreen() {
        // Create fullscreen overlay
        const fullscreenOverlay = document.createElement('div');
        fullscreenOverlay.className = 'fullscreen-overlay';
        fullscreenOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.95);
            z-index: 10000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            box-sizing: border-box;
            overflow: hidden;
        `;

        // Create video container
        const videoContainer = document.createElement('div');
        videoContainer.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            max-width: 100vw;
            max-height: 85vh;
            position: relative;
        `;

        // Clone and setup canvas for fullscreen
        const originalCanvas = this.cameraCanvas;
        const fullscreenCanvas = originalCanvas.cloneNode(true);
        fullscreenCanvas.id = 'fullscreenCanvas';
        fullscreenCanvas.className = 'fullscreen-canvas';
        fullscreenCanvas.style.cssText = `
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            display: block;
        `;

        // Create close button
        const closeBtn = document.createElement('button');
        closeBtn.className = 'fullscreen-close-btn';
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            position: absolute;
            top: 30px;
            right: 30px;
            background: rgba(255, 255, 255, 0.9);
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            color: #333;
            z-index: 10001;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        `;

        closeBtn.addEventListener('click', () => this.exitFullscreen());
        closeBtn.addEventListener('mouseenter', () => {
            closeBtn.style.background = 'white';
            closeBtn.style.transform = 'scale(1.1)';
        });
        closeBtn.addEventListener('mouseleave', () => {
            closeBtn.style.background = 'rgba(255, 255, 255, 0.9)';
            closeBtn.style.transform = 'scale(1)';
        });

        // Create info overlay
        const infoOverlay = document.createElement('div');
        infoOverlay.className = 'fullscreen-info';
        infoOverlay.style.cssText = `
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 25px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            z-index: 10001;
        `;

        const statsHtml = `
            <div style="display: flex; gap: 30px; color: white; font-size: 14px; font-weight: 500;">
                <span style="display: flex; align-items: center; gap: 8px;">
                    <span>Objects:</span>
                    <strong style="color: #00ff00;">${this.detectionStats.objectCount}</strong>
                </span>
                <span style="display: flex; align-items: center; gap: 8px;">
                    <span>FPS:</span>
                    <strong style="color: #00ff00;">${parseInt(document.getElementById('processingFps')?.value || '3')}</strong>
                </span>
                <span style="display: flex; align-items: center; gap: 8px;">
                    <span>Latency:</span>
                    <strong style="color: #00ff00;">${Math.round(this.detectionStats.averageLatency)}ms</strong>
                </span>
            </div>
        `;
        infoOverlay.innerHTML = statsHtml;

        // Assemble fullscreen layout
        videoContainer.appendChild(fullscreenCanvas);
        fullscreenOverlay.appendChild(videoContainer);
        fullscreenOverlay.appendChild(closeBtn);
        fullscreenOverlay.appendChild(infoOverlay);

        // Add to document
        document.body.appendChild(fullscreenOverlay);

        // Prevent body scrolling
        document.body.style.overflow = 'hidden';
        document.body.classList.add('fullscreen-active');

        // Setup canvas context for fullscreen
        const fullscreenContext = fullscreenCanvas.getContext('2d');
        this.fullscreenCanvas = fullscreenCanvas;
        this.fullscreenContext = fullscreenContext;
        this.fullscreenOverlay = fullscreenOverlay;

        // Update button text
        if (this.fullscreenBtn) {
            this.fullscreenBtn.innerHTML = 'Exit Fullscreen';
        }

        this.isFullscreen = true;

        // Handle escape key
        this.handleEscapeKey = (e) => {
            if (e.key === 'Escape') {
                this.exitFullscreen();
            }
        };
        document.addEventListener('keydown', this.handleEscapeKey);

        console.log('[ComputerVision] Entered fullscreen mode');
    }

    /**
     * Exit fullscreen mode
     * @private
     */
    exitFullscreen() {
        // Remove fullscreen overlay
        if (this.fullscreenOverlay) {
            document.body.removeChild(this.fullscreenOverlay);
            this.fullscreenOverlay = null;
        }

        // Restore body scrolling
        document.body.style.overflow = '';
        document.body.classList.remove('fullscreen-active');

        // Clean up fullscreen canvas references
        this.fullscreenCanvas = null;
        this.fullscreenContext = null;

        // Update button text
        if (this.fullscreenBtn) {
            this.fullscreenBtn.innerHTML = 'Fullscreen';
        }

        this.isFullscreen = false;

        // Remove escape key handler
        if (this.handleEscapeKey) {
            document.removeEventListener('keydown', this.handleEscapeKey);
            this.handleEscapeKey = null;
        }

        console.log('[ComputerVision] Exited fullscreen mode');
    }

    /**
     * Resize canvas for fullscreen mode
     * @private
     */
    resizeCanvasToFullscreen() {
        if (this.cameraCanvas && this.cameraStream) {
            const video = document.getElementById('liveVideo');
            if (video) {
                // Keep aspect ratio
                const videoAspect = video.videoWidth / video.videoHeight;
                const screenAspect = window.innerWidth / window.innerHeight;

                let canvasWidth, canvasHeight;

                if (videoAspect > screenAspect) {
                    canvasWidth = window.innerWidth;
                    canvasHeight = window.innerWidth / videoAspect;
                } else {
                    canvasHeight = window.innerHeight;
                    canvasWidth = window.innerHeight * videoAspect;
                }

                this.cameraCanvas.style.width = canvasWidth + 'px';
                this.cameraCanvas.style.height = canvasHeight + 'px';
            }
        }
    }

    /**
     * Resize canvas back to normal size
     * @private
     */
    resizeCanvasToNormal() {
        if (this.cameraCanvas) {
            this.cameraCanvas.style.width = '';
            this.cameraCanvas.style.height = '';
        }
    }

    /**
     * Initialize canvas for live camera processing
     * @private
     */
    initializeLiveCanvas() {
        this.cameraCanvas = document.getElementById('liveCanvas');
        if (this.cameraCanvas) {
            this.cameraContext = this.cameraCanvas.getContext('2d');
        }
    }

    /**
     * Start camera stream and detection
     * @private
     */
    async startCamera() {
        try {
            // Request camera permissions
            this.cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            const video = document.getElementById('liveVideo');
            const placeholder = document.getElementById('cameraPlaceholder');

            if (video && this.cameraStream) {
                video.srcObject = this.cameraStream;
                video.style.display = 'none'; // Hide video, show only canvas

                if (placeholder) {
                    placeholder.style.display = 'none';
                }

                // Update controls
                this.updateLiveControls(true);

                // Setup canvas for processing
                this.setupLiveProcessing(video);

                // Start detection with improved stability
                this.startDetection();

                this.showSuccess('Camera started successfully');
                console.log('[ComputerVision] Camera stream started');
            }

        } catch (error) {
            console.error('[ComputerVision] Camera start error:', error);
            this.showError('Failed to start camera: ' + error.message);
        }
    }

    /**
     * Setup live video processing
     * @param {HTMLVideoElement} video - The video element
     * @private
     */
    setupLiveProcessing(video) {
        video.addEventListener('loadedmetadata', () => {
            if (this.cameraCanvas && this.cameraContext) {
                this.cameraCanvas.width = video.videoWidth;
                this.cameraCanvas.height = video.videoHeight;
                this.cameraCanvas.style.display = 'block';
            }
        });
    }

    /**
     * Start detection processing loop with improved stability and performance
     * @private
     */
    startDetection() {
        if (this.isDetecting) return;

        this.isDetecting = true;
        this.isProcessingFrame = false; // Reset processing flag
        this.sessionStartTime = Date.now();
        this.lastFrameTime = 0;
        this.frameSkipCounter = 0;

        this.detectionStats = {
            totalFrames: 0,
            averageLatency: 0,
            objectCount: 0,
            sessionTime: 0
        };

        // Reset detection stability tracking
        this.lastDetections = [];
        this.stableDetections = [];
        this.detectionHistory = [];

        // FIXED: Get processing FPS from user selection
        const processingFps = parseInt(document.getElementById('processingFps')?.value || '3');
        const intervalMs = 1000 / processingFps;

        this.detectionInterval = setInterval(() => {
            if (this.isDetecting) {
                this.processLiveFrame();
            }
        }, intervalMs);

        // Start stats update
        this.startStatsUpdate();

        // Show live stats and fullscreen button
        const liveStats = document.getElementById('liveStats');
        if (liveStats) {
            liveStats.style.display = 'block';
        }

        if (this.fullscreenBtn) {
            this.fullscreenBtn.style.display = 'block';
        }

        console.log('[ComputerVision] Detection started at', processingFps, 'FPS with performance optimizations');
    }

    /**
     * OPTIMIZED: Process a single frame with throttling, resolution reduction, and LOCAL MODEL support
     * @private
     */
    async processLiveFrame() {
        // THROTTLING: Skip if already processing a frame
        if (this.isProcessingFrame || !this.cameraCanvas || !this.cameraContext || !this.isDetecting) {
            return;
        }

        // ADAPTIVE FPS: Different throttling for local vs server models
        const now = Date.now();
        const timeSinceLastFrame = now - this.lastFrameTime;

        let minInterval;
        if (this.isUsingLocalModel()) {
            // More aggressive FPS for local models - minimal throttling
            minInterval = Math.max(33, this.detectionStats.averageLatency * 1.1); // At least 30 FPS or 1.1x latency
        } else {
            // Conservative throttling for server models
            minInterval = Math.max(200, this.detectionStats.averageLatency * 1.2); // At least 200ms or 1.2x latency
        }

        if (timeSinceLastFrame < minInterval) {
            return; // Skip this frame to prevent overload
        }

        this.isProcessingFrame = true;
        this.lastFrameTime = now;
        const startTime = Date.now();

        try {
            const video = document.getElementById('liveVideo');
            if (!video || video.videoWidth === 0) {
                this.isProcessingFrame = false;
                return;
            }

            // Always update visual canvas with current frame
            this.cameraContext.drawImage(video, 0, 0, this.cameraCanvas.width, this.cameraCanvas.height);

            // Check if using local model
            if (this.isUsingLocalModel()) {
                await this.processFrameLocally(video, startTime);
            } else {
                await this.processFrameOnServer(video, startTime);
            }

        } catch (error) {
            console.error('[ComputerVision] Live frame processing error:', error);
            this.isProcessingFrame = false;
        }
    }

    /**
     * Process frame using local TensorFlow.js model
     * @param {HTMLVideoElement} video - Video element
     * @param {number} startTime - Processing start time
     * @private
     */
    async processFrameLocally(video, startTime) {
        try {
            if (!this.localModels.cocoSsd) {
                console.warn('[ComputerVision] Local model not loaded');
                this.isProcessingFrame = false;
                return;
            }

            // Get confidence threshold
            const confidenceThreshold = parseFloat(document.getElementById('liveConfidence')?.value || '0.3');

            // Create temporary canvas for detection - use full resolution for local processing
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');

            // For local models, we can afford higher resolution since processing is local
            const scaleFactor = Math.min(640 / video.videoWidth, 480 / video.videoHeight);
            tempCanvas.width = Math.floor(video.videoWidth * scaleFactor);
            tempCanvas.height = Math.floor(video.videoHeight * scaleFactor);
            tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

            // Perform local detection
            const predictions = await this.performLocalDetection(tempCanvas);

            // Filter by confidence
            const filteredPredictions = predictions.filter(pred => pred.confidence >= confidenceThreshold);

            // Scale detections back to canvas size
            const scaledDetections = this.scaleDetections(filteredPredictions,
                tempCanvas.width, tempCanvas.height,
                this.cameraCanvas.width, this.cameraCanvas.height);

            const latency = Date.now() - startTime;

            // Process detections with stability filtering
            this.processStableDetections({ objects: scaledDetections }, latency);

            console.log(`[ComputerVision] Local detection: ${scaledDetections.length} objects in ${latency}ms`);

        } catch (error) {
            console.error('[ComputerVision] Local detection error:', error);
        } finally {
            this.isProcessingFrame = false;
        }
    }

    /**
     * Process frame on server (original method)
     * @param {HTMLVideoElement} video - Video element
     * @param {number} startTime - Processing start time
     * @private
     */
    async processFrameOnServer(video, startTime) {
        try {
            // RESOLUTION REDUCTION: Create smaller canvas for processing
            const processCanvas = document.createElement('canvas');
            const processCtx = processCanvas.getContext('2d');

            // Reduce to 480x360 for processing (keeps aspect ratio)
            const scaleFactor = Math.min(480 / video.videoWidth, 360 / video.videoHeight);
            processCanvas.width = Math.floor(video.videoWidth * scaleFactor);
            processCanvas.height = Math.floor(video.videoHeight * scaleFactor);

            // Draw scaled-down frame
            processCtx.drawImage(video, 0, 0, processCanvas.width, processCanvas.height);

            // Convert to blob with lower quality
            processCanvas.toBlob(async (blob) => {
                if (!blob || !this.isDetecting) {
                    this.isProcessingFrame = false;
                    return;
                }

                try {
                    const config = this.getLiveDetectionConfig();

                    // Create form data
                    const formData = new FormData();
                    formData.append('frame', blob);
                    formData.append('detection_model', config.detectionModel);
                    formData.append('confidence_threshold', config.confidenceThreshold);

                    // Send frame for processing with timeout
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

                    const response = await fetch('/api/process_live_frame', {
                        method: 'POST',
                        body: formData,
                        signal: controller.signal
                    });

                    clearTimeout(timeoutId);

                    if (response.ok) {
                        const data = await response.json();
                        const latency = Date.now() - startTime;

                        // Scale detections back to original canvas size
                        const scaledDetections = this.scaleDetections(data.objects || [],
                            processCanvas.width, processCanvas.height,
                            this.cameraCanvas.width, this.cameraCanvas.height);

                        // Process detections with stability filtering
                        this.processStableDetections({ objects: scaledDetections }, latency);
                    }

                } catch (error) {
                    if (error.name !== 'AbortError') {
                        console.error('[ComputerVision] Frame processing error:', error);
                    }
                } finally {
                    this.isProcessingFrame = false;
                }
            }, 'image/jpeg', 0.6); // Lower quality for faster upload

        } catch (error) {
            console.error('[ComputerVision] Server processing error:', error);
            this.isProcessingFrame = false;
        }
    }

    /**
     * Scale detection coordinates from processed image back to canvas size
     * @param {Array} detections - Array of detection objects
     * @param {number} fromWidth - Source image width
     * @param {number} fromHeight - Source image height
     * @param {number} toWidth - Target canvas width
     * @param {number} toHeight - Target canvas height
     * @returns {Array} Scaled detection objects
     * @private
     */
    scaleDetections(detections, fromWidth, fromHeight, toWidth, toHeight) {
        const scaleX = toWidth / fromWidth;
        const scaleY = toHeight / fromHeight;

        return detections.map(detection => ({
            ...detection,
            bbox: [
                detection.bbox[0] * scaleX,
                detection.bbox[1] * scaleY,
                detection.bbox[2] * scaleX,
                detection.bbox[3] * scaleY
            ]
        }));
    }

    /**
     * Process detections with stability filtering to reduce flickering
     * @param {Object} detectionData - Detection results
     * @param {number} latency - Processing latency
     * @private
     */
    processStableDetections(detectionData, latency) {
        if (!detectionData.objects || !this.cameraContext) return;

        // Add current detections to history
        this.detectionHistory.push(detectionData.objects);

        // Keep only recent history
        if (this.detectionHistory.length > this.maxHistoryLength) {
            this.detectionHistory.shift();
        }

        // Filter stable detections
        const stableObjects = this.filterStableDetections();

        // Update display with stable detections
        this.updateLiveDetections(stableObjects, latency);
        this.updateDetectionStats({ objects: stableObjects }, latency);
    }

    /**
     * Filter detections for stability (reduce flickering)
     * @returns {Array} Stable detection objects
     * @private
     */
    filterStableDetections() {
        if (this.detectionHistory.length < 2) {
            return this.detectionHistory[0] || [];
        }

        const stableObjects = [];
        const recentDetections = this.detectionHistory.slice(-3); // Last 3 frames

        // Group similar detections across frames
        const allDetections = recentDetections.flat();
        const detectionGroups = {};

        allDetections.forEach(detection => {
            const key = `${detection.class_name}_${Math.round(detection.bbox[0] / 50)}_${Math.round(detection.bbox[1] / 50)}`;

            if (!detectionGroups[key]) {
                detectionGroups[key] = [];
            }
            detectionGroups[key].push(detection);
        });

        // Only keep detections that appear in multiple recent frames
        Object.values(detectionGroups).forEach(group => {
            if (group.length >= 2) { // Appears in at least 2 recent frames
                // Use the detection with highest confidence
                const bestDetection = group.reduce((best, current) =>
                    current.confidence > best.confidence ? current : best
                );
                stableObjects.push(bestDetection);
            }
        });

        return stableObjects;
    }

    /**
     * Get live detection configuration
     * @returns {Object} Live detection configuration
     * @private
     */
    getLiveDetectionConfig() {
        return {
            detectionModel: document.getElementById('liveDetectionModel')?.value || 'yolo11n',
            confidenceThreshold: parseFloat(document.getElementById('liveConfidence')?.value || '0.3')
        };
    }

    /**
     * Update live detection display with stable detections
     * @param {Array} objects - Stable detection objects
     * @param {number} latency - Processing latency
     * @private
     */
    updateLiveDetections(objects, latency) {
        if (!this.cameraContext) return;

        // Redraw the current video frame first
        const video = document.getElementById('liveVideo');
        if (video && video.videoWidth > 0) {
            this.cameraContext.drawImage(video, 0, 0, this.cameraCanvas.width, this.cameraCanvas.height);
        }

        // Draw stable detection boxes
        objects.forEach((obj, index) => {
            this.drawDetectionBox(obj, index);
        });

        // Update fullscreen canvas if active
        if (this.isFullscreen && this.fullscreenCanvas && this.fullscreenContext) {
            this.updateFullscreenCanvas(objects);
        }

        // Update detection log with stable detections
        if (objects.length > 0) {
            this.updateDetectionLog(objects);
        }
    }

    /**
     * Update fullscreen canvas with detections
     * @param {Array} objects - Detection objects
     * @private
     */
    updateFullscreenCanvas(objects) {
        const video = document.getElementById('liveVideo');
        if (!video || video.videoWidth === 0 || !this.fullscreenContext) return;

        // Set canvas size to match video
        this.fullscreenCanvas.width = video.videoWidth;
        this.fullscreenCanvas.height = video.videoHeight;

        // Draw video frame
        this.fullscreenContext.drawImage(video, 0, 0, this.fullscreenCanvas.width, this.fullscreenCanvas.height);

        // Draw detection boxes
        objects.forEach((obj, index) => {
            this.drawDetectionBoxOnCanvas(this.fullscreenContext, obj, index);
        });

        // Update fullscreen stats
        this.updateFullscreenStats();
    }

    /**
     * Draw detection box on specific canvas context
     * @param {CanvasRenderingContext2D} context - Canvas context
     * @param {Object} detection - Detection object
     * @param {number} index - Detection index for color
     * @private
     */
    drawDetectionBoxOnCanvas(context, detection, index) {
        const [x1, y1, x2, y2] = detection.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Color palette for different detections
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'];
        const color = colors[index % colors.length];

        // Draw bounding box with thicker lines for better visibility
        context.strokeStyle = color;
        context.lineWidth = 4;
        context.strokeRect(x1, y1, width, height);

        // Draw label background
        const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`;
        context.font = 'bold 18px Arial';
        const textMetrics = context.measureText(label);
        const textWidth = textMetrics.width;

        context.fillStyle = color;
        context.fillRect(x1, y1 - 30, textWidth + 12, 30);

        // Draw label text
        context.fillStyle = 'white';
        context.fillText(label, x1 + 6, y1 - 8);
    }

    /**
     * Update fullscreen statistics overlay
     * @private
     */
    updateFullscreenStats() {
        if (!this.isFullscreen) return;

        const infoOverlay = document.querySelector('.fullscreen-info');
        if (infoOverlay) {
            const statsHtml = `
                <div style="display: flex; gap: 30px; color: white; font-size: 14px; font-weight: 500;">
                    <span style="display: flex; align-items: center; gap: 8px;">
                        <span>Objects:</span>
                        <strong style="color: #00ff00;">${this.detectionStats.objectCount}</strong>
                    </span>
                    <span style="display: flex; align-items: center; gap: 8px;">
                        <span>FPS:</span>
                        <strong style="color: #00ff00;">${parseInt(document.getElementById('processingFps')?.value || '3')}</strong>
                    </span>
                    <span style="display: flex; align-items: center; gap: 8px;">
                        <span>Latency:</span>
                        <strong style="color: #00ff00;">${Math.round(this.detectionStats.averageLatency)}ms</strong>
                    </span>
                </div>
            `;
            infoOverlay.innerHTML = statsHtml;
        }
    }

    /**
     * Draw detection bounding box on canvas
     * @param {Object} detection - Detection object
     * @param {number} index - Detection index for color
     * @private
     */
    drawDetectionBox(detection, index) {
        if (!this.cameraContext) return;

        const [x1, y1, x2, y2] = detection.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Color palette for different detections
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'];
        const color = colors[index % colors.length];

        // Draw bounding box with thicker lines for better visibility
        this.cameraContext.strokeStyle = color;
        this.cameraContext.lineWidth = 4;
        this.cameraContext.strokeRect(x1, y1, width, height);

        // Draw label background
        const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`;
        this.cameraContext.font = 'bold 18px Arial';
        const textMetrics = this.cameraContext.measureText(label);
        const textWidth = textMetrics.width;

        this.cameraContext.fillStyle = color;
        this.cameraContext.fillRect(x1, y1 - 30, textWidth + 12, 30);

        // Draw label text
        this.cameraContext.fillStyle = 'white';
        this.cameraContext.fillText(label, x1 + 6, y1 - 8);
    }

    /**
     * Update detection statistics
     * @param {Object} detectionData - Detection results
     * @param {number} latency - Processing latency
     * @private
     */
    updateDetectionStats(detectionData, latency) {
        this.detectionStats.totalFrames++;
        this.detectionStats.averageLatency = (
            (this.detectionStats.averageLatency * (this.detectionStats.totalFrames - 1) + latency) /
            this.detectionStats.totalFrames
        );
        this.detectionStats.objectCount = detectionData.objects?.length || 0;
    }

    /**
     * Start statistics update loop
     * @private
     */
    startStatsUpdate() {
        setInterval(() => {
            if (!this.isDetecting || !this.sessionStartTime) return;

            const sessionTime = Math.floor((Date.now() - this.sessionStartTime) / 1000);
            this.detectionStats.sessionTime = sessionTime;

            this.updateLiveStatsDisplay();
        }, 1000);
    }

    /**
     * FIXED: Update live statistics display with correct FPS and model info
     * @private
     */
    updateLiveStatsDisplay() {
        const objectCountEl = document.getElementById('liveObjectCount');
        const fpsEl = document.getElementById('liveFps');
        const latencyEl = document.getElementById('liveLatency');
        const uptimeEl = document.getElementById('liveUptime');

        if (objectCountEl) objectCountEl.textContent = this.detectionStats.objectCount;
        if (latencyEl) latencyEl.textContent = Math.round(this.detectionStats.averageLatency) + 'ms';

        if (fpsEl) {
            // Show selected processing rate, not calculated FPS
            const selectedFps = parseInt(document.getElementById('processingFps')?.value || '3');
            const modelType = this.isUsingLocalModel() ? ' (Local)' : ' (Server)';
            fpsEl.textContent = selectedFps + modelType;
        }

        if (uptimeEl) {
            const minutes = Math.floor(this.detectionStats.sessionTime / 60);
            const seconds = this.detectionStats.sessionTime % 60;
            uptimeEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    /**
     * Update detection log with recent detections
     * @param {Array} detections - Array of detection objects
     * @private
     */
    updateDetectionLog(detections) {
        const timestamp = new Date().toLocaleTimeString();

        detections.forEach(detection => {
            this.detectionLog.unshift({
                timestamp,
                class_name: detection.class_name,
                confidence: detection.confidence
            });
        });

        // Keep only recent entries
        if (this.detectionLog.length > this.maxLogEntries) {
            this.detectionLog = this.detectionLog.slice(0, this.maxLogEntries);
        }

        this.updateDetectionLogDisplay();
    }

    /**
     * Update detection log display
     * @private
     */
    updateDetectionLogDisplay() {
        const logList = document.getElementById('detectionLogList');
        if (!logList) return;

        if (this.detectionLog.length === 0) {
            logList.innerHTML = '<p>No detections yet...</p>';
            return;
        }

        const recentDetections = this.detectionLog.slice(0, 10);
        logList.innerHTML = recentDetections.map(detection => `
            <div class="detection-log-item" style="
                padding: 5px 10px;
                margin-bottom: 3px;
                background: var(--bg-secondary);
                border-radius: 4px;
                font-size: 0.9em;
                display: flex;
                justify-content: space-between;
            ">
                <span>${detection.timestamp}</span>
                <span>${detection.class_name}</span>
                <span>${(detection.confidence * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }

    /**
     * Stop camera and detection
     * @private
     */
    stopCamera() {
        // Stop detection
        this.stopDetection();

        // Stop camera stream
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => {
                track.stop();
            });
            this.cameraStream = null;
        }

        // Exit fullscreen if active
        if (this.isFullscreen) {
            this.toggleFullscreen();
        }

        // Hide video elements
        const video = document.getElementById('liveVideo');
        const canvas = document.getElementById('liveCanvas');
        const placeholder = document.getElementById('cameraPlaceholder');
        const liveStats = document.getElementById('liveStats');

        if (video) video.style.display = 'none';
        if (canvas) canvas.style.display = 'none';
        if (placeholder) placeholder.style.display = 'block';
        if (liveStats) liveStats.style.display = 'none';

        // Hide fullscreen button
        if (this.fullscreenBtn) {
            this.fullscreenBtn.style.display = 'none';
        }

        // Update controls
        this.updateLiveControls(false);

        this.showSuccess('Camera stopped');
        console.log('[ComputerVision] Camera stopped');
    }

    /**
     * Stop detection processing
     * @private
     */
    stopDetection() {
        this.isDetecting = false;
        this.isProcessingFrame = false; // Reset processing flag

        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }

        // Reset detection history
        this.detectionHistory = [];
        this.lastDetections = [];
        this.stableDetections = [];
    }

    /**
     * Update live camera control buttons
     * @param {boolean} cameraActive - Whether camera is active
     * @private
     */
    updateLiveControls(cameraActive) {
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const captureBtn = document.getElementById('captureFrameBtn');
        const toggleBtn = document.getElementById('toggleDetectionBtn');

        if (startBtn) startBtn.style.display = cameraActive ? 'none' : 'inline-block';
        if (stopBtn) stopBtn.style.display = cameraActive ? 'inline-block' : 'none';
        if (captureBtn) captureBtn.style.display = cameraActive ? 'inline-block' : 'none';
        if (toggleBtn) toggleBtn.style.display = cameraActive ? 'inline-block' : 'none';
    }

    /**
     * Capture current frame as image
     * @private
     */
    captureCurrentFrame() {
        if (!this.cameraCanvas) {
            this.showWarning('No active camera stream');
            return;
        }

        // Create download link
        this.cameraCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `capture_${Date.now()}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showSuccess('Frame captured and downloaded');
        }, 'image/jpeg', 0.9);
    }

    /**
     * Toggle detection on/off while keeping camera active
     * @private
     */
    toggleDetection() {
        if (this.isDetecting) {
            this.stopDetection();
            document.getElementById('toggleDetectionBtn').textContent = 'Resume Detection';
            this.showSuccess('Detection paused');
        } else {
            this.startDetection();
            document.getElementById('toggleDetectionBtn').textContent = 'Pause Detection';
            this.showSuccess('Detection resumed');
        }
    }

    /**
     * Setup analysis tab navigation for detailed results
     * @private
     */
    setupAnalysisTabNavigation() {
        const analysisTabs = document.querySelectorAll('.analysis-tab');

        analysisTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabType = tab.getAttribute('data-tab');
                this.switchAnalysisTab(tabType);
            });
        });
    }

    /**
     * Switch analysis detail tabs
     * @param {string} tabType - The analysis tab to switch to
     * @private
     */
    switchAnalysisTab(tabType) {
        // Update active tab
        document.querySelectorAll('.analysis-tab').forEach(tab => {
            tab.classList.remove('active');
        });

        document.querySelector(`[data-tab="${tabType}"]`)?.classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        document.getElementById(`${tabType}-content`)?.classList.add('active');

        this.activeTab = tabType;
    }

    /**
     * Display image analysis results
     * @param {Object} data - Analysis results data
     * @param {File} originalFile - Original image file
     * @private
     */
    displayImageResults(data, originalFile) {
        this.currentResults = data;

        // Display original image
        this.displayOriginalImage(originalFile, data);

        // Display detection results
        this.displayDetectionResults(data.detection);

        // Display segmentation results
        this.displaySegmentationResults(data.segmentation);

        // Update detailed analysis
        this.updateDetailedAnalysis(data);

        // Show results with animation
        const results = document.getElementById('results');
        if (results) {
            results.style.display = 'block';
            setTimeout(() => {
                results.classList.add('show');
            }, 100);
        }

        // Hide loading
        this.hideImageLoading();
    }

    /**
     * Display original image with metadata
     * @param {File} file - Original image file
     * @param {Object} data - Analysis data
     * @private
     */
    displayOriginalImage(file, data) {
        const container = document.getElementById('originalImageContainer');
        const info = document.getElementById('originalImageInfo');

        if (!container || !info) return;

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = 'Original image';
        img.style.cursor = 'pointer';
        img.onclick = () => window.openImageModal(img.src);

        container.innerHTML = '';
        container.appendChild(img);

        info.innerHTML = `
            <p><strong>File:</strong> ${file.name}</p>
            <p><strong>Size:</strong> ${this.formatFileSize(file.size)}</p>
            <p><strong>Dimensions:</strong> ${data.image_info?.width || 'Unknown'} × ${data.image_info?.height || 'Unknown'}</p>
        `;
    }

    /**
     * Validate image file
     * @param {File} file - File to validate
     * @param {number} maxSizeMB - Maximum size in MB
     * @private
     */
    validateImageFile(file, maxSizeMB = 10) {
        if (!file.type.startsWith('image/')) {
            throw new Error('Please select a valid image file (JPG, PNG, WebP)');
        }

        if (file.size > maxSizeMB * 1024 * 1024) {
            throw new Error(`File size must be less than ${maxSizeMB}MB`);
        }
    }

    /**
     * Validate video file
     * @param {File} file - File to validate
     * @param {number} maxSizeMB - Maximum size in MB
     * @private
     */
    validateVideoFile(file, maxSizeMB = 100) {
        if (!file.type.startsWith('video/')) {
            throw new Error('Please select a valid video file (MP4, AVI, MOV, WebM)');
        }

        if (file.size > maxSizeMB * 1024 * 1024) {
            throw new Error(`File size must be less than ${maxSizeMB}MB`);
        }
    }

    /**
     * Format file size in human readable format
     * @param {number} bytes - File size in bytes
     * @returns {string} Formatted file size
     * @private
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Show image loading state
     * @private
     */
    showImageLoading() {
        this.isProcessing = true;

        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        if (loading) loading.style.display = 'block';
        if (results) {
            results.style.display = 'none';
            results.classList.remove('show');
        }
    }

    /**
     * Hide image loading state
     * @private
     */
    hideImageLoading() {
        this.isProcessing = false;

        const loading = document.getElementById('loading');
        const progressContainer = document.getElementById('progressContainer');

        if (loading) loading.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'none';
    }

    /**
     * Show video processing progress
     * @private
     */
    showVideoProgress() {
        const videoProgress = document.getElementById('videoProgress');
        const processBtn = document.getElementById('processVideoBtn');
        const cancelBtn = document.getElementById('cancelVideoBtn');

        if (videoProgress) videoProgress.style.display = 'block';
        if (processBtn) processBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'inline-block';
    }

    /**
     * Hide video processing progress
     * @private
     */
    hideVideoProgress() {
        const videoProgress = document.getElementById('videoProgress');
        const processBtn = document.getElementById('processVideoBtn');
        const cancelBtn = document.getElementById('cancelVideoBtn');

        if (videoProgress) videoProgress.style.display = 'none';
        if (processBtn) processBtn.style.display = 'inline-block';
        if (cancelBtn) cancelBtn.style.display = 'none';
    }

    /**
     * Simulate progress for better user experience
     * @returns {number} Interval ID
     * @private
     */
    simulateImageProgress() {
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');

        if (progressContainer) progressContainer.style.display = 'block';

        let progress = 0;
        const stages = [
            'Initializing models...',
            'Loading image...',
            'Running object detection...',
            'Performing segmentation...',
            'Generating visualizations...',
            'Finalizing results...'
        ];
        let stageIndex = 0;

        const interval = setInterval(() => {
            progress += Math.random() * 12;
            if (progress > 90) progress = 90;

            if (progressBar) {
                progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
            }

            if (progressText && stageIndex < stages.length) {
                if (progress > (stageIndex + 1) * 15) {
                    progressText.textContent = stages[stageIndex];
                    stageIndex++;
                }
            }
        }, 300);

        return interval;
    }

    /**
     * Complete progress animation
     * @private
     */
    completeProgress() {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');

        if (progressBar) {
            progressBar.style.width = '100%';
        }
        if (progressText) {
            progressText.textContent = 'Analysis complete!';
        }
    }

    /**
     * Show success message
     * @param {string} message - Success message
     * @private
     */
    showSuccess(message) {
        if (window.CNNPlayground && window.CNNPlayground.showSuccess) {
            window.CNNPlayground.showSuccess(message);
        }
    }

    /**
     * Show error message
     * @param {string} message - Error message
     * @private
     */
    showError(message) {
        if (window.CNNPlayground && window.CNNPlayground.showError) {
            window.CNNPlayground.showError(message);
        }
    }

    /**
     * Show warning message
     * @param {string} message - Warning message
     * @private
     */
    showWarning(message) {
        if (window.CNNPlayground && window.CNNPlayground.showWarning) {
            window.CNNPlayground.showWarning(message);
        }
    }

    /**
     * Stop all active processes when switching modes
     * @private
     */
    stopAllProcesses() {
        // Stop camera if active
        if (this.cameraStream) {
            this.stopCamera();
        }

        // Cancel any video processing
        this.isProcessing = false;

        // Clear intervals
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
    }

    /**
     * Reset analyzer state
     * @private
     */
    resetState() {
        this.currentResults = null;
        this.activeTab = 'summary';
        this.isProcessing = false;
        this.currentVideo = null;
        this.detectionLog = [];
        this.detectionHistory = [];

        // Reset file inputs
        const fileInput = document.getElementById('fileInput');
        const videoInput = document.getElementById('videoInput');

        if (fileInput) fileInput.value = '';
        if (videoInput) videoInput.value = '';

        // Hide all result containers
        const results = document.getElementById('results');
        const videoResults = document.getElementById('videoResults');
        const liveStats = document.getElementById('liveStats');

        if (results) {
            results.style.display = 'none';
            results.classList.remove('show');
        }
        if (videoResults) videoResults.style.display = 'none';
        if (liveStats) liveStats.style.display = 'none';

        // Reset to summary tab
        this.switchAnalysisTab('summary');
    }

    /**
     * Display object detection results
     * @param {Object} detectionData - Detection results from API
     * @private
     */
    displayDetectionResults(detectionData) {
        const container = document.getElementById('detectionImageContainer');
        const objectsCount = document.getElementById('detectedObjectsCount');
        const detectionTime = document.getElementById('detectionTime');
        const objectsList = document.getElementById('detectedObjectsList');

        if (!container || !detectionData) return;

        // Display detection image with bounding boxes
        if (detectionData.image_url) {
            const img = document.createElement('img');
            img.src = detectionData.image_url;
            img.alt = 'Object detection results';
            img.style.cursor = 'pointer';
            img.onclick = () => window.openImageModal(img.src);

            container.innerHTML = '';
            container.appendChild(img);
        } else {
            container.innerHTML = '<div class="image-placeholder"><p>No detection visualization available</p></div>';
        }

        // Update statistics
        if (objectsCount) objectsCount.textContent = detectionData.objects?.length || 0;
        if (detectionTime) detectionTime.textContent = `${detectionData.processing_time || 0}ms`;

        // Update detected objects list
        if (objectsList && detectionData.objects) {
            if (detectionData.objects.length > 0) {
                objectsList.innerHTML = detectionData.objects.map(obj => `
                    <div class="object-item">
                        <span class="object-name">${obj.class_name}</span>
                        <span class="object-confidence">${(obj.confidence * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            } else {
                objectsList.innerHTML = '<p>No objects detected</p>';
            }
        }

        console.log('[ComputerVision] Detection results displayed:', detectionData.objects?.length || 0, 'objects');
    }

    /**
     * Display instance segmentation results
     * @param {Object} segmentationData - Segmentation results from API
     * @private
     */
    displaySegmentationResults(segmentationData) {
        const container = document.getElementById('segmentationImageContainer');
        const regionsCount = document.getElementById('segmentedRegionsCount');
        const segmentationTime = document.getElementById('segmentationTime');
        const classesList = document.getElementById('segmentationClassesList');

        if (!container || !segmentationData) return;

        // Display segmentation image with pixel-level masks
        if (segmentationData.image_url) {
            const img = document.createElement('img');
            img.src = segmentationData.image_url;
            img.alt = 'Instance segmentation results';
            img.style.cursor = 'pointer';
            img.onclick = () => window.openImageModal(img.src);

            container.innerHTML = '';
            container.appendChild(img);
        } else {
            container.innerHTML = '<div class="image-placeholder"><p>No segmentation visualization available</p></div>';
        }

        // Update statistics
        if (regionsCount) regionsCount.textContent = segmentationData.segments?.length || 0;
        if (segmentationTime) segmentationTime.textContent = `${segmentationData.processing_time || 0}ms`;

        // Update segmented classes list
        if (classesList && segmentationData.segments) {
            if (segmentationData.segments.length > 0) {
                classesList.innerHTML = segmentationData.segments.map(segment => `
                    <div class="class-item">
                        <span class="class-name">${segment.class_name}</span>
                        <span class="class-coverage">${(segment.coverage * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            } else {
                classesList.innerHTML = '<p>No segments identified</p>';
            }
        }

        console.log('[ComputerVision] Segmentation results displayed:', segmentationData.segments?.length || 0, 'segments');
    }

    /**
     * Update detailed analysis tabs with comprehensive data
     * @param {Object} data - Complete analysis results
     * @private
     */
    updateDetailedAnalysis(data) {
        this.updateSummaryTab(data);
        this.updateDetectionDetailsTab(data.detection);
        this.updateSegmentationDetailsTab(data.segmentation);
        this.updateModelInfoTab(data.model_info);
    }

    /**
     * Update summary tab with performance overview
     * @param {Object} data - Analysis results
     * @private
     */
    updateSummaryTab(data) {
        const performanceSummary = document.getElementById('performanceSummary');
        const detectionSummary = document.getElementById('detectionSummary');
        const segmentationSummary = document.getElementById('segmentationSummary');

        if (performanceSummary) {
            const totalTime = (data.detection?.processing_time || 0) + (data.segmentation?.processing_time || 0);
            performanceSummary.innerHTML = `
                <p><strong>Total Processing Time:</strong> ${totalTime}ms</p>
                <p><strong>Detection Time:</strong> ${data.detection?.processing_time || 0}ms</p>
                <p><strong>Segmentation Time:</strong> ${data.segmentation?.processing_time || 0}ms</p>
                <p><strong>Device:</strong> ${data.model_info?.device || 'Unknown'}</p>
                <p><strong>Status:</strong> <span style="color: #10b981;">Analysis Complete</span></p>
            `;
        }

        if (detectionSummary && data.detection) {
            const objects = data.detection.objects || [];
            const avgConfidence = objects.length > 0
                ? objects.reduce((sum, obj) => sum + obj.confidence, 0) / objects.length
                : 0;
            const uniqueClasses = new Set(objects.map(obj => obj.class_name)).size;

            detectionSummary.innerHTML = `
                <p><strong>Objects Detected:</strong> ${objects.length}</p>
                <p><strong>Average Confidence:</strong> ${(avgConfidence * 100).toFixed(1)}%</p>
                <p><strong>Unique Classes:</strong> ${uniqueClasses}</p>
                <p><strong>Model:</strong> ${data.model_info?.detection?.architecture || 'YOLO11'}</p>
            `;
        }

        if (segmentationSummary && data.segmentation) {
            const segments = data.segmentation.segments || [];
            const totalCoverage = segments.reduce((sum, seg) => sum + (seg.coverage || 0), 0);
            const avgPixelCount = segments.length > 0
                ? segments.reduce((sum, seg) => sum + (seg.pixel_count || 0), 0) / segments.length
                : 0;

            segmentationSummary.innerHTML = `
                <p><strong>Segments Found:</strong> ${segments.length}</p>
                <p><strong>Total Coverage:</strong> ${(totalCoverage * 100).toFixed(1)}%</p>
                <p><strong>Background:</strong> ${((1 - totalCoverage) * 100).toFixed(1)}%</p>
                <p><strong>Avg. Pixels per Segment:</strong> ${Math.round(avgPixelCount).toLocaleString()}</p>
            `;
        }
    }

    /**
     * Update detection details tab with comprehensive object information
     * @param {Object} detectionData - Detection results
     * @private
     */
    updateDetectionDetailsTab(detectionData) {
        const table = document.getElementById('detectionDetailsTable');
        if (!table || !detectionData?.objects) return;

        const objects = detectionData.objects;
        if (objects.length === 0) {
            table.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">No objects detected in this image</p>';
            return;
        }

        const tableHTML = `
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="background: var(--bg-secondary); border-bottom: 2px solid var(--border-color);">
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: var(--text-primary);">#</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: var(--text-primary);">Object Class</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: var(--text-primary);">Confidence</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: var(--text-primary);">Bounding Box</th>
                            <th style="padding: 12px; text-align: right; font-weight: 600; color: var(--text-primary);">Area (px²)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${objects.map((obj, index) => `
                            <tr style="border-bottom: 1px solid var(--border-color); transition: background-color 0.2s ease;">
                                <td style="padding: 10px; font-weight: 500; color: var(--accent-color);">${index + 1}</td>
                                <td style="padding: 10px; color: var(--text-primary); font-weight: 500;">${obj.class_name}</td>
                                <td style="padding: 10px; text-align: center;">
                                    <span style="
                                        background: ${obj.confidence > 0.8 ? '#10b981' : obj.confidence > 0.5 ? '#f59e0b' : '#ef4444'};
                                        color: white;
                                        padding: 4px 8px;
                                        border-radius: 12px;
                                        font-size: 0.8em;
                                        font-weight: 600;
                                    ">${(obj.confidence * 100).toFixed(1)}%</span>
                                </td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.8em; color: var(--text-secondary);">
                                    [${obj.bbox?.map(n => Math.round(n)).join(', ') || 'N/A'}]
                                </td>
                                <td style="padding: 10px; text-align: right; color: var(--text-secondary);">
                                    ${obj.area ? Math.round(obj.area).toLocaleString() : 'N/A'}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: var(--bg-secondary); border-radius: 8px; font-size: 0.9em; color: var(--text-secondary);">
                <strong>Legend:</strong> 
                <span style="color: #10b981;">●</span> High Confidence (>80%) |
                <span style="color: #f59e0b;">●</span> Medium Confidence (50-80%) |
                <span style="color: #ef4444;">●</span> Low Confidence (<50%)
            </div>
        `;

        table.innerHTML = tableHTML;
    }

    /**
     * Update segmentation details tab with pixel-level analysis
     * @param {Object} segmentationData - Segmentation results
     * @private
     */
    updateSegmentationDetailsTab(segmentationData) {
        const table = document.getElementById('segmentationDetailsTable');
        if (!table || !segmentationData?.segments) return;

        const segments = segmentationData.segments;
        if (segments.length === 0) {
            table.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">No segments identified in this image</p>';
            return;
        }

        const tableHTML = `
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="background: var(--bg-secondary); border-bottom: 2px solid var(--border-color);">
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: var(--text-primary);">#</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: var(--text-primary);">Segment Class</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: var(--text-primary);">Coverage</th>
                            <th style="padding: 12px; text-align: right; font-weight: 600; color: var(--text-primary);">Pixel Count</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: var(--text-primary);">Color</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: var(--text-primary);">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${segments.map((segment, index) => `
                            <tr style="border-bottom: 1px solid var(--border-color); transition: background-color 0.2s ease;">
                                <td style="padding: 10px; font-weight: 500; color: var(--accent-color);">${index + 1}</td>
                                <td style="padding: 10px; color: var(--text-primary); font-weight: 500;">${segment.class_name}</td>
                                <td style="padding: 10px; text-align: center;">
                                    <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                                        <div style="
                                            width: 60px;
                                            height: 8px;
                                            background: var(--border-color);
                                            border-radius: 4px;
                                            overflow: hidden;
                                        ">
                                            <div style="
                                                width: ${(segment.coverage * 100).toFixed(1)}%;
                                                height: 100%;
                                                background: var(--accent-color);
                                            "></div>
                                        </div>
                                        <span style="font-weight: 600; color: var(--accent-color);">
                                            ${(segment.coverage * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </td>
                                <td style="padding: 10px; text-align: right; color: var(--text-secondary);">
                                    ${segment.pixel_count?.toLocaleString() || 'N/A'}
                                </td>
                                <td style="padding: 10px; text-align: center;">
                                    <div style="
                                        width: 24px;
                                        height: 24px;
                                        background: ${segment.color || '#888'};
                                        border-radius: 4px;
                                        margin: 0 auto;
                                        border: 1px solid var(--border-color);
                                    "></div>
                                </td>
                                <td style="padding: 10px; text-align: center; color: var(--text-secondary);">
                                    ${segment.confidence ? (segment.confidence * 100).toFixed(1) + '%' : 'N/A'}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: var(--bg-secondary); border-radius: 8px; font-size: 0.9em; color: var(--text-secondary);">
                <strong>Total Image Coverage:</strong> ${(segments.reduce((sum, seg) => sum + (seg.coverage || 0), 0) * 100).toFixed(1)}% |
                <strong>Background:</strong> ${((1 - segments.reduce((sum, seg) => sum + (seg.coverage || 0), 0)) * 100).toFixed(1)}% |
                <strong>Total Pixels:</strong> ${segments.reduce((sum, seg) => sum + (seg.pixel_count || 0), 0).toLocaleString()}
            </div>
        `;

        table.innerHTML = tableHTML;
    }

    /**
     * Update model information tab with detailed technical specifications
     * @param {Object} modelInfo - Model information from API
     * @private
     */
    updateModelInfoTab(modelInfo) {
        const detectionInfo = document.getElementById('detectionModelInfo');
        const segmentationInfo = document.getElementById('segmentationModelInfo');

        if (detectionInfo && modelInfo?.detection) {
            const detection = modelInfo.detection;
            detectionInfo.innerHTML = `
                <p><strong>Architecture:</strong> ${detection.architecture || 'YOLO11'}</p>
                <p><strong>Model Variant:</strong> ${detection.model_size || 'Medium'}</p>
                <p><strong>Parameters:</strong> ${detection.parameters || '~20M'}</p>
                <p><strong>Input Resolution:</strong> ${detection.input_size || '640×640'}</p>
                <p><strong>Output Classes:</strong> ${detection.num_classes || '80'} (COCO dataset)</p>
                <p><strong>Speed Rating:</strong> ${detection.speed_rating || 'N/A'}/5</p>
                <p><strong>Accuracy Rating:</strong> ${detection.accuracy_rating || 'N/A'}/5</p>
                <p><strong>Device:</strong> ${modelInfo.device || 'CPU'}</p>
            `;
        }

        if (segmentationInfo && modelInfo?.segmentation) {
            const segmentation = modelInfo.segmentation;
            segmentationInfo.innerHTML = `
                <p><strong>Architecture:</strong> ${segmentation.architecture || 'YOLO11-seg'}</p>
                <p><strong>Segmentation Type:</strong> ${segmentation.segmentation_type || 'Instance'}</p>
                <p><strong>Parameters:</strong> ${segmentation.parameters || '~22M'}</p>
                <p><strong>Classes Supported:</strong> ${segmentation.classes || '80'} categories</p>
                <p><strong>Framework:</strong> ${segmentation.framework || 'YOLOv11'}</p>
                <p><strong>Status:</strong> ${segmentation.status || 'Stable'}</p>
                <p><strong>Precision:</strong> Pixel-level accuracy</p>
                <p><strong>Device:</strong> ${modelInfo.device || 'CPU'}</p>
            `;
        }
    }

    /**
     * Simulate video processing progress with realistic stages
     * @returns {number} Interval ID
     * @private
     */
    simulateVideoProgress() {
        const progressBar = document.getElementById('videoProgressBar');
        const progressText = document.getElementById('videoProgressText');
        const progressStats = document.getElementById('videoProgressStats');
        const progressPercentage = document.getElementById('videoProgressPercentage');

        let progress = 0;
        let currentFrame = 0;
        const totalFrames = Math.floor(Math.random() * 500) + 100; // Simulate frame count

        const stages = [
            'Extracting video frames...',
            'Initializing detection models...',
            'Processing frames with neural networks...',
            'Applying object detection...',
            'Rendering visualization overlays...',
            'Encoding processed video...',
            'Finalizing output...'
        ];
        let stageIndex = 0;

        const interval = setInterval(() => {
            progress += Math.random() * 8 + 2;
            currentFrame = Math.floor((progress / 100) * totalFrames);

            if (progress > 95) progress = 95;

            if (progressBar) {
                progressBar.style.width = `${progress}%`;
            }

            if (progressPercentage) {
                progressPercentage.textContent = `${Math.round(progress)}%`;
            }

            if (progressText && stageIndex < stages.length) {
                if (progress > (stageIndex + 1) * 14) {
                    progressText.textContent = stages[stageIndex];
                    stageIndex++;
                }
            }

            if (progressStats) {
                progressStats.textContent = `Frame ${Math.min(currentFrame, totalFrames)} of ${totalFrames}`;
            }
        }, 400);

        return interval;
    }

    /**
     * Display video processing results
     * @param {Object} data - Video processing results
     * @private
     */
    displayVideoResults(data) {
        const processedVideo = document.getElementById('processedVideo');
        const videoStats = document.getElementById('videoStats');
        const downloadBtn = document.getElementById('downloadVideoBtn');

        if (processedVideo && data.processed_video_url) {
            processedVideo.src = data.processed_video_url;
            processedVideo.style.display = 'block';
        }

        if (downloadBtn && data.processed_video_url) {
            downloadBtn.style.display = 'inline-block';
            downloadBtn.onclick = () => {
                const a = document.createElement('a');
                a.href = data.processed_video_url;
                a.download = `processed_video_${Date.now()}.mp4`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };
        }

        if (videoStats && data.stats) {
            videoStats.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; padding: 20px; background: var(--bg-primary); border-radius: 12px; border: 1px solid var(--border-color);">
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; font-weight: bold; color: var(--accent-color);">${data.stats.total_frames || 0}</div>
                        <div style="font-size: 0.9em; color: var(--text-secondary);">Total Frames</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; font-weight: bold; color: var(--accent-color);">${data.stats.processing_time || 0}s</div>
                        <div style="font-size: 0.9em; color: var(--text-secondary);">Processing Time</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; font-weight: bold; color: var(--accent-color);">${data.stats.objects_detected || 0}</div>
                        <div style="font-size: 0.9em; color: var(--text-secondary);">Objects Detected</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5em; font-weight: bold; color: var(--accent-color);">${data.stats.avg_fps || 0}</div>
                        <div style="font-size: 0.9em; color: var(--text-secondary);">Average FPS</div>
                    </div>
                </div>
            `;
        }

        console.log('[ComputerVision] Video results displayed');
    }

    /**
     * Cancel video processing
     * @private
     */
    cancelVideoProcessing() {
        this.isProcessing = false;
        this.hideVideoProgress();
        this.showWarning('Video processing cancelled');
        console.log('[ComputerVision] Video processing cancelled by user');
    }

    /**
     * Download processed video
     * @private
     */
    downloadProcessedVideo() {
        const processedVideo = document.getElementById('processedVideo');
        if (processedVideo && processedVideo.src) {
            const a = document.createElement('a');
            a.href = processedVideo.src;
            a.download = `processed_video_${Date.now()}.mp4`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            this.showSuccess('Video download started');
        } else {
            this.showWarning('No processed video available for download');
        }
    }

    /**
     * Public API method to get current analyzer state
     * @returns {Object} Current analyzer state
     * @public
     */
    getState() {
        return {
            currentMode: this.currentMode,
            isProcessing: this.isProcessing,
            activeTab: this.activeTab,
            hasResults: !!this.currentResults,
            cameraActive: !!this.cameraStream,
            detectingActive: this.isDetecting,
            videoLoaded: !!this.currentVideo,
            detectionLogCount: this.detectionLog.length,
            isProcessingFrame: this.isProcessingFrame,
            averageLatency: this.detectionStats.averageLatency
        };
    }

    /**
     * Public method to force cleanup (useful for page navigation)
     * @public
     */
    cleanup() {
        this.stopAllProcesses();
        this.resetState();
        console.log('[ComputerVision] Analyzer cleaned up');
    }
}

// Export for global use
window.ComputerVisionAnalyzer = ComputerVisionAnalyzer;