/**
 * MNIST Canvas Drawing Interface - COMPLETELY FIXED VERSION
 * Working erase functionality without white dots
 */

class MNISTCanvas {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.brushSize = 20;
        this.eraseSize = 35; // Zwiększony rozmiar gumki

        // Store all strokes for redrawing
        this.allStrokes = [];
        this.currentStroke = [];

        // Erase mode
        this.isEraseMode = false;

        // Prediction settings
        this.autoPredictEnabled = true;
        this.predictionTimeout = null;
        this.isRequesting = false;
        this.lastImageHash = null;
        this.lastPredictionId = 0;

        this.initialize();
    }

    initialize() {
        if (!this.canvas || !this.ctx) {
            console.error('Canvas or context not found');
            return;
        }

        this.setupCanvas();
        this.setupEventListeners();
        this.setupBrushControls();
        this.initializeConfidenceBars();
        this.setupImageModalDelegation();

        console.log('MNIST Canvas initialized');
    }
    setupImageModalDelegation() {
    document.addEventListener('click', (e) => {
        const target = e.target;
        if (target.tagName === 'IMG' &&
            (target.classList.contains('gradcam-image') ||
             target.classList.contains('layer-image') ||
             target.hasAttribute('data-modal-src'))) {

            e.preventDefault();
            e.stopPropagation();

            const src = target.getAttribute('data-modal-src') || target.src;
            if (window.openImageModal) {
                window.openImageModal(src);
            }
        }
    });
}
    setupCanvas() {
        // Set canvas properties
        this.canvas.style.position = 'relative';
        this.canvas.style.display = 'block';
        this.canvas.style.touchAction = 'none';

        // Configure context - zawsze zaczynamy w trybie normalnego rysowania
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.globalCompositeOperation = 'source-over';
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = this.brushSize;
        this.canvas.style.cursor = 'crosshair';

        // White background
        this.clearCanvas();
    }

    updateCanvasMode() {
        // Aktualizuje tryb canvas w zależności od isEraseMode
        if (this.isEraseMode) {
            this.ctx.globalCompositeOperation = 'destination-out';
            this.ctx.lineWidth = this.eraseSize;
            this.updateCursor();
            console.log('Canvas mode set to ERASE (destination-out)');
        } else {
            this.ctx.globalCompositeOperation = 'source-over';
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = this.brushSize;
            this.updateCursor();
            console.log('Canvas mode set to DRAW (source-over)');
        }
    }

    updateCursor() {
        if (this.isEraseMode) {
            const cursorSize = Math.min(this.eraseSize, 50); // Ogranicz max rozmiar kursora
            const cursorUrl = this.createEraseCursor(cursorSize);
            this.canvas.style.cursor = `url(${cursorUrl}) ${cursorSize/2} ${cursorSize/2}, auto`;
        } else {
            this.canvas.style.cursor = 'crosshair';
        }
    }

    createEraseCursor(size) {
        // Utwórz niestandardowy kursor gumki
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = size;
        canvas.height = size;

        // Narysuj koło z obramowaniem
        const center = size / 2;
        const radius = Math.max(2, center - 2);

        // Przezroczyste tło
        ctx.clearRect(0, 0, size, size);

        // Białe koło z czarnym obramowaniem
        ctx.beginPath();
        ctx.arc(center, center, radius, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fill();
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Dodaj krzyżyk w środku
        ctx.beginPath();
        ctx.moveTo(center - 3, center);
        ctx.lineTo(center + 3, center);
        ctx.moveTo(center, center - 3);
        ctx.lineTo(center, center + 3);
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.stroke();

        return canvas.toDataURL();
    }

    toggleEraseMode() {
        this.isEraseMode = !this.isEraseMode;

        console.log('Toggling erase mode to:', this.isEraseMode);

        // Natychmiast aktualizuj tryb canvas
        this.updateCanvasMode();

        // Update button appearance
        this.updateEraseButton();

        // Aktualizuj slider label
        this.updateSliderLabel();
    }

    updateSliderLabel() {
        const brushSizeSlider = document.getElementById('brushSize');
        if (brushSizeSlider) {
            const label = brushSizeSlider.previousElementSibling;
            if (label && label.tagName === 'LABEL') {
                if (this.isEraseMode) {
                    label.textContent = 'Erase Size:';
                } else {
                    label.textContent = 'Brush Size:';
                }
            }
        }
    }

    updateEraseButton() {
        const eraseBtn = document.getElementById('eraseBtn');
        console.log('Updating erase button, found element:', !!eraseBtn);

        if (eraseBtn) {
            if (this.isEraseMode) {
                eraseBtn.classList.add('active');
                eraseBtn.innerHTML = '✏️ Draw';
                eraseBtn.title = 'Switch to draw mode';
                console.log('Button set to Draw mode');
            } else {
                eraseBtn.classList.remove('active');
                eraseBtn.innerHTML = '🧽 Erase';
                eraseBtn.title = 'Switch to erase mode';
                console.log('Button set to Erase mode');
            }
        } else {
            console.error('Erase button not found!');
        }
    }

    clearCanvas() {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    redrawAllStrokes() {
        // Clear canvas
        this.clearCanvas();

        // Redraw all strokes
        this.allStrokes.forEach(stroke => {
            if (stroke.length === 0) return;

            const firstPoint = stroke[0];

            if (firstPoint.isErase) {
                // For erase strokes, use destination-out
                this.ctx.save();
                this.ctx.globalCompositeOperation = 'destination-out';
                this.ctx.lineWidth = firstPoint.size;
                this.ctx.strokeStyle = '#000000'; // Color doesn't matter for destination-out

                this.ctx.beginPath();
                this.ctx.moveTo(firstPoint.x, firstPoint.y);
                for (let i = 1; i < stroke.length; i++) {
                    this.ctx.lineTo(stroke[i].x, stroke[i].y);
                }
                this.ctx.stroke();
                this.ctx.restore();

            } else {
                // For draw strokes, use normal drawing
                this.ctx.save();
                this.ctx.globalCompositeOperation = 'source-over';
                this.ctx.strokeStyle = '#000000';
                this.ctx.lineWidth = firstPoint.size;

                this.ctx.beginPath();
                this.ctx.moveTo(firstPoint.x, firstPoint.y);
                for (let i = 1; i < stroke.length; i++) {
                    this.ctx.lineTo(stroke[i].x, stroke[i].y);
                }
                this.ctx.stroke();
                this.ctx.restore();
            }
        });

        // Reset to current mode
        this.updateCanvasMode();
    }

    setupEventListeners() {
        // Prevent default behaviors
        this.canvas.addEventListener('selectstart', (e) => e.preventDefault());
        this.canvas.addEventListener('dragstart', (e) => e.preventDefault());
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.startDrawing(e);
        });

        this.canvas.addEventListener('mousemove', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.draw(e);
        });

        this.canvas.addEventListener('mouseup', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.stopDrawing();
        });

        this.canvas.addEventListener('mouseleave', (e) => {
            e.preventDefault();
            this.stopDrawing();
        });

        // Touch events
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (e.touches.length === 1) {
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousedown', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                this.canvas.dispatchEvent(mouseEvent);
            }
        }, { passive: false });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (e.touches.length === 1) {
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousemove', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                this.canvas.dispatchEvent(mouseEvent);
            }
        }, { passive: false });

        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.dispatchEvent(new MouseEvent('mouseup', {}));
        }, { passive: false });

        // Button events with debugging
        const predictBtn = document.getElementById('predictBtn');
        const clearBtn = document.getElementById('clearBtn');
        const eraseBtn = document.getElementById('eraseBtn');

        console.log('Setting up button listeners:');
        console.log('- predictBtn found:', !!predictBtn);
        console.log('- clearBtn found:', !!clearBtn);
        console.log('- eraseBtn found:', !!eraseBtn);

        if (predictBtn) {
            predictBtn.addEventListener('click', () => {
                console.log('Predict button clicked');
                this.predict(true);
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                console.log('Clear button clicked');
                this.clear();
            });
        }

        if (eraseBtn) {
            eraseBtn.addEventListener('click', () => {
                console.log('Erase button clicked');
                this.toggleEraseMode();
            });
        } else {
            console.error('Erase button not found during setup!');
        }
    }

    setupBrushControls() {
        const brushSizeSlider = document.getElementById('brushSize');
        const brushPreview = document.getElementById('brushPreview');

        if (brushSizeSlider && brushPreview) {
            brushSizeSlider.min = 10;
            brushSizeSlider.max = 40;
            brushSizeSlider.value = this.brushSize;

            brushSizeSlider.addEventListener('input', (e) => {
                const newSize = parseInt(e.target.value);

                if (this.isEraseMode) {
                    // W trybie erase, kontroluj rozmiar gumki
                    this.eraseSize = Math.max(newSize + 10, 25); // Gumka większa niż pędzel
                    this.ctx.lineWidth = this.eraseSize;
                    this.updateCursor(); // Aktualizuj kursor gumki
                } else {
                    // W trybie draw, kontroluj rozmiar pędzla
                    this.brushSize = newSize;
                    this.ctx.lineWidth = this.brushSize;
                }

                // Preview ZAWSZE pokazuje tylko wartość slidera w tym samym kolorze
                const previewSize = Math.max(10, newSize);
                brushPreview.style.width = previewSize + 'px';
                brushPreview.style.height = previewSize + 'px';
                // Kolor ZAWSZE ten sam
                brushPreview.style.backgroundColor = '#333';
                brushPreview.style.border = '2px solid #666';
            });

            // Ustaw początkowy preview - tylko raz na start
            const previewSize = Math.max(10, this.brushSize);
            brushPreview.style.width = previewSize + 'px';
            brushPreview.style.height = previewSize + 'px';
            brushPreview.style.backgroundColor = '#333';
            brushPreview.style.border = '2px solid #666';

            this.updateSliderLabel();
        }
    }

    updateBrushPreview() {
        const brushPreview = document.getElementById('brushPreview');
        if (!brushPreview) return;

        // Preview zawsze pokazuje rozmiar z slidera, nie rzeczywisty rozmiar gumki
        const brushSizeSlider = document.getElementById('brushSize');
        const sliderValue = brushSizeSlider ? parseInt(brushSizeSlider.value) : this.brushSize;
        const previewSize = Math.max(10, sliderValue);

        brushPreview.style.width = previewSize + 'px';
        brushPreview.style.height = previewSize + 'px';

        // Zawsze ten sam kolor, niezależnie od trybu
        brushPreview.style.backgroundColor = '#333';
        brushPreview.style.border = '2px solid #666';
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.lastX = pos.x;
        this.lastY = pos.y;

        // Ustaw prawidłowy tryb rysowania PRZED rozpoczęciem ruchu
        this.updateCanvasMode();

        this.currentStroke = [{
            x: pos.x,
            y: pos.y,
            size: this.isEraseMode ? this.eraseSize : this.brushSize,
            isErase: this.isEraseMode
        }];

        // Hide overlay
        const overlay = document.getElementById('canvasOverlay');
        if (overlay) {
            overlay.classList.remove('show');
        }

        this.cancelPrediction();

        console.log('Started drawing in mode:', this.isEraseMode ? 'erase' : 'draw');
    }

    draw(e) {
        if (!this.isDrawing) return;

        const pos = this.getMousePos(e);

        // Upewnij się, że context jest w prawidłowym trybie
        this.updateCanvasMode();

        // Rysuj w bieżącym trybie (normalny lub erase)
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();

        this.currentStroke.push({
            x: pos.x,
            y: pos.y,
            size: this.isEraseMode ? this.eraseSize : this.brushSize,
            isErase: this.isEraseMode
        });

        this.lastX = pos.x;
        this.lastY = pos.y;
    }

    stopDrawing() {
        if (!this.isDrawing) return;

        this.isDrawing = false;

        if (this.currentStroke.length > 0) {
            // Add current stroke to all strokes
            this.allStrokes.push([...this.currentStroke]);

            // WAŻNE: Jeśli to był ruch wymazywania, od razu przerysuj wszystko
            // aby canvas pokazywał prawidłowy obraz
            if (this.currentStroke[0].isErase) {
                this.redrawAllStrokes();
            }

            this.currentStroke = [];
        }

        // Zawsze resetuj do normalnego trybu po zakończeniu ruchu
        this.ctx.globalCompositeOperation = 'source-over';
        this.ctx.strokeStyle = '#000000';

        // Ustaw prawidłową szerokość linii dla bieżącego trybu
        this.updateCanvasMode();

        if (this.autoPredictEnabled) {
            this.scheduleAutoPrediction();
        }

        console.log('Stopped drawing, total strokes:', this.allStrokes.length);
    }

    scheduleAutoPrediction() {
        this.cancelPrediction();

        this.predictionTimeout = setTimeout(() => {
            this.predict(false);
        }, 1200);
    }

    cancelPrediction() {
        if (this.predictionTimeout) {
            clearTimeout(this.predictionTimeout);
            this.predictionTimeout = null;
        }
    }

    async predict(forcePrediction = false) {
        try {
            if (this.isRequesting && !forcePrediction) {
                console.log('Skipping prediction - request in progress');
                return;
            }

            if (this.allStrokes.length === 0) {
                this.showNoDrawingMessage();
                return;
            }

            const currentImageData = this.getCanvasData();
            const currentHash = this.hashString(currentImageData);

            if (!forcePrediction && currentHash === this.lastImageHash) {
                console.log('Skipping prediction - image unchanged');
                return;
            }

            this.lastImageHash = currentHash;
            this.isRequesting = true;
            const predictionId = ++this.lastPredictionId;

            this.showPredictionLoading();

            console.log('Sending MNIST prediction request...');

            const response = await fetch('/api/predict_mnist', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_data: currentImageData
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (predictionId !== this.lastPredictionId) {
                console.log('Ignoring outdated prediction result');
                return;
            }

            if (result.status === 'success') {
                this.displayPrediction(result);
            } else {
                this.showPredictionError(result.message || 'Prediction failed');
            }

        } catch (error) {
            console.error('Prediction error:', error);
            this.showPredictionError('Network error: ' + error.message);
        } finally {
            this.isRequesting = false;
        }
    }

    displayPrediction(result) {
        const predictedDigit = result.predicted_digit;
        const confidence = result.confidence;

        const digitEl = document.getElementById('predictedDigit');
        const scoreEl = document.getElementById('confidenceScore');

        if (digitEl) {
            digitEl.textContent = predictedDigit >= 0 ? predictedDigit : '?';
        }

        if (scoreEl) {
            scoreEl.textContent = predictedDigit >= 0
                ? `Confidence: ${this.formatPercentage(confidence)}`
                : 'No clear prediction';
        }

        this.updateStatusIndicator(confidence);
        this.updateConfidenceBars(result.predictions);

        if (result.processed_image_url) {
            this.showProcessedImage(result.processed_image_url);
        }
    }

    updateStatusIndicator(confidence) {
        const statusIndicator = document.getElementById('statusIndicator');
        if (!statusIndicator) return;

        if (confidence > 0.8) {
            statusIndicator.className = 'status-indicator status-confident';
            statusIndicator.textContent = 'High Confidence';
        } else if (confidence > 0.5) {
            statusIndicator.className = 'status-indicator status-uncertain';
            statusIndicator.textContent = 'Medium Confidence';
        } else {
            statusIndicator.className = 'status-indicator status-unclear';
            statusIndicator.textContent = 'Low Confidence';
        }
    }

    updateConfidenceBars(predictions) {
        const container = document.getElementById('confidenceBars');
        if (!container) return;

        container.innerHTML = '';

        const sortedPredictions = [...predictions].sort((a, b) => a.digit - b.digit);

        sortedPredictions.forEach((pred, index) => {
            const digitPrediction = document.createElement('div');
            digitPrediction.className = 'digit-prediction';

            setTimeout(() => {
                digitPrediction.innerHTML = `
                    <div class="digit-label">${pred.digit}</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fill" style="width: ${pred.percentage.toFixed(1)}%"></div>
                    </div>
                    <div class="confidence-percentage">${pred.percentage.toFixed(1)}%</div>
                `;
            }, index * 40);

            container.appendChild(digitPrediction);
        });
    }

    initializeConfidenceBars() {
        const container = document.getElementById('confidenceBars');
        if (!container) return;

        container.innerHTML = '';

        for (let digit = 0; digit < 10; digit++) {
            const digitPrediction = document.createElement('div');
            digitPrediction.className = 'digit-prediction';
            digitPrediction.innerHTML = `
                <div class="digit-label">${digit}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: 10%"></div>
                </div>
                <div class="confidence-percentage">10.0%</div>
            `;
            container.appendChild(digitPrediction);
        }
    }

    showPredictionLoading() {
        const digitEl = document.getElementById('predictedDigit');
        const scoreEl = document.getElementById('confidenceScore');
        const statusEl = document.getElementById('statusIndicator');

        if (digitEl) {
            digitEl.innerHTML = '<div class="loading-spinner"></div>';
        }
        if (scoreEl) {
            scoreEl.textContent = 'Analyzing drawing...';
        }
        if (statusEl) {
            statusEl.className = 'status-indicator status-uncertain';
            statusEl.textContent = 'Processing...';
        }
    }

    showNoDrawingMessage() {
        const digitEl = document.getElementById('predictedDigit');
        const scoreEl = document.getElementById('confidenceScore');
        const statusEl = document.getElementById('statusIndicator');

        if (digitEl) digitEl.textContent = '?';
        if (scoreEl) scoreEl.textContent = 'Please draw a digit first';
        if (statusEl) {
            statusEl.className = 'status-indicator status-unclear';
            statusEl.textContent = 'No input';
        }
    }

    showPredictionError(message) {
        const digitEl = document.getElementById('predictedDigit');
        const scoreEl = document.getElementById('confidenceScore');
        const statusEl = document.getElementById('statusIndicator');

        if (digitEl) digitEl.textContent = 'Error';
        if (scoreEl) scoreEl.textContent = 'Error: ' + message;
        if (statusEl) {
            statusEl.className = 'status-indicator status-unclear';
            statusEl.textContent = 'Error';
        }

        if (window.CNNPlayground && window.CNNPlayground.showError) {
            window.CNNPlayground.showError('Prediction failed: ' + message);
        }
    }

    showProcessedImage(imageUrl) {
        const container = document.getElementById('processedImageContainer');
        const image = document.getElementById('processedImage');

        if (container && image) {
            const cacheBuster = `?pid=${this.lastPredictionId}&t=${Date.now()}`;
            image.src = imageUrl + cacheBuster;
            container.style.display = 'block';

            image.onload = () => {
                console.log('Processed image loaded successfully');
            };

            image.onerror = () => {
                console.log('Failed to load processed image');
                container.style.display = 'none';
            };
        }
    }

    clear() {
        // Reset everything
        this.allStrokes = [];
        this.currentStroke = [];
        this.lastImageHash = null;
        this.lastPredictionId++;

        // Reset to draw mode
        this.isEraseMode = false;
        this.updateEraseButton();
        this.updateSliderLabel();

        // Upewnij się, że canvas jest w prawidłowym trybie
        this.updateCanvasMode();

        // Clear canvas
        this.clearCanvas();

        this.cancelPrediction();

        // Show overlay
        const overlay = document.getElementById('canvasOverlay');
        if (overlay) {
            overlay.classList.add('show');
        }

        // Reset UI
        this.resetPredictions();
        this.hideProcessedImage();

        console.log('Canvas cleared and reset to draw mode');
    }

    resetPredictions() {
        const digitEl = document.getElementById('predictedDigit');
        const scoreEl = document.getElementById('confidenceScore');
        const statusEl = document.getElementById('statusIndicator');

        if (digitEl) digitEl.textContent = '?';
        if (scoreEl) scoreEl.textContent = 'Draw a digit to see prediction';
        if (statusEl) {
            statusEl.className = 'status-indicator status-unclear';
            statusEl.textContent = 'Waiting for input';
        }

        this.initializeConfidenceBars();
    }

    hideProcessedImage() {
        const container = document.getElementById('processedImageContainer');
        if (container) {
            container.style.display = 'none';
        }
    }

    // Utility methods
    formatPercentage(num) {
        return `${(num * 100).toFixed(1)}%`;
    }

    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash;
    }

    // Public API
    enableAutoPrediction() {
        this.autoPredictEnabled = true;
    }

    disableAutoPrediction() {
        this.autoPredictEnabled = false;
        this.cancelPrediction();
    }

    getCanvasData() {
        // Przed pobraniem danych, upewnij się że wszystkie operacje są zastosowane
        this.ensureCanvasUpdated();

        // Utwórz tymczasowy canvas z białym tłem dla export
        return this.getCanvasDataWithWhiteBackground();
    }

    getCanvasDataWithWhiteBackground() {
        // Utwórz tymczasowy canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        // Narysuj białe tło
        tempCtx.fillStyle = '#ffffff';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

        // Skopiuj zawartość oryginalnego canvas
        tempCtx.drawImage(this.canvas, 0, 0);

        // Zwróć dane z białym tłem
        return tempCanvas.toDataURL('image/png');
    }

    ensureCanvasUpdated() {
        // Jeśli są jakieś operacje wymazywania, przerysuj wszystko
        const hasEraseStrokes = this.allStrokes.some(stroke =>
            stroke.length > 0 && stroke[0].isErase
        );

        if (hasEraseStrokes) {
            console.log('Ensuring canvas is updated with erase operations');
            this.redrawAllStrokes();
        }
    }

    hasDrawing() {
        return this.allStrokes.length > 0;
    }

    isCurrentlyPredicting() {
        return this.isRequesting;
    }

}

window.MNISTCanvas = MNISTCanvas;