"""
CNN Playground Models Package

ML models for image classification, object detection, and instance segmentation.
Provides unified interface for computer vision tasks with automatic model loading
and graceful fallback handling.
"""

from typing import Dict, Union, List, Any, Optional

__version__: str = "1.0.0"

# Import availability flags
CNN_ANALYZER_AVAILABLE: bool = False
MNIST_CLASSIFIER_AVAILABLE: bool = False
VISION_ANALYZER_AVAILABLE: bool = False

# CNN Analyzer imports - RELATIVE IMPORTS
try:
    from .cnn_analyzer import analyze_image_endpoint, get_analyzer

    CNN_ANALYZER_AVAILABLE = True
    print("[MODELS] CNN Analyzer loaded successfully")
except ImportError as e:
    print(f"[MODELS] CNN Analyzer not available: {e}")

# MNIST Classifier imports - RELATIVE IMPORTS
try:
    from .mnist_classifier import predict_mnist_digit, get_mnist_classifier, get_mnist_model_info

    MNIST_CLASSIFIER_AVAILABLE = True
    print("[MODELS] MNIST Classifier loaded successfully")
except ImportError as e:
    print(f"[MODELS] MNIST Classifier not available: {e}")

# Vision Analyzer imports - NEW
try:
    from .vision_analyzer import (
        VisionAnalyzer,
        ObjectDetector,
        ImageSegmenter,
        analyze_vision_complete,
        process_video_complete,
        process_live_frame,
        get_sota_vision_analyzer
    )

    VISION_ANALYZER_AVAILABLE = True
    print("[MODELS] Vision Analyzer loaded successfully")
except ImportError as e:
    print(f"[MODELS] Vision Analyzer not available: {e}")


def get_available_features() -> Dict[str, bool]:
    """
    Get available model features based on successful imports.

    Checks which ML models and features are available based on
    successful module imports and dependency availability.

    Returns:
        Dictionary mapping feature names to availability status:
            - imagenet_classification: ResNet50 ImageNet classification
            - feature_maps_visualization: CNN layer feature map extraction
            - gradcam_attention: Grad-CAM attention heatmap generation
            - mnist_digit_recognition: MNIST handwritten digit recognition
            - object_detection: YOLO/RT-DETR object detection
            - instance_segmentation: YOLO-seg/SAM2 instance segmentation
            - video_analysis: Frame-by-frame video processing
            - live_camera_detection: Real-time camera feed processing
    """
    return {
        'imagenet_classification': CNN_ANALYZER_AVAILABLE,
        'feature_maps_visualization': CNN_ANALYZER_AVAILABLE,
        'gradcam_attention': CNN_ANALYZER_AVAILABLE,
        'mnist_digit_recognition': MNIST_CLASSIFIER_AVAILABLE,
        'object_detection': VISION_ANALYZER_AVAILABLE,
        'instance_segmentation': VISION_ANALYZER_AVAILABLE,
        'video_analysis': VISION_ANALYZER_AVAILABLE,
        'live_camera_detection': VISION_ANALYZER_AVAILABLE,
    }


def get_system_status() -> Dict[str, Any]:
    """
    Get system status with detailed feature availability and model information.

    Provides overview of package version, available features,
    and model-specific metadata for system monitoring and debugging.

    Returns:
        System status dictionary containing:
            - package_version: Current package version string
            - total_features: Total number of available features
            - available_features: Count of successfully loaded features
            - features: Feature availability mapping from get_available_features()
            - mnist_model_info: MNIST model metadata (if available)
            - mnist_model_error: Error message if MNIST model failed to load

    Note:
        Gracefully handles model loading errors and provides diagnostic information
    """
    features: Dict[str, bool] = get_available_features()

    status: Dict[str, Any] = {
        'package_version': __version__,
        'total_features': len(features),
        'available_features': sum(features.values()),
        'features': features
    }

    # Add MNIST model info if available
    if MNIST_CLASSIFIER_AVAILABLE:
        try:
            status['mnist_model_info'] = get_mnist_model_info()
        except Exception as e:
            status['mnist_model_error'] = str(e)

    return status


# Public API exports
__all__: List[str] = [
    'get_available_features',
    'get_system_status',
    'CNN_ANALYZER_AVAILABLE',
    'MNIST_CLASSIFIER_AVAILABLE',
    'VISION_ANALYZER_AVAILABLE',
]

# Add available functions to exports
if CNN_ANALYZER_AVAILABLE:
    __all__.extend(['analyze_image_endpoint', 'get_analyzer'])

if MNIST_CLASSIFIER_AVAILABLE:
    __all__.extend(['predict_mnist_digit', 'get_mnist_classifier', 'get_mnist_model_info'])

if VISION_ANALYZER_AVAILABLE:
    __all__.extend([
        'VisionAnalyzer',
        'ObjectDetector',
        'ImageSegmenter',
        'analyze_vision_complete',
        'process_video_complete',
        'process_live_frame',
        'get_sota_vision_analyzer'
    ])
