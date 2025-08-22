"""
CNN Playground Models Package
Professional ML models for image classification
"""
__version__ = "2.0.0"

# Import availability flags
CNN_ANALYZER_AVAILABLE = False
MNIST_CLASSIFIER_AVAILABLE = False
VISION_ANALYZER_AVAILABLE = False

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


def get_available_features():
    """Get available model features"""
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


def get_system_status():
    """Get comprehensive system status"""
    features = get_available_features()

    status = {
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


# Public API
__all__ = [
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
