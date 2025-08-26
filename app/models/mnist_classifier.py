#!/usr/bin/env python3
"""
MNIST Digit Classifier

MNIST digit classification system with pre-trained CNN model.
Provides digit recognition from canvas drawings with confidence scoring
and model interpretation capabilities.

Features:
- Pre-trained CNN model loading and inference
- Canvas image preprocessing with color inversion
- Top-10 prediction confidence scoring
- Debug visualization capabilities
- Error handling and logging
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import base64
import io
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network architecture for MNIST digit classification.

    CNN implementation featuring batch normalization, dropout
    regularization, and feature extraction through multiple
    convolutional layers. Optimized for 28x28 grayscale digit recognition.

    Architecture:
        - 3 Convolutional layers (32, 64, 128 filters)
        - Batch normalization after each conv layer
        - MaxPooling for spatial dimension reduction
        - 3 Fully connected layers with dropout
        - Final classification layer

    Args:
        num_classes: Number of output classes (default: 10 for digits 0-9)

    Attributes:
        conv1-3 (nn.Conv2d): Convolutional feature extraction layers
        bn1-3 (nn.BatchNorm2d): Batch normalization layers
        fc1-3 (nn.Linear): Fully connected classification layers
        pool (nn.MaxPool2d): Max pooling operation
        dropout (nn.Dropout): Spatial dropout for regularization
        dropout_fc (nn.Dropout): FC layer dropout for regularization
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize MNIST CNN architecture with specified output classes.

        Args:
            num_classes: Number of classification classes (default: 10)
        """
        super(MNISTNet, self).__init__()

        # Feature extraction layers
        self.conv1: nn.Conv2d = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(32)
        self.conv2: nn.Conv2d = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.conv3: nn.Conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3: nn.BatchNorm2d = nn.BatchNorm2d(128)

        self.pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        self.dropout: nn.Dropout = nn.Dropout(0.25)

        # Classification layers
        self.fc1: nn.Linear = nn.Linear(128 * 3 * 3, 256)
        self.fc2: nn.Linear = nn.Linear(256, 128)
        self.fc3: nn.Linear = nn.Linear(128, num_classes)
        self.dropout_fc: nn.Dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN architecture.

        Performs feature extraction through convolutional layers followed
        by classification through fully connected layers. Applies
        activation functions, normalization, and regularization.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits tensor of shape (batch_size, num_classes)

        Note:
            Returns raw logits - apply softmax for probabilities
        """
        # Convolutional feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 3 * 3)

        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)

        return x


class MNISTClassifier:
    """
    MNIST digit classifier with pre-trained model inference.

    Digit recognition system that handles canvas image preprocessing,
    model loading, inference, and provides prediction results with
    confidence scoring.

    Features:
        - Automatic pre-trained model discovery and loading
        - Canvas image preprocessing with color inversion
        - Deterministic inference with confidence scoring
        - Debug visualization capabilities
        - Error handling and logging

    Attributes:
        device (torch.device): Computing device (CPU/CUDA)
        model (MNISTNet): Loaded pre-trained CNN model
        model_path (str): Path to loaded model file
        transform (transforms.Compose): Image preprocessing pipeline
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None) -> None:
        """
        Initialize MNIST classifier with pre-trained model loading.

        Args:
            model_path: Optional path to specific model file
            device: Optional computing device (auto-detected if None)

        Raises:
            FileNotFoundError: If no pre-trained model found
            RuntimeError: If model loading fails
        """
        self.device: torch.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing MNIST classifier on {self.device}")

        # Initialize model architecture
        self.model: MNISTNet = MNISTNet()
        self.model.to(self.device)

        # Resolve and load pre-trained model
        self.model_path: str = self._resolve_model_path(model_path)
        self._load_model()

        # Set evaluation mode for inference
        self.model.eval()

        # Preprocessing pipeline matching MNIST training statistics
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset statistics
        ])

        logger.info("MNIST classifier initialized successfully")

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """
        Locate pre-trained model file using multiple search strategies.

        Searches common locations for pre-trained model files using
        hierarchical directory structure. Provides helpful error messages
        if no model is found.

        Args:
            model_path: Optional explicit path to model file

        Returns:
            Resolved absolute path to model file

        Raises:
            FileNotFoundError: If no model found in any search location
        """
        if model_path and Path(model_path).exists():
            return model_path

        # Use absolute paths relative to this file
        models_dir: Path = Path(__file__).parent / "saved_models"

        # Try different possible locations
        possible_paths: List[Path] = [
            models_dir / "mnist_model.pth",  # app/models/saved_models/
            Path(__file__).parent.parent / "mnist_model.pth",  # app/ folder
            Path("mnist_model.pth"),  # current directory
            Path("models") / "mnist_model.pth",  # legacy location
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found pre-trained model at: {path}")
                return str(path)

        # If no model found, create helpful error message
        raise FileNotFoundError(
            f"Pre-trained MNIST model not found!\n"
            f"Searched locations:\n" +
            "\n".join(f"  - {path}" for path in possible_paths) +
            f"\n\nPlease ensure mnist_model.pth is in: {models_dir}"
        )

    def _load_model(self) -> None:
        """
        Load pre-trained model weights and metadata.

        Handles multiple checkpoint formats including full training metadata
        and bare state dictionaries. Logs model information when available.

        Raises:
            RuntimeError: If model loading or format validation fails
        """
        try:
            logger.info(f"Loading model from: {self.model_path}")
            checkpoint: Union[Dict[str, Any], torch.nn.Module] = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with metadata
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Log model info if available
                accuracy: Union[str, float] = checkpoint.get('test_accuracy', 'Unknown')
                training_date: str = checkpoint.get('training_date', 'Unknown')
                architecture: str = checkpoint.get('architecture', 'Unknown')

                logger.info("Model loaded successfully")
                logger.info(f"   Architecture: {architecture}")
                logger.info(f"   Test Accuracy: {accuracy}")
                logger.info(f"   Training Date: {training_date}")

            elif isinstance(checkpoint, dict):
                # Assume it's just the state dict
                self.model.load_state_dict(checkpoint)
                logger.info("Model state dict loaded successfully")

            else:
                raise ValueError("Unsupported model format")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def preprocess_canvas_image(self, base64_data: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Convert base64 canvas image to model-ready tensor with preprocessing.

        Preprocessing pipeline for canvas drawings including:
        - Base64 decoding and image loading
        - Color space conversion and inversion
        - Empty image detection and warnings
        - MNIST-compatible normalization
        - Optional debug visualization

        Args:
            base64_data: Base64 encoded image data from HTML5 canvas

        Returns:
            Tuple containing:
                - Preprocessed input tensor ready for model inference
                - PIL Image of processed grayscale digit

        Raises:
            ValueError: If base64 decoding fails
            IOError: If image processing fails
        """
        try:
            # Decode base64 image
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]

            image_data: bytes = base64.b64decode(base64_data)
            image: Image.Image = Image.open(io.BytesIO(image_data))

            # Debug mode for development
            debug_mode: bool = os.getenv('DEBUG_MNIST', 'false').lower() == 'true'
            if debug_mode:
                debug_dir: Path = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                image.save(debug_dir / "01_original.png")

            # Convert to RGB first, then grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to grayscale
            image_gray: Image.Image = image.convert('L')

            # Check if image is mostly empty
            image_array: np.ndarray = np.array(image_gray)
            white_pixels: int = np.sum(image_array > 240)
            total_pixels: int = image_array.size
            white_percentage: float = (white_pixels / total_pixels) * 100

            if white_percentage > 95:
                logger.warning("Image is mostly white - may be empty")

            # Invert colors (canvas white background -> MNIST black background)
            image_inverted: np.ndarray = 255 - image_array
            image_inverted_pil: Image.Image = Image.fromarray(image_inverted)

            if debug_mode:
                image_inverted_pil.save(debug_dir / "02_inverted.png")

            # Apply preprocessing pipeline
            input_tensor: torch.Tensor = self.transform(image_inverted_pil)

            if debug_mode:
                # Save final tensor visualization
                tensor_np: np.ndarray = input_tensor.squeeze().numpy()
                tensor_denorm: np.ndarray = (tensor_np * 0.3081) + 0.1307
                tensor_denorm = np.clip(tensor_denorm * 255, 0, 255).astype(np.uint8)
                Image.fromarray(tensor_denorm).save(debug_dir / "03_final_tensor.png")

            input_batch: torch.Tensor = input_tensor.unsqueeze(0).to(self.device)
            return input_batch, image_inverted_pil

        except Exception as e:
            logger.error(f"Error preprocessing canvas image: {e}")
            raise

    def predict_digit(self, base64_image_data: str) -> Dict[str, Any]:
        """
        Perform digit prediction from base64 canvas image with confidence scoring.

        Prediction pipeline including preprocessing, inference,
        confidence calculation, and result formatting. Provides deterministic
        results with prediction metadata.

        Args:
            base64_image_data: Base64 encoded canvas image data

        Returns:
            Prediction results dictionary containing:
                - status: Prediction completion status
                - predicted_digit: Most likely digit (0-9)
                - confidence: Prediction confidence score (0-1)
                - predictions: All digit probabilities ranked by confidence
                - processed_image_url: URL to processed image visualization
                - model_info: Model architecture and processing metadata

        Raises:
            Exception: Any prediction error (handled gracefully with error response)
        """
        try:
            # Set deterministic behavior for reproducible results
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

            # Preprocess image
            input_tensor: torch.Tensor
            processed_image: Image.Image
            input_tensor, processed_image = self.preprocess_canvas_image(base64_image_data)

            # Inference in evaluation mode
            self.model.eval()
            with torch.no_grad():
                outputs: torch.Tensor = self.model(input_tensor)
                probabilities: torch.Tensor = F.softmax(outputs, dim=1)

                # Get prediction
                predicted_class: int = outputs.argmax(dim=1).item()
                confidence: float = probabilities[0][predicted_class].item()

                # Get all class probabilities
                all_probs: np.ndarray = probabilities[0].cpu().numpy()

                # Prepare predictions list
                predictions: List[Dict[str, Union[int, float]]] = []
                for digit in range(10):
                    predictions.append({
                        'digit': digit,
                        'probability': float(all_probs[digit]),
                        'percentage': float(all_probs[digit] * 100)
                    })

                # Sort by confidence
                predictions.sort(key=lambda x: x['probability'], reverse=True)

                # Save processed image for debugging
                processed_path: str = f"static/outputs/mnist/processed_mnist_{predicted_class}.png"
                Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
                processed_image.save(processed_path)

                result: Dict[str, Any] = {
                    'status': 'success',
                    'predicted_digit': int(predicted_class),
                    'confidence': float(confidence),
                    'predictions': predictions,
                    'processed_image_url': f"/outputs/mnist/processed_mnist_{predicted_class}.png",
                    'model_info': self.get_model_info()
                }

                logger.info(f"Predicted digit: {predicted_class} with confidence: {confidence:.3f}")
                return result

        except Exception as e:
            logger.error(f"Error in digit prediction: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'predicted_digit': -1,
                'confidence': 0.0,
                'predictions': []
            }

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Retrieve model architecture and configuration information.

        Provides detailed metadata about the loaded model including parameter
        counts, architecture details, and processing configuration for
        debugging and monitoring.

        Returns:
            Model information dictionary containing:
                - architecture: Model architecture name
                - total_parameters: Total number of model parameters
                - trainable_parameters: Number of trainable parameters
                - input_size: Expected input dimensions
                - output_classes: Number of output classes
                - device: Computing device information
                - preprocessing: Preprocessing pipeline description
                - model_path: Path to loaded model file
        """
        total_params: int = sum(p.numel() for p in self.model.parameters())
        trainable_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'architecture': 'MNISTNet CNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': '28x28 grayscale',
            'output_classes': 10,
            'device': str(self.device),
            'preprocessing': 'Normalize to MNIST statistics',
            'model_path': str(self.model_path)
        }


# Global classifier instance for API endpoints
_mnist_classifier: Optional[MNISTClassifier] = None


def get_mnist_classifier() -> MNISTClassifier:
    """
    Get or create global MNIST classifier instance using singleton pattern.

    Implements lazy initialization for efficient resource management.
    Reuses existing classifier instance to avoid repeated model loading.

    Returns:
        Global MNISTClassifier instance ready for predictions
    """
    global _mnist_classifier
    if _mnist_classifier is None:
        _mnist_classifier = MNISTClassifier()
    return _mnist_classifier


def predict_mnist_digit(image_data: str) -> Dict[str, Any]:
    """
    FastAPI endpoint function for MNIST digit prediction.

    Primary API entry point for digit recognition requests.
    Handles image preprocessing, model inference, and result formatting
    using the global classifier instance.

    Args:
        image_data: Base64 encoded canvas image data

    Returns:
        Prediction results dictionary suitable for JSON API response

    Note:
        Uses global classifier instance for efficient resource utilization
        across multiple API requests.
    """
    classifier: MNISTClassifier = get_mnist_classifier()
    return classifier.predict_digit(image_data)


def get_mnist_model_info() -> Dict[str, Union[str, int]]:
    """
    FastAPI endpoint function for model architecture information.

    Provides model metadata and configuration details for API clients.
    Useful for debugging, monitoring, and feature detection.

    Returns:
        Model information dictionary with architecture details
    """
    classifier: MNISTClassifier = get_mnist_classifier()
    return classifier.get_model_info()


if __name__ == "__main__":
    # Test the classifier functionality
    print("Testing MNIST Classifier...")
    try:
        classifier: MNISTClassifier = MNISTClassifier()
        print("Model loaded successfully")
        print("Model Info:", classifier.get_model_info())
        print("Classifier ready for predictions")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure your trained model is in the models/ directory.")