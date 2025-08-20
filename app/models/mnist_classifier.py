#!/usr/bin/env python3
"""
MNIST Digit Classifier - Production Version
Loads pre-trained model without training functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import base64
import io
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedLeNet(nn.Module):
    """
    Enhanced LeNet-5 with BatchNorm and Dropout for better accuracy
    Same architecture as in training notebook
    """

    def __init__(self, num_classes=10):
        super(ImprovedLeNet, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Classification
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Conv layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, 128 * 3 * 3)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)

        return x


class MNISTClassifier:
    """
    Production MNIST digit classifier - loads pre-trained model only
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing MNIST classifier on {self.device}")

        # Initialize model
        self.model = ImprovedLeNet()
        self.model.to(self.device)

        # Resolve and load model
        self.model_path = self._resolve_model_path(model_path)
        self._load_model()

        # Set model to evaluation mode for inference
        self.model.eval()

        # Preprocessing pipeline matching MNIST training data
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset statistics
        ])

        logger.info("MNIST classifier initialized successfully")

    def _resolve_model_path(self, model_path):
        """Find the pre-trained model file"""
        if model_path and Path(model_path).exists():
            return model_path

        # Use absolute paths relative to this file
        models_dir = Path(__file__).parent / "saved_models"

        # Try different possible locations
        possible_paths = [
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

    def _load_model(self):
        """Load pre-trained model - PRODUCTION ONLY, NO TRAINING"""
        try:
            logger.info(f"Loading pre-trained model from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with metadata
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Log model info if available
                accuracy = checkpoint.get('test_accuracy', 'Unknown')
                training_date = checkpoint.get('training_date', 'Unknown')
                architecture = checkpoint.get('architecture', 'Unknown')

                logger.info(f"✅ Model loaded successfully!")
                logger.info(f"   Architecture: {architecture}")
                logger.info(f"   Test Accuracy: {accuracy}")
                logger.info(f"   Training Date: {training_date}")

            elif isinstance(checkpoint, dict):
                # Assume it's just the state dict
                self.model.load_state_dict(checkpoint)
                logger.info("✅ Model state dict loaded successfully!")

            else:
                raise ValueError("Unsupported model format")

        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from {self.model_path}: {e}")

    def preprocess_canvas_image(self, base64_data):
        """
        Convert base64 canvas image to model input tensor
        Handles the white background to black background conversion
        """
        try:
            # Decode base64 image
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]

            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))

            # Debug mode
            debug_mode = os.getenv('DEBUG_MNIST', 'false').lower() == 'true'
            if debug_mode:
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                image.save(debug_dir / "01_original.png")

            # Convert to RGB first, then grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to grayscale
            image_gray = image.convert('L')

            # Check if image is mostly empty
            image_array = np.array(image_gray)
            white_pixels = np.sum(image_array > 240)
            total_pixels = image_array.size
            white_percentage = (white_pixels / total_pixels) * 100

            if white_percentage > 95:
                logger.warning("Image is mostly white - may be empty")

            # Invert colors (canvas white background -> MNIST black background)
            image_inverted = 255 - image_array
            image_inverted_pil = Image.fromarray(image_inverted)

            if debug_mode:
                image_inverted_pil.save(debug_dir / "02_inverted.png")

            # Apply preprocessing pipeline
            input_tensor = self.transform(image_inverted_pil)

            if debug_mode:
                # Save final tensor visualization
                tensor_np = input_tensor.squeeze().numpy()
                tensor_denorm = (tensor_np * 0.3081) + 0.1307
                tensor_denorm = np.clip(tensor_denorm * 255, 0, 255).astype(np.uint8)
                Image.fromarray(tensor_denorm).save(debug_dir / "03_final_tensor.png")

            input_batch = input_tensor.unsqueeze(0).to(self.device)
            return input_batch, image_inverted_pil

        except Exception as e:
            logger.error(f"Error preprocessing canvas image: {e}")
            raise

    def predict_digit(self, base64_image_data):
        """
        Predict digit from base64 encoded canvas image
        Returns deterministic results with confidence scores
        """
        try:
            # Set deterministic behavior
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

            # Preprocess image
            input_tensor, processed_image = self.preprocess_canvas_image(base64_image_data)

            # Inference in evaluation mode
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

                # Get prediction
                predicted_class = outputs.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                # Get all class probabilities
                all_probs = probabilities[0].cpu().numpy()

                # Prepare predictions list
                predictions = []
                for digit in range(10):
                    predictions.append({
                        'digit': digit,
                        'probability': float(all_probs[digit]),
                        'percentage': float(all_probs[digit] * 100)
                    })

                # Sort by confidence
                predictions.sort(key=lambda x: x['probability'], reverse=True)

                # Save processed image for debugging
                processed_path = f"static/outputs/mnist/processed_mnist_{predicted_class}.png"
                Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
                processed_image.save(processed_path)

                result = {
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

    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'architecture': 'Improved LeNet-5',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': '28x28 grayscale',
            'output_classes': 10,
            'device': str(self.device),
            'preprocessing': 'Normalize to MNIST statistics',
            'model_path': str(self.model_path)
        }


# Global classifier instance for API endpoints
_mnist_classifier = None


def get_mnist_classifier():
    """Get or create global MNIST classifier instance"""
    global _mnist_classifier
    if _mnist_classifier is None:
        _mnist_classifier = MNISTClassifier()
    return _mnist_classifier


def predict_mnist_digit(image_data):
    """API endpoint function for MNIST digit prediction"""
    classifier = get_mnist_classifier()
    return classifier.predict_digit(image_data)


def get_mnist_model_info():
    """API endpoint function for model information"""
    classifier = get_mnist_classifier()
    return classifier.get_model_info()


if __name__ == "__main__":
    # Test the classifier
    print("Testing MNIST Classifier...")
    try:
        classifier = MNISTClassifier()
        print("✅ Model loaded successfully!")
        print("Model Info:", classifier.get_model_info())
        print("🚀 Classifier ready for predictions!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure your trained model is in the models/ directory.")