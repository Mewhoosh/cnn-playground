#!/usr/bin/env python3
"""
CNN Analyzer with Feature Maps Visualization - FIXED VERSION
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageOps
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import json


class CNNFeatureAnalyzer:
    def __init__(self, output_dir="static/outputs"):
        """Initialize the CNN analyzer with feature maps extraction"""
        print("[CNN] Loading ResNet50 model with feature extraction...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CNN] Using device: {self.device}")

        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)

        # Normalize only
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Output directory - FIXED TO USE CORRECT PATH
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CNN] Output directory: {self.output_dir}")

        # Feature maps storage
        self.feature_maps = {}
        self.original_image = None

        # Layer info
        self.layer_info = {
            'conv1': {'size': 112, 'channels': 64},
            'conv2_1': {'size': 56, 'channels': 256},
            'conv3_1': {'size': 28, 'channels': 512},
            'conv4_1': {'size': 14, 'channels': 1024},
            'conv5_1': {'size': 7, 'channels': 2048}
        }

        # Load ImageNet class labels
        self.class_labels = self._load_imagenet_labels()

        print("[CNN] CNN Feature Analyzer initialized successfully")

    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        try:
            labels_file = self.output_dir / "imagenet_classes.txt"

            if not labels_file.exists():
                print("[CNN] Downloading ImageNet class labels...")
                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                urllib.request.urlretrieve(url, labels_file)

            with open(labels_file, 'r') as f:
                labels = [line.strip() for line in f.readlines()]

            return {i: labels[i] for i in range(len(labels))}

        except Exception as e:
            print(f"[CNN] Could not download labels: {e}")
            return {i: f"class_{i}" for i in range(1000)}

    def resize_with_padding(self, image, target_size=224):
        """Resize image preserving aspect ratio and add padding to make it square"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        orig_width, orig_height = image.size
        print(f"[CNN] Original size: {orig_width}x{orig_height}")

        scale = min(target_size / orig_width, target_size / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        print(f"[CNN] Scaled size: {new_width}x{new_height}")

        image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_width = (target_size - new_width) // 2
        pad_height = (target_size - new_height) // 2

        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        padded_image.paste(image_resized, (pad_width, pad_height))

        padding_info = {
            'pad_left': pad_width,
            'pad_top': pad_height,
            'pad_right': target_size - new_width - pad_width,
            'pad_bottom': target_size - new_height - pad_height,
            'content_width': new_width,
            'content_height': new_height,
            'original_width': orig_width,
            'original_height': orig_height,
            'scale': scale
        }

        return padded_image, padding_info

    def setup_feature_hooks(self):
        """Setup hooks to capture feature maps from ResNet50"""

        def hook_function(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach().cpu()

            return hook

        # Register hooks for ResNet50 layers
        hooks = []
        hooks.append(self.model.conv1.register_forward_hook(hook_function('conv1')))
        hooks.append(self.model.layer1.register_forward_hook(hook_function('conv2_1')))
        hooks.append(self.model.layer2.register_forward_hook(hook_function('conv3_1')))
        hooks.append(self.model.layer3.register_forward_hook(hook_function('conv4_1')))
        hooks.append(self.model.layer4.register_forward_hook(hook_function('conv5_1')))

        return hooks

    def create_overview_image(self, original_image, filename_prefix="overview"):
        """Create overview with original + one feature map from each layer"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CNN Analysis Overview', fontsize=16, fontweight='bold')

        # Original image
        ax = axes[0, 0]
        ax.imshow(np.array(original_image))
        ax.set_title('Original Image', fontweight='bold')
        ax.axis('off')

        # Feature maps from each layer
        layer_names = ['conv1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        layer_titles = ['Conv1', 'Conv2_1', 'Conv3_1', 'Conv4_1', 'Conv5_1']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for idx, (layer_name, title, (row, col)) in enumerate(zip(layer_names, layer_titles, positions)):
            ax = axes[row, col]

            if layer_name in self.feature_maps:
                # Get average activation across channels
                features = self.feature_maps[layer_name][0]
                avg_activation = torch.mean(features, dim=0).numpy()
                avg_activation = (avg_activation - avg_activation.min()) / (
                        avg_activation.max() - avg_activation.min() + 1e-8)

                # Show without colorbar
                ax.imshow(avg_activation, cmap='viridis', interpolation='nearest')

                # Add size info
                size_info = f"{self.layer_info[layer_name]['size']}×{self.layer_info[layer_name]['size']}\n{self.layer_info[layer_name]['channels']} channels"
                ax.text(0.02, 0.98, size_info, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)

            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        feature_maps_dir = self.output_dir / "feature_maps"
        output_path = feature_maps_dir / f"{filename_prefix}_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[CNN] Saved overview to: {output_path}")
        return output_path

    def visualize_layer_activations(self, layer_name, max_channels=8, filename_prefix="layer"):
        """Show top 8 activations for specific layer"""
        if layer_name not in self.feature_maps:
            print(f"[CNN] No feature maps found for layer: {layer_name}")
            return None

        features = self.feature_maps[layer_name][0]  # Remove batch dimension

        # Calculate activation strength for each channel
        channel_activations = []
        for i in range(features.shape[0]):
            activation_strength = torch.sum(torch.relu(features[i])).item()
            channel_activations.append((i, activation_strength))

        # Sort by activation strength and take top channels
        channel_activations.sort(key=lambda x: x[1], reverse=True)
        top_channels = [idx for idx, _ in channel_activations[:max_channels]]

        # Create 2x4 grid
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'{layer_name.upper()} - Top {max_channels} Most Active Filters',
                     fontsize=14, fontweight='bold')

        plt.subplots_adjust(hspace=0.3, wspace=0.2)

        for i in range(8):
            row = i // 4
            col = i % 4
            ax = axes[row, col]

            if i < len(top_channels):
                channel_idx = top_channels[i]

                # Get feature map and normalize
                feature_map = features[channel_idx].numpy()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

                # Display without colorbar
                ax.imshow(feature_map, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Filter {channel_idx + 1}', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.set_title('N/A', fontsize=10)

            ax.axis('off')

        # Add layer info
        info_text = f"Layer: {layer_name} | Size: {self.layer_info[layer_name]['size']}×{self.layer_info[layer_name]['size']} | Channels: {features.shape[0]}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Save visualization
        feature_maps_dir = self.output_dir / "feature_maps"
        output_path = feature_maps_dir / f"{filename_prefix}_{layer_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[CNN] Saved {layer_name} activations to: {output_path}")
        return output_path

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image"""
        print(f"[CNN] Loading image: {image_path}")

        original_image = Image.open(image_path).convert('RGB')
        self.original_image = original_image  # Store for overview

        padded_image, padding_info = self.resize_with_padding(original_image)

        input_tensor = self.normalize(padded_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        return original_image, padded_image, input_batch, padding_info

    def predict(self, input_batch):
        """Make prediction and return top-5 results"""
        print("[CNN] Making prediction...")

        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_idx = torch.topk(probabilities, 5)

        predictions = []
        for i in range(5):
            class_idx = top5_idx[i].item()
            confidence = top5_prob[i].item()
            class_name = self.class_labels.get(class_idx, f"class_{class_idx}")

            predictions.append({
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': confidence
            })

        return predictions

    def generate_gradcam(self, input_batch, target_class_idx=None):
        """Generate Grad-CAM heatmap"""
        print("[CNN] Generating Grad-CAM...")

        if target_class_idx is None:
            with torch.no_grad():
                output = self.model(input_batch)
                target_class_idx = output.argmax(dim=1).item()

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        target_layer = self.model.layer4[2].conv3
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        self.model.zero_grad()
        output = self.model(input_batch)

        class_score = output[:, target_class_idx]
        class_score.backward()

        forward_handle.remove()
        backward_handle.remove()

        gradients = gradients[0].cpu()
        activations = activations[0].cpu()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)

        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.detach().numpy()

    def crop_gradcam_to_content(self, gradcam, padding_info):
        """Remove padding from gradcam"""
        gradcam_resized = cv2.resize(gradcam, (224, 224))

        pad_left = padding_info['pad_left']
        pad_top = padding_info['pad_top']
        content_width = padding_info['content_width']
        content_height = padding_info['content_height']

        gradcam_content = gradcam_resized[
                          pad_top:pad_top + content_height,
                          pad_left:pad_left + content_width
                          ]

        return gradcam_content

    def save_gradcam_overlay_only(self, gradcam, original_image, padding_info, filename_prefix):
        """Save overlay Grad-CAM"""
        print("[CNN] Saving Grad-CAM overlay...")

        plt.ioff()

        gradcam_content = self.crop_gradcam_to_content(gradcam, padding_info)

        orig_width = padding_info['original_width']
        orig_height = padding_info['original_height']

        gradcam_final = cv2.resize(gradcam_content, (orig_width, orig_height))

        display_max_size = 600
        scale = min(display_max_size / orig_width, display_max_size / orig_height)

        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)

        img_display = original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        gradcam_display = cv2.resize(gradcam_final, (display_width, display_height))

        fig, ax = plt.subplots(figsize=(10, 6))

        img_array = np.array(img_display)

        heatmap = plt.cm.jet(gradcam_display)[:, :, :3]
        overlay = 0.6 * img_array / 255.0 + 0.4 * heatmap

        ax.imshow(overlay)
        ax.set_axis_off()

        fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)

        gradcam_dir = self.output_dir / "gradcam"
        gradcam_path = gradcam_dir / f"{filename_prefix}_gradcam.png"
        fig.savefig(gradcam_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close(fig)

        print(f"[CNN] Saved Grad-CAM to: {gradcam_path}")
        return gradcam_path

    def analyze_image_with_features(self, image_path, save_visualizations=True):
        """Complete analysis with feature maps"""
        print(f"[CNN] Starting analysis for: {image_path}")

        try:
            timestamp = Path(image_path).stem

            # RESET feature maps and original image
            self.feature_maps = {}
            self.original_image = None

            # Load image
            original_image, padded_image, input_batch, padding_info = self.load_and_preprocess_image(image_path)

            # Setup hooks
            hooks = self.setup_feature_hooks()

            # Make prediction (captures feature maps)
            predictions = self.predict(input_batch)

            print("[CNN] Top 5 predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred['class_name']}: {pred['confidence']:.3f}")

            # Generate Grad-CAM
            gradcam = self.generate_gradcam(input_batch, predictions[0]['class_idx'])

            # Remove hooks
            for hook in hooks:
                hook.remove()

            gradcam_url = None
            overview_url = None
            layer_urls = {}

            if save_visualizations:
                # Save Grad-CAM
                gradcam_path = self.save_gradcam_overlay_only(
                    gradcam, original_image, padding_info, timestamp
                )
                # FIXED: Correct URL prefix for FastAPI static serving
                gradcam_url = f"/outputs/gradcam/{gradcam_path.name}"

                # Save overview
                overview_path = self.create_overview_image(original_image, filename_prefix=timestamp)
                if overview_path:
                    # FIXED: Correct URL prefix for FastAPI static serving
                    overview_url = f"/outputs/feature_maps/{overview_path.name}"

                # Save individual layer visualizations
                for layer_name in ['conv1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
                    if layer_name in self.feature_maps:
                        layer_path = self.visualize_layer_activations(layer_name, filename_prefix=timestamp)
                        if layer_path:
                            # FIXED: Correct URL prefix for FastAPI static serving
                            layer_urls[layer_name] = f"/outputs/feature_maps/{layer_path.name}"

            # Create layer details for frontend
            layer_details = {}
            for layer_name in ['conv1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
                layer_details[layer_name] = {
                    'size': self.layer_info[layer_name]['size'],
                    'channels': self.layer_info[layer_name]['channels'],
                    'available': layer_name in self.feature_maps,
                    'url': layer_urls.get(layer_name)
                }

            results = {
                'status': 'success',
                'predictions': predictions,
                'model_info': {
                    'architecture': 'ResNet50',
                    'dataset': 'ImageNet',
                    'input_size': '224x224 (with aspect ratio preservation)',
                    'device': str(self.device),
                    'original_size': f"{padding_info['original_width']}x{padding_info['original_height']}",
                    'padding_applied': f"left:{padding_info['pad_left']}, top:{padding_info['pad_top']}"
                },
                'gradcam_url': gradcam_url,
                'feature_maps': {
                    'overview_url': overview_url,
                    'individual_layers': layer_details
                },
                'analysis_complete': True
            }

            print(f"[CNN] Analysis complete. Overview: {overview_url}")
            print(f"[CNN] Layer visualizations: {len(layer_urls)}")
            return results

        except Exception as e:
            print(f"[CNN] Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'analysis_complete': False
            }


# Global analyzer instance
_analyzer = None


def get_analyzer(output_dir="static/outputs"):
    """Get or create global analyzer instance with configurable output directory"""
    global _analyzer
    if _analyzer is None:
        _analyzer = CNNFeatureAnalyzer(output_dir=output_dir)
    return _analyzer


def analyze_image_endpoint(image_path, output_dir="static/outputs"):
    """
    Endpoint function for FastAPI integration
    FIXED: Now accepts output_dir parameter to specify where files should be saved
    """
    print(f"[CNN] analyze_image_endpoint called with output_dir: {output_dir}")

    # Create analyzer with specified output directory
    analyzer = CNNFeatureAnalyzer(output_dir=output_dir)
    return analyzer.analyze_image_with_features(image_path)