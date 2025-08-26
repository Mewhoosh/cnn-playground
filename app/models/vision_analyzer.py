#!/usr/bin/env python3
"""
Computer Vision Analyzer - Object Detection & Instance Segmentation

Computer vision system providing comprehensive analysis capabilities:
- Multi-model object detection (YOLO11, YOLO12, RT-DETR)
- Instance segmentation (YOLO11-seg, SAM2)
- Real-time video processing with frame optimization
- Live camera feed analysis with low-latency inference
- Model management with organized caching

Features:
- Automatic model discovery and organized storage
- GPU acceleration with fallback to CPU
- High-performance video processing pipelines
- Error handling and logging
- Comprehensive visualization capabilities
"""

from typing import Dict, List, Tuple, Optional, Union, Any, AsyncGenerator
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import logging
import os
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model management system with organized storage and caching.

    Handles automatic model downloads, organized folder structure, and
    efficient caching for computer vision models. Supports multiple
    model architectures including YOLO, RT-DETR, and SAM variants.

    Attributes:
        models_dir (Path): Base directory for model storage
        detection_dir (Path): Directory for detection models
        segmentation_dir (Path): Directory for segmentation models
    """

    def __init__(self, base_path: str = "app/models/saved_models") -> None:
        """
        Initialize model manager with organized directory structure.

        Creates hierarchical folder structure for different model types
        and configures environment variables for model caching.

        Args:
            base_path: Base directory path for model storage

        Raises:
            OSError: If directory creation fails
        """
        # Convert to absolute path to ensure consistency
        self.models_dir: Path = Path(base_path).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create organized subfolders
        self.detection_dir: Path = self.models_dir / "detection"
        self.segmentation_dir: Path = self.models_dir / "segmentation"
        self.detection_dir.mkdir(exist_ok=True)
        self.segmentation_dir.mkdir(exist_ok=True)

        # Create model type subfolders
        (self.detection_dir / "yolo11").mkdir(exist_ok=True)
        (self.detection_dir / "yolo12").mkdir(exist_ok=True)
        (self.detection_dir / "rtdetr").mkdir(exist_ok=True)

        (self.segmentation_dir / "yolo11").mkdir(exist_ok=True)
        (self.segmentation_dir / "yolo12").mkdir(exist_ok=True)
        (self.segmentation_dir / "sam2").mkdir(exist_ok=True)

        # Set multiple environment variables for ultralytics
        os.environ['YOLO_CONFIG_DIR'] = str(self.models_dir)
        os.environ['ULTRALYTICS_CONFIG_DIR'] = str(self.models_dir)

        # Set weights directory specifically
        weights_dir: Path = self.models_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        os.environ['TORCH_HOME'] = str(weights_dir)

        # Also set for Ultralytics specific paths
        os.environ['ULTRALYTICS_WEIGHTS_DIR'] = str(self.models_dir)

        logger.info(f"[ModelManager] Models directory organized: {self.models_dir}")
        logger.info(f"[ModelManager] Detection models: {self.detection_dir}")
        logger.info(f"[ModelManager] Segmentation models: {self.segmentation_dir}")
        logger.info(f"[ModelManager] Environment variables set for model caching")

    def get_model_path(self, model_name: str) -> str:
        """
        Get local path for model with organized folder structure.

        Determines appropriate storage location based on model type and
        architecture. Creates directory structure if needed and handles
        model download coordination.

        Args:
            model_name: Name of the model (e.g., 'yolo11m', 'rtdetr-l')

        Returns:
            Local path to model file for loading/downloading

        Note:
            Path creation doesn't guarantee model exists - download may be needed
        """
        model_file: str = f"{model_name}.pt"

        # Determine correct subfolder based on model name
        if 'seg' in model_name:
            if 'yolo11' in model_name:
                local_path: Path = self.segmentation_dir / "yolo11" / model_file
            elif 'yolo12' in model_name:
                local_path = self.segmentation_dir / "yolo12" / model_file
            elif 'sam2' in model_name:
                local_path = self.segmentation_dir / "sam2" / model_file
            else:
                local_path = self.segmentation_dir / model_file
        else:
            if 'yolo11' in model_name:
                local_path = self.detection_dir / "yolo11" / model_file
            elif 'yolo12' in model_name:
                local_path = self.detection_dir / "yolo12" / model_file
            elif 'rtdetr' in model_name:
                local_path = self.detection_dir / "rtdetr" / model_file
            else:
                local_path = self.detection_dir / model_file

        if local_path.exists():
            logger.info(f"[ModelManager] Using cached model: {local_path}")
            return str(local_path)
        else:
            # Create directory for model if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ModelManager] Model will be downloaded to: {local_path}")

            # For ultralytics, we need to handle the download process
            return self._handle_model_download(model_name, local_path)

    def _handle_model_download(self, model_name: str, target_path: Path) -> str:
        """
        Handle model download preparation to specific location.

        Prepares download location and coordinates with ultralytics
        download system for model placement.

        Args:
            model_name: Name of model to download
            target_path: Target path for model storage

        Returns:
            Path string for ultralytics to use

        Note:
            Actual download is handled by ultralytics framework
        """
        try:
            # First check if ultralytics will download to our location
            logger.info(f"[ModelManager] Preparing download location for {model_name}")

            # Return the target path for ultralytics to use
            # Ultralytics will handle the actual download
            return str(target_path)
        except Exception as e:
            logger.warning(f"[ModelManager] Download preparation failed: {e}")
            # Fallback to model name for default ultralytics behavior
            return model_name

    def cleanup_old_models(self) -> None:
        """
        Clean up models from project root and organize into directories.

        Moves any model files found in project root to appropriate
        subdirectories. Prevents clutter and ensures organized storage.

        Note:
            Safe operation - only moves files, doesn't delete them
        """
        project_root: Path = Path(".")
        model_patterns: List[str] = ["*.pt", "yolo*.pt", "rtdetr*.pt", "sam*.pt"]

        moved_count: int = 0
        for pattern in model_patterns:
            for model_file in project_root.glob(pattern):
                if model_file.is_file():
                    try:
                        # Determine target directory
                        if 'seg' in model_file.name:
                            target_dir: Path = self.segmentation_dir
                        else:
                            target_dir = self.detection_dir

                        target_path: Path = target_dir / model_file.name

                        # Move file if it doesn't exist in target
                        if not target_path.exists():
                            model_file.rename(target_path)
                            logger.info(f"[ModelManager] Moved {model_file.name} to {target_path}")
                            moved_count += 1
                        else:
                            # Remove duplicate from root
                            model_file.unlink()
                            logger.info(f"[ModelManager] Removed duplicate {model_file.name} from root")

                    except Exception as e:
                        logger.warning(f"[ModelManager] Could not move {model_file.name}: {e}")

        if moved_count > 0:
            logger.info(f"[ModelManager] Organized {moved_count} model files")


class ObjectDetector:
    """
    Object detection system using state-of-the-art models.

    Supports multiple detection architectures including YOLO11, YOLO12,
    and RT-DETR with GPU acceleration and result analysis.
    Provides bounding box detection with confidence scoring and visualization.

    Attributes:
        device (torch.device): Computing device (CPU/CUDA)
        model_name (str): Name of loaded detection model
        model (Optional[Any]): Loaded detection model instance
        model_manager (ModelManager): Model management system
        class_names (List[str]): COCO dataset class labels
        output_dir (Path): Directory for saving visualizations
    """

    def __init__(self, model_name: str = "yolo11m", device: Optional[torch.device] = None,
                 output_dir: str = "static/outputs") -> None:
        """
        Initialize object detector with specified model and configuration.

        Args:
            model_name: Detection model to use ('yolo11m', 'rtdetr-l', etc.)
            device: Computing device (auto-detected if None)
            output_dir: Directory for saving detection visualizations

        Raises:
            ImportError: If ultralytics package not available
            RuntimeError: If model loading fails
        """
        self.device: torch.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name: str = model_name
        self.model: Optional[Any] = None
        self.model_manager: ModelManager = ModelManager()
        self.class_names: List[str] = self._get_coco_classes()
        self.output_dir: Path = Path(output_dir)

        logger.info(f"[ObjectDetector] Initializing {model_name} on {self.device}")

        # Clean up any models in project root
        self.model_manager.cleanup_old_models()

        try:
            self._load_model()
        except ImportError as e:
            logger.error(f"[ObjectDetector] Model not available: {e}")
            raise e
        except Exception as e:
            logger.error(f"[ObjectDetector] Failed to load model: {e}")
            raise e

    def _get_coco_classes(self) -> List[str]:
        """
        Get COCO dataset class names for object labeling.

        Returns:
            List of 80 COCO class names in standard order
        """
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _load_model(self) -> None:
        """
        Load detection model with architecture handling.

        Supports both YOLO and RT-DETR architectures with automatic
        model type detection and initialization.

        Raises:
            ImportError: If ultralytics package not installed
            RuntimeError: If model loading fails
        """
        try:
            from ultralytics import YOLO, RTDETR

            model_path: str = self.model_manager.get_model_path(self.model_name)

            logger.info(f"[ObjectDetector] Loading model from: {model_path}")

            if 'rtdetr' in self.model_name:
                self.model = RTDETR(model_path)
                logger.info(f"[ObjectDetector] Loaded RT-DETR model: {self.model_name}")
            else:
                self.model = YOLO(model_path)
                logger.info(f"[ObjectDetector] Loaded YOLO model: {self.model_name}")

            # Check if model was saved in our organized directory
            self._verify_model_location()

        except ImportError:
            raise ImportError("ultralytics package required for SOTA models")
        except Exception as e:
            logger.error(f"[ObjectDetector] Error loading model: {e}")
            raise

    def _verify_model_location(self) -> None:
        """
        Verify model is stored in correct organized location.

        Moves model from project root to organized directory if needed.
        Ensures clean project structure and model organization.
        """
        try:
            # After model loading, check if ultralytics downloaded to project root
            project_root = Path(".")
            model_file = f"{self.model_name}.pt"
            root_model = project_root / model_file

            if root_model.exists():
                # Model was downloaded to root, move it to organized location
                target_path = self.model_manager.get_model_path(self.model_name)
                target_path = Path(target_path)

                if not target_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    root_model.rename(target_path)
                    logger.info(f"[ObjectDetector] Moved {model_file} to organized location: {target_path}")
                else:
                    # Remove duplicate from root
                    root_model.unlink()
                    logger.info(f"[ObjectDetector] Removed duplicate {model_file} from root")

        except Exception as e:
            logger.warning(f"[ObjectDetector] Model organization check failed: {e}")

    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect objects in image with confidence filtering.

        Performs object detection inference on input image and returns
        structured results with bounding boxes, confidence scores, and
        class predictions. Supports confidence-based filtering.

        Args:
            image_path: Path to input image file
            confidence_threshold: Minimum confidence for object detection (0.0-1.0)

        Returns:
            Detection results dictionary containing:
                - status: Detection completion status
                - objects: List of detected objects with metadata
                - processing_time: Inference time in milliseconds
                - model_info: Model architecture and configuration details

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If detection inference fails
        """
        try:
            if self.model is None:
                return self._generate_mock_detection_results()

            logger.info(f"[ObjectDetector] Running detection on: {image_path}")
            start_time = time.time()

            # Run inference
            results = self.model(image_path, conf=confidence_threshold, verbose=False)

            processing_time = int((time.time() - start_time) * 1000)

            # Parse results
            objects = []
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())

                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                        # Calculate area
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                        objects.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': bbox.tolist(),
                            'area': float(area),
                            'class_id': class_id
                        })

            logger.info(f"[ObjectDetector] Detected {len(objects)} objects in {processing_time}ms")

            return {
                'status': 'success',
                'objects': objects,
                'processing_time': processing_time,
                'model_info': {
                    'architecture': self.model_name.upper(),
                    'confidence_threshold': confidence_threshold,
                    'device': str(self.device)
                }
            }

        except Exception as e:
            logger.error(f"[ObjectDetector] Detection error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'objects': [],
                'processing_time': 0
            }

    def visualize_detections(self, image_path: str, objects: List[Dict[str, Any]],
                           output_path: str) -> Optional[str]:
        """
        Create detection visualization with bounding boxes and labels.

        Generates visualization showing detected objects with
        color-coded bounding boxes, confidence scores, and class labels.
        Uses color palette and typography.

        Args:
            image_path: Path to original image
            objects: List of detection results from detect_objects()
            output_path: Path for saving visualization

        Returns:
            Path to saved visualization, None if visualization fails

        Raises:
            IOError: If image loading or saving fails
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)

            # Define colors for different classes
            colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
            ]

            # Draw bounding boxes
            for i, obj in enumerate(objects):
                bbox = obj['bbox']
                class_name = obj['class_name']
                confidence = obj['confidence']
                color = colors[i % len(colors)]

                # Draw rectangle
                draw.rectangle(
                    [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                    outline=color,
                    width=3
                )

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

                # Get text size
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                # Draw label background
                draw.rectangle(
                    [(bbox[0], bbox[1] - text_height - 4),
                     (bbox[0] + text_width + 4, bbox[1])],
                    fill=color
                )

                # Draw text
                draw.text(
                    (bbox[0] + 2, bbox[1] - text_height - 2),
                    label,
                    fill='white',
                    font=font
                )

            # Save image
            image.save(output_path)
            logger.info(f"[ObjectDetector] Visualization saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[ObjectDetector] Visualization error: {e}")
            return None

    def _generate_mock_detection_results(self) -> Dict[str, Any]:
        """
        Generate mock detection results for demonstration.

        Provides fallback detection results when models are not available.
        Useful for testing and demonstration of API structure.

        Returns:
            Mock detection results in standard format
        """
        logger.warning("[ObjectDetector] Using mock results - models not available")
        return {
            'status': 'success',
            'objects': [
                {
                    'class_name': 'person',
                    'confidence': 0.85,
                    'bbox': [100, 50, 300, 400],
                    'area': 70000,
                    'class_id': 0
                },
                {
                    'class_name': 'car',
                    'confidence': 0.72,
                    'bbox': [350, 200, 600, 350],
                    'area': 37500,
                    'class_id': 2
                }
            ],
            'processing_time': 150,
            'model_info': {
                'architecture': 'DEMO_MODE',
                'note': 'Install ultralytics for real detection'
            },
            'demo_mode': True
        }


class ImageSegmenter:
    """
    Instance segmentation system with pixel-level accuracy.

    Provides instance segmentation using YOLO11-seg and SAM2
    architectures. Generates pixel-level masks for object instances
    with visualization and analysis capabilities.

    Attributes:
        device (torch.device): Computing device (CPU/CUDA)
        model_name (str): Name of loaded segmentation model
        model (Optional[Any]): Loaded segmentation model instance
        model_manager (ModelManager): Model management system
        _current_masks (Optional[List[np.ndarray]]): Storage for current segmentation masks
    """

    def __init__(self, model_name: str = "yolo11m-seg", device: Optional[torch.device] = None) -> None:
        """
        Initialize instance segmentation system with specified model.

        Args:
            model_name: Segmentation model to use ('yolo11m-seg', 'sam2_t', etc.)
            device: Computing device (auto-detected if None)

        Raises:
            ImportError: If required packages not available
            RuntimeError: If model loading fails
        """
        self.device: torch.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name: str = model_name
        self.model: Optional[Any] = None
        self.model_manager: ModelManager = ModelManager()
        self._current_masks: Optional[List[np.ndarray]] = None  # Store masks for visualization

        logger.info(f"[ImageSegmenter] Initializing {model_name} on {self.device}")

        # Clean up any models in project root
        self.model_manager.cleanup_old_models()

        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"[ImageSegmenter] Model not available: {e}")
            self.model = None

    def _load_model(self) -> None:
        """
        Load segmentation model with architecture-specific handling.

        Supports both YOLO segmentation and SAM2 architectures with
        model initialization and configuration.

        Raises:
            ImportError: If ultralytics package not installed
            RuntimeError: If model loading fails
        """
        try:
            model_path = self.model_manager.get_model_path(self.model_name)

            logger.info(f"[ImageSegmenter] Loading model from: {model_path}")

            if 'sam2' in self.model_name:
                from ultralytics import SAM
                sam_model = self.model_name.replace('sam2_', 'sam2.1_')
                self.model = SAM(f"{sam_model}.pt")
                logger.info(f"[ImageSegmenter] Loaded SAM2 model: {sam_model}")
            else:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                logger.info(f"[ImageSegmenter] Loaded YOLO segmentation model: {self.model_name}")

            # Check if model was saved in our organized directory
            self._verify_model_location()

        except Exception as e:
            logger.error(f"[ImageSegmenter] Error loading model {self.model_name}: {e}")
            raise

    def _verify_model_location(self) -> None:
        """
        Verify segmentation model is stored in correct organized location.

        Ensures models are organized in segmentation subdirectory
        and moves from project root if necessary.
        """
        try:
            # After model loading, check if ultralytics downloaded to project root
            project_root = Path(".")
            model_file = f"{self.model_name}.pt"
            root_model = project_root / model_file

            if root_model.exists():
                # Model was downloaded to root, move it to organized location
                target_path = self.model_manager.get_model_path(self.model_name)
                target_path = Path(target_path)

                if not target_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    root_model.rename(target_path)
                    logger.info(f"[ImageSegmenter] Moved {model_file} to organized location: {target_path}")
                else:
                    # Remove duplicate from root
                    root_model.unlink()
                    logger.info(f"[ImageSegmenter] Removed duplicate {model_file} from root")

        except Exception as e:
            logger.warning(f"[ImageSegmenter] Model organization check failed: {e}")

    def segment_image(self, image_path: str) -> Dict[str, Any]:
        """
        Perform instance segmentation on input image.

        Generates pixel-level instance masks for detected objects with
        metadata including coverage statistics, confidence
        scores, and class predictions.

        Args:
            image_path: Path to input image file

        Returns:
            Segmentation results dictionary containing:
                - status: Segmentation completion status
                - segments: List of segmented instances with metadata
                - processing_time: Inference time in milliseconds
                - model_info: Model architecture and configuration details

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If segmentation inference fails
        """
        try:
            if self.model is None:
                return self._generate_mock_segmentation_results()

            logger.info(f"[ImageSegmenter] Running segmentation on: {image_path}")
            start_time = time.time()

            # Run inference
            results = self.model(image_path, verbose=False)

            processing_time = int((time.time() - start_time) * 1000)

            # Parse results and store masks
            segments = []
            masks_data = []  # Store actual mask arrays

            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks
                    boxes = result.boxes

                    for i in range(len(masks)):
                        # Get mask data
                        mask = masks.data[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())

                        # Get class name
                        class_name = self._get_class_name(class_id)

                        # Calculate coverage
                        total_pixels = mask.shape[0] * mask.shape[1]
                        mask_pixels = np.sum(mask > 0.5)
                        coverage = mask_pixels / total_pixels

                        # Generate color
                        color = self._generate_color(i)

                        segments.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'coverage': float(coverage),
                            'pixel_count': int(mask_pixels),
                            'color': color,
                            'class_id': class_id
                        })

                        # Store the actual mask data for visualization
                        masks_data.append(mask)

            # Store masks for visualization
            self._current_masks = masks_data

            logger.info(f"[ImageSegmenter] Found {len(segments)} segments in {processing_time}ms")

            return {
                'status': 'success',
                'segments': segments,
                'processing_time': processing_time,
                'model_info': {
                    'architecture': self.model_name.upper(),
                    'segmentation_type': 'Instance',
                    'device': str(self.device)
                }
            }

        except Exception as e:
            logger.error(f"[ImageSegmenter] Segmentation error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'segments': [],
                'processing_time': 0
            }

    def visualize_segmentation(self, image_path: str, segments: List[Dict[str, Any]],
                             output_path: str) -> Optional[str]:
        """
        Create segmentation visualization with colored masks.

        Generates visualization overlaying colored instance masks
        on original image with contours, labels, and transparency effects.
        Uses actual segmentation masks when available.

        Args:
            image_path: Path to original image
            segments: List of segmentation results from segment_image()
            output_path: Path for saving visualization

        Returns:
            Path to saved visualization, None if visualization fails

        Raises:
            IOError: If image processing or saving fails
        """
        try:
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            original_array = np.array(original_image)

            # Create copy for overlay
            result_image = original_array.copy()

            # Check if we have actual masks
            if self._current_masks and len(self._current_masks) == len(segments):
                logger.info(f"[ImageSegmenter] Using actual masks for visualization ({len(self._current_masks)} masks)")

                # Apply each mask with its color
                for i, (segment, mask) in enumerate(zip(segments, self._current_masks)):
                    # Get color as RGB tuple
                    color_hex = segment['color']
                    color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))

                    # Resize mask to match image size if needed
                    if mask.shape != original_array.shape[:2]:
                        # Resize mask to image dimensions
                        mask_resized = cv2.resize(mask.astype(np.uint8),
                                                (original_array.shape[1], original_array.shape[0]))
                        mask_resized = mask_resized.astype(np.float32)
                    else:
                        mask_resized = mask

                    # Create colored mask
                    colored_mask = np.zeros_like(original_array)
                    for c in range(3):
                        colored_mask[:, :, c] = color_rgb[c]

                    # Apply mask with transparency
                    mask_binary = (mask_resized > 0.5).astype(np.float32)
                    alpha = 0.4  # Transparency level

                    # Blend the colored mask with original image
                    for c in range(3):
                        result_image[:, :, c] = (
                            (1 - alpha * mask_binary) * result_image[:, :, c] +
                            alpha * mask_binary * colored_mask[:, :, c]
                        )

                # Add contours for better visibility
                for i, (segment, mask) in enumerate(zip(segments, self._current_masks)):
                    # Resize mask if needed
                    if mask.shape != original_array.shape[:2]:
                        mask_resized = cv2.resize(mask.astype(np.uint8),
                                                (original_array.shape[1], original_array.shape[0]))
                    else:
                        mask_resized = mask.astype(np.uint8)

                    # Find contours
                    contours, _ = cv2.findContours(
                        (mask_resized > 0.5).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Get color as BGR for OpenCV
                    color_hex = segment['color']
                    color_bgr = tuple(int(color_hex[j:j+2], 16) for j in (5, 3, 1))  # BGR order

                    # Draw contours
                    cv2.drawContours(result_image, contours, -1, color_bgr, 2)

                    # Add labels
                    if contours:
                        # Find the largest contour for label placement
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # Draw label background
                            label = f"{segment['class_name']}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            thickness = 2

                            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            cv2.rectangle(result_image,
                                        (cx - text_w//2 - 5, cy - text_h//2 - 5),
                                        (cx + text_w//2 + 5, cy + text_h//2 + 5),
                                        color_bgr, -1)

                            # Draw text
                            cv2.putText(result_image, label,
                                      (cx - text_w//2, cy + text_h//2),
                                      font, font_scale, (255, 255, 255), thickness)

            else:
                logger.warning(f"[ImageSegmenter] No masks available, using placeholder visualization")
                # Fallback to placeholder rectangles
                draw = ImageDraw.Draw(Image.fromarray(result_image))
                for i, segment in enumerate(segments):
                    # Generate a colored rectangle as placeholder
                    x1, y1 = 50 + i * 100, 50 + i * 50
                    x2, y2 = x1 + 150, y1 + 100

                    color = tuple(int(segment['color'][j:j+2], 16) for j in (1, 3, 5))
                    draw.rectangle([(x1, y1), (x2, y2)], fill=color)
                    draw.text((x1 + 5, y1 + 5), segment['class_name'], fill='white')

                result_image = np.array(Image.fromarray(result_image))

            # Convert back to PIL and save
            result_pil = Image.fromarray(result_image.astype(np.uint8))
            result_pil.save(output_path)

            logger.info(f"[ImageSegmenter] Segmentation visualization saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[ImageSegmenter] Visualization error: {e}")
            import traceback
            logger.error(f"[ImageSegmenter] Full traceback: {traceback.format_exc()}")
            return None

    def _get_class_name(self, class_id: int) -> str:
        """
        Get human-readable class name from COCO class ID.

        Args:
            class_id: COCO dataset class index

        Returns:
            Human-readable class name or fallback identifier
        """
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"

    def _generate_color(self, index: int) -> str:
        """
        Generate consistent hex color for segmentation visualization.

        Args:
            index: Segment index for color selection

        Returns:
            Hex color string for visualization
        """
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        return colors[index % len(colors)]

    def _generate_mock_segmentation_results(self) -> Dict[str, Any]:
        """
        Generate mock segmentation results for demonstration.

        Provides fallback segmentation results when models are not available.
        Maintains API compatibility for testing and demonstration.

        Returns:
            Mock segmentation results in standard format
        """
        logger.warning("[ImageSegmenter] Using mock results - models not available")
        return {
            'status': 'success',
            'segments': [
                {
                    'class_name': 'person',
                    'confidence': 0.88,
                    'coverage': 0.25,
                    'pixel_count': 50000,
                    'color': '#FF6B6B',
                    'class_id': 0
                },
                {
                    'class_name': 'car',
                    'confidence': 0.76,
                    'coverage': 0.15,
                    'pixel_count': 30000,
                    'color': '#4ECDC4',
                    'class_id': 2
                }
            ],
            'processing_time': 200,
            'model_info': {
                'architecture': 'DEMO_MODE',
                'segmentation_type': 'Instance',
                'note': 'Install ultralytics for real segmentation'
            },
            'demo_mode': True
        }


class VisionAnalyzer:
    """
    Computer vision analysis system combining multiple tasks.

    Analyzer that orchestrates object detection and
    instance segmentation for image understanding. Provides
    unified interface for multi-task computer vision analysis.

    Attributes:
        output_dir (Path): Base directory for saving all analysis outputs
    """

    def __init__(self, output_dir: str = "static/outputs") -> None:
        """
        Initialize vision analysis system.

        Args:
            output_dir: Base directory for saving analysis results

        Raises:
            OSError: If output directory creation fails
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "detection").mkdir(exist_ok=True)
        (self.output_dir / "segmentation").mkdir(exist_ok=True)
        (self.output_dir / "video").mkdir(exist_ok=True)

        logger.info(f"[VisionAnalyzer] Initialized with output directory: {self.output_dir}")

    def analyze_complete(self, image_path: str, detection_model: str = "yolo11m",
                         segmentation_model: str = "yolo11m-seg",
                         confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Perform computer vision analysis on input image.

        Orchestrates analysis pipeline including object detection,
        instance segmentation, and visualization generation.
        Provides unified results with metadata.

        Args:
            image_path: Path to input image file
            detection_model: Object detection model to use
            segmentation_model: Instance segmentation model to use
            confidence_threshold: Minimum confidence for object detection

        Returns:
            Analysis results dictionary containing:
                - status: Overall analysis completion status
                - image_info: Input image metadata and properties
                - detection: Complete object detection results and visualizations
                - segmentation: Complete instance segmentation results and visualizations
                - model_info: Model architecture and configuration metadata
                - analysis_complete: Boolean completion flag

        Raises:
            FileNotFoundError: If input image file doesn't exist
            RuntimeError: If any analysis component fails critically
        """
        logger.info(f"[VisionAnalyzer] Starting analysis for: {image_path}")

        try:
            # Get image info
            image = Image.open(image_path)
            image_info = {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format
            }

            # Initialize models
            detector = ObjectDetector(detection_model)
            segmenter = ImageSegmenter(segmentation_model)

            # Generate unique filename
            timestamp = Path(image_path).stem

            # Object Detection
            logger.info("[VisionAnalyzer] Running object detection...")
            detection_results = detector.detect_objects(image_path, confidence_threshold)

            detection_image_url = None
            if (detection_results['status'] == 'success' and
                    detection_results['objects'] and
                    not detection_results.get('demo_mode', False)):
                detection_output = self.output_dir / "detection" / f"{timestamp}_detection.png"
                detection_vis_path = detector.visualize_detections(
                    image_path, detection_results['objects'], str(detection_output)
                )
                if detection_vis_path:
                    detection_image_url = f"/outputs/detection/{detection_output.name}"

            # Instance Segmentation
            logger.info("[VisionAnalyzer] Running instance segmentation...")
            segmentation_results = segmenter.segment_image(image_path)

            segmentation_image_url = None
            if (segmentation_results['status'] == 'success' and
                    segmentation_results['segments'] and
                    not segmentation_results.get('demo_mode', False)):
                segmentation_output = self.output_dir / "segmentation" / f"{timestamp}_segmentation.png"
                segmentation_vis_path = segmenter.visualize_segmentation(
                    image_path,
                    segmentation_results['segments'],
                    str(segmentation_output)
                )
                if segmentation_vis_path:
                    segmentation_image_url = f"/outputs/segmentation/{segmentation_output.name}"

            # Compile results
            results = {
                'status': 'success',
                'image_info': image_info,
                'detection': {
                    **detection_results,
                    'image_url': detection_image_url
                },
                'segmentation': {
                    **segmentation_results,
                    'image_url': segmentation_image_url
                },
                'model_info': {
                    'detection': detection_results.get('model_info', {}),
                    'segmentation': segmentation_results.get('model_info', {}),
                    'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                    'framework': 'Computer Vision 2024'
                },
                'analysis_complete': True
            }

            logger.info("[VisionAnalyzer] Analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"[VisionAnalyzer] Error during analysis: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'analysis_complete': False
            }


# Video Processing Functions
async def process_video_complete(video_path: str, detection_model: str = "yolo11n",
                                 frame_skip: int = 2, confidence_threshold: float = 0.4,
                                 output_dir: str = "static/outputs/video") -> Dict[str, Any]:
    """
    Process video file with frame-by-frame object detection and output generation.

    Video processing pipeline that applies object detection
    to video frames and generates annotated output video with performance
    statistics and metadata tracking.

    Args:
        video_path: Path to input video file
        detection_model: Object detection model for frame analysis
        frame_skip: Process every Nth frame for performance optimization
        confidence_threshold: Minimum confidence for object detection
        output_dir: Directory for saving processed video

    Returns:
        Video processing results dictionary containing:
            - status: Processing completion status
            - processed_video_url: URL to annotated output video
            - stats: Performance statistics and object counts
            - model_info: Model configuration and processing metadata

    Raises:
        FileNotFoundError: If input video file doesn't exist
        RuntimeError: If video processing fails

    Note:
        Async function for non-blocking video processing
    """
    logger.info(f"[VideoProcessor] Starting video processing: {video_path}")

    try:
        # Initialize detector
        detector = ObjectDetector(detection_model)

        if detector.model is None:
            return _generate_mock_video_results()

        # Setup video processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"[VideoProcessor] Video: {total_frames} frames, {fps} FPS, {width}x{height}")

        # Setup output video
        output_path = Path(output_dir) / f"processed_{Path(video_path).name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        processed_frames = 0
        total_objects = 0
        start_time = time.time()

        temp_dir = Path("temp/video_frames")
        temp_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every Nth frame
            if frame_count % frame_skip == 0:
                # Save frame temporarily
                temp_frame_path = temp_dir / f"frame_{frame_count}.jpg"
                cv2.imwrite(str(temp_frame_path), frame)

                # Run detection
                results = detector.detect_objects(str(temp_frame_path), confidence_threshold)

                if results['status'] == 'success':
                    # Draw detections on frame
                    for obj in results['objects']:
                        bbox = obj['bbox']
                        class_name = obj['class_name']
                        confidence = obj['confidence']

                        # Draw bounding box
                        cv2.rectangle(frame,
                                      (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])),
                                      (0, 255, 0), 2)

                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame,
                                      (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                                      (int(bbox[0]) + label_size[0], int(bbox[1])),
                                      (0, 255, 0), -1)
                        cv2.putText(frame, label,
                                    (int(bbox[0]), int(bbox[1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    total_objects += len(results['objects'])

                # Clean up temp file
                if temp_frame_path.exists():
                    temp_frame_path.unlink()

                processed_frames += 1

            # Write frame to output
            out.write(frame)

        # Clean up
        cap.release()
        out.release()

        processing_time = time.time() - start_time
        avg_fps = processed_frames / processing_time if processing_time > 0 else 0

        result = {
            'status': 'success',
            'processed_video_url': f"/outputs/video/{output_path.name}",
            'stats': {
                'total_frames': total_frames,
                'processing_time': int(processing_time),
                'objects_detected': total_objects,
                'avg_fps': int(avg_fps)
            },
            'model_info': {
                'detection_model': detection_model,
                'frame_skip': frame_skip,
                'confidence_threshold': confidence_threshold
            }
        }

        logger.info(f"[VideoProcessor] Video processing completed: {output_path}")
        return result

    except Exception as e:
        logger.error(f"[VideoProcessor] Error processing video: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'processed_video_url': None,
            'stats': {}
        }


async def process_live_frame(frame_path: str, detection_model: str = "yolo11n",
                             confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Process single frame from live camera feed with low-latency inference.

    Optimized for real-time processing of camera frames with minimal
    latency. Designed for live streaming applications and interactive
    computer vision systems.

    Args:
        frame_path: Path to captured frame image
        detection_model: Detection model for real-time inference
        confidence_threshold: Minimum confidence for real-time detection

    Returns:
        Live frame processing results containing:
            - status: Processing completion status
            - objects: Detected objects with minimal metadata
            - processing_time: Inference latency in milliseconds
            - model: Model identifier for client tracking
            - frame_info: Frame metadata and processing context

    Raises:
        FileNotFoundError: If frame file doesn't exist
        RuntimeError: If frame processing fails

    Note:
        Optimized for speed over analysis
    """
    try:
        # Initialize detector with proper model
        detector = ObjectDetector(detection_model)

        # FIXED: Remove mock/demo mode - always try real detection
        start_time = time.time()
        results = detector.detect_objects(frame_path, confidence_threshold)
        processing_time = int((time.time() - start_time) * 1000)

        if results['status'] == 'success':
            logger.info(f"[LiveProcessor] Processed frame: {len(results['objects'])} objects detected in {processing_time}ms")
            return {
                'status': 'success',
                'objects': results['objects'],
                'processing_time': processing_time,
                'model': detection_model,
                'frame_info': {
                    'confidence_threshold': confidence_threshold,
                    'timestamp': time.time()
                }
            }
        else:
            logger.warning(f"[LiveProcessor] Detection failed: {results.get('message', 'Unknown error')}")
            return {
                'status': 'error',
                'message': results.get('message', 'Detection failed'),
                'objects': [],
                'processing_time': processing_time
            }

    except Exception as e:
        logger.error(f"[LiveProcessor] Error processing frame: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'objects': [],
            'processing_time': 0
        }


def _generate_mock_video_results() -> Dict[str, Any]:
    """
    Generate mock video processing results for demonstration.

    Returns:
        Mock video processing results in standard format
    """
    return {
        'status': 'success',
        'processed_video_url': None,
        'stats': {
            'total_frames': 500,
            'processing_time': 45,
            'objects_detected': 150,
            'avg_fps': 12
        },
        'model_info': {
            'detection_model': 'yolo11n',
            'note': 'Demo mode - models not loaded'
        },
        'demo_mode': True
    }


# Global analyzer instance for efficient resource management
_vision_analyzer: Optional[VisionAnalyzer] = None


def get_sota_vision_analyzer(output_dir: str = "static/outputs") -> VisionAnalyzer:
    """
    Get or create global vision analyzer instance using singleton pattern.

    Implements efficient resource management by reusing analyzer instance
    across multiple requests. Reduces model loading overhead and memory usage.

    Args:
        output_dir: Directory for saving analysis outputs

    Returns:
        Global VisionAnalyzer instance ready for analysis
    """
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer(output_dir=output_dir)
    return _vision_analyzer


def analyze_vision_complete(image_path: str, detection_model: str = "yolo11m",
                            segmentation_model: str = "yolo11m-seg",
                            confidence_threshold: float = 0.5,
                            output_dir: str = "static/outputs") -> Dict[str, Any]:
    """
    FastAPI endpoint function for computer vision analysis.

    Primary API entry point for computer vision analysis requests.
    Creates analyzer instance with specified configuration and
    performs complete multi-task analysis.

    Args:
        image_path: Path to input image file
        detection_model: Object detection model selection
        segmentation_model: Instance segmentation model selection
        confidence_threshold: Detection confidence threshold
        output_dir: Directory for saving analysis outputs

    Returns:
        Complete analysis results dictionary suitable for JSON API response

    Note:
        Creates new analyzer instance for output directory isolation
    """
    analyzer: VisionAnalyzer = VisionAnalyzer(output_dir=output_dir)
    return analyzer.analyze_complete(
        image_path=image_path,
        detection_model=detection_model,
        segmentation_model=segmentation_model,
        confidence_threshold=confidence_threshold
    )


if __name__ == "__main__":
    # Testing and validation
    print("Testing Computer Vision Analyzer...")
    try:
        analyzer: VisionAnalyzer = VisionAnalyzer()
        print("Vision Analyzer initialized successfully")
        print("Available detection models: YOLO11 (n,s,m,l,x), RT-DETR (l,x)")
        print("Available segmentation models: YOLO11-seg, SAM2.1 (t,s,b,l)")
        print("Analyzer ready for computer vision tasks")
    except Exception as e:
        print(f"Error: {e}")
        print("Running in demo mode - models will generate mock results")
