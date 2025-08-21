#!/usr/bin/env python3
"""
Computer Vision Analyzer - Object Detection & Instance Segmentation
Professional implementation with YOLO11, RT-DETR, SAM2 models
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Optional
import os
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloads and caching with organized folder structure"""

    def __init__(self, base_path="app/models/saved_models"):
        self.models_dir = Path(base_path)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create organized subfolders
        self.detection_dir = self.models_dir / "detection"
        self.segmentation_dir = self.models_dir / "segmentation"
        self.detection_dir.mkdir(exist_ok=True)
        self.segmentation_dir.mkdir(exist_ok=True)

        # Create model type subfolders
        (self.detection_dir / "yolo11").mkdir(exist_ok=True)
        (self.detection_dir / "yolo12").mkdir(exist_ok=True)
        (self.detection_dir / "rtdetr").mkdir(exist_ok=True)

        (self.segmentation_dir / "yolo11").mkdir(exist_ok=True)
        (self.segmentation_dir / "yolo12").mkdir(exist_ok=True)
        (self.segmentation_dir / "sam2").mkdir(exist_ok=True)

        # Set environment variable to use our custom path
        os.environ['YOLO_CONFIG_DIR'] = str(self.models_dir)

        logger.info(f"[ModelManager] Models directory organized: {self.models_dir}")

    def get_model_path(self, model_name: str) -> str:
        """Get local path for model with organized folder structure"""
        model_file = f"{model_name}.pt"

        # Determine correct subfolder based on model name
        if 'seg' in model_name:
            if 'yolo11' in model_name:
                local_path = self.segmentation_dir / "yolo11" / model_file
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
            local_path.parent.mkdir(exist_ok=True)
            logger.info(f"[ModelManager] Model will be downloaded to: {local_path}")
            return model_name


class ObjectDetector:
    """Object Detection using YOLO11, YOLO12, RT-DETR"""

    def __init__(self, model_name="yolo11m", device=None, output_dir="static/outputs"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.model_manager = ModelManager()
        self.class_names = self._get_coco_classes()
        self.output_dir = Path(output_dir)

        logger.info(f"[ObjectDetector] Initializing {model_name} on {self.device}")

        try:
            self._load_model()
        except ImportError as e:
            logger.error(f"[ObjectDetector] Model not available: {e}")
            # Don't set model to None - raise the error
            raise e
        except Exception as e:
            logger.error(f"[ObjectDetector] Failed to load model: {e}")
            raise e

    def _load_model(self):
        """Load detection model"""
        try:
            from ultralytics import YOLO, RTDETR

            model_path = self.model_manager.get_model_path(self.model_name)

            if 'rtdetr' in self.model_name:
                self.model = RTDETR(model_path)
                logger.info(f"[ObjectDetector] Loaded RT-DETR model: {self.model_name}")
            else:
                self.model = YOLO(model_path)
                logger.info(f"[ObjectDetector] Loaded YOLO model: {self.model_name}")

        except ImportError:
            raise ImportError("ultralytics package required for SOTA models")
        except Exception as e:
            logger.error(f"[ObjectDetector] Error loading model: {e}")
            raise

    def _get_coco_classes(self):
        """Get COCO dataset class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """Detect objects using models - NO MORE MOCK RESULTS"""
        start_time = time.time()

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size

            # FIXED: Always require real model
            if self.model is None:
                raise RuntimeError("Model not loaded - cannot perform detection")

            # Run inference with error handling
            try:
                results = self.model(image_path, conf=confidence_threshold, verbose=False)
            except Exception as e:
                logger.error(f"[ObjectDetector] Model inference failed: {e}")
                raise RuntimeError(f"Model inference failed: {str(e)}")

            # Process results
            objects = []
            if results:
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            try:
                                bbox = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])

                                # Validate class_id
                                if 0 <= class_id < len(self.class_names):
                                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                                    objects.append({
                                        'class_name': self.class_names[class_id],
                                        'class_id': class_id,
                                        'confidence': confidence,
                                        'bbox': bbox.tolist(),
                                        'area': float(area)
                                    })
                                else:
                                    logger.warning(f"[ObjectDetector] Invalid class_id: {class_id}")

                            except Exception as e:
                                logger.error(f"[ObjectDetector] Error processing box: {e}")
                                continue

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(f"[ObjectDetector] Detected {len(objects)} objects in {processing_time}ms")

            return {
                'status': 'success',
                'objects': objects,
                'processing_time': processing_time,
                'model_info': {
                    'architecture': self.model_name.upper(),
                    'confidence_threshold': confidence_threshold,
                    'device': str(self.device),
                    'image_size': f"{image_width}x{image_height}"
                }
            }

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"[ObjectDetector] Error during detection: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'objects': [],
                'processing_time': processing_time
            }

    def visualize_detections(self, image_path: str, objects: List[Dict], output_path: str) -> str:
        """Create detection visualization"""
        try:
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080'
            ]

            for i, obj in enumerate(objects):
                bbox = obj['bbox']
                class_name = obj['class_name']
                confidence = obj['confidence']
                color = colors[i % len(colors)]

                # Draw bounding box
                draw.rectangle(bbox, outline=color, width=3)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                text_bg = [
                    bbox[0], bbox[1] - text_height - 4,
                             bbox[0] + text_width + 8, bbox[1]
                ]
                draw.rectangle(text_bg, fill=color)
                draw.text((bbox[0] + 4, bbox[1] - text_height - 2), label, fill='white', font=font)

            # Save result
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            image.save(output_path, 'PNG', quality=95)

            logger.info(f"[ObjectDetector] Saved detection visualization to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[ObjectDetector] Error creating visualization: {e}")
            return None


class ImageSegmenter:
    """Instance Segmentation using YOLO11-seg, SAM2"""

    def __init__(self, model_name="yolo11m-seg", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.model_manager = ModelManager()
        self._current_masks = None

        logger.info(f"[ImageSegmenter] Initializing {model_name} on {self.device}")

        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"[ImageSegmenter] Model not available: {e}")
            self.model = None

    def _load_model(self):
        """Load segmentation model"""
        try:
            model_path = self.model_manager.get_model_path(self.model_name)

            if 'sam2' in self.model_name:
                from ultralytics import SAM
                sam_model = self.model_name.replace('sam2_', 'sam2.1_')
                self.model = SAM(f"{sam_model}.pt")
                logger.info(f"[ImageSegmenter] Loaded SAM2 model: {sam_model}")
            else:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                logger.info(f"[ImageSegmenter] Loaded YOLO segmentation model: {self.model_name}")

        except Exception as e:
            logger.error(f"[ImageSegmenter] Error loading model {self.model_name}: {e}")
            raise

    def segment_image(self, image_path: str) -> Dict:
        """Perform segmentation"""
        start_time = time.time()

        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            if self.model is None:
                return self._generate_mock_segmentation(original_size)

            # Run segmentation
            results = self.model(image_path, verbose=False)

            segments = []
            masks_data = []

            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    masks_data = masks
                    boxes = result.boxes

                    for i, mask in enumerate(masks):
                        # Get class information
                        if boxes is not None and i < len(boxes.cls):
                            class_id = int(boxes.cls[i])
                            class_name = self._get_class_name(class_id)
                            confidence = float(boxes.conf[i]) if boxes.conf is not None else 1.0
                        else:
                            class_id = i
                            class_name = f'object_{i + 1}'
                            confidence = 1.0

                        # Calculate coverage
                        pixel_count = np.sum(mask > 0.5)
                        coverage = pixel_count / mask.size

                        if coverage > 0.0003:
                            color_seed = hash(class_name) % 16777215
                            color = f"#{color_seed:06x}"

                            segments.append({
                                'class_name': class_name,
                                'class_id': class_id,
                                'pixel_count': int(pixel_count),
                                'coverage': float(coverage),
                                'confidence': confidence,
                                'color': color,
                                'mask_index': i
                            })

            self._current_masks = masks_data
            processing_time = int((time.time() - start_time) * 1000)

            return {
                'status': 'success',
                'segments': sorted(segments, key=lambda x: x['coverage'], reverse=True),
                'processing_time': processing_time,
                'model_info': {
                    'architecture': self.model_name.upper(),
                    'device': str(self.device)
                }
            }

        except Exception as e:
            logger.error(f"[ImageSegmenter] Error during segmentation: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'segments': [],
                'processing_time': 0
            }

    def _get_class_name(self, class_id):
        """Get class name for COCO dataset"""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return coco_classes[class_id] if class_id < len(coco_classes) else f'class_{class_id}'

    def _generate_mock_segmentation(self, image_size: Tuple[int, int]) -> Dict:
        """Generate mock segmentation for demo"""
        logger.info("[ImageSegmenter] Generating mock segmentation results")
        return {
            'status': 'success',
            'segments': [],
            'processing_time': 340,
            'model_info': {
                'architecture': self.model_name.upper(),
                'device': 'CPU (Demo Mode)'
            },
            'demo_mode': True
        }

    def visualize_segmentation(self, image_path: str, segments: List[Dict], output_path: str) -> Optional[str]:
        """Create segmentation visualization"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            height, width = image_array.shape[:2]

            if self._current_masks is None or len(self._current_masks) == 0:
                logger.warning("[ImageSegmenter] No masks available for visualization")
                return None

            overlay = np.zeros((height, width, 3), dtype=np.uint8)

            # Apply each mask with its color
            for i, segment in enumerate(segments):
                if i < len(self._current_masks):
                    mask = self._current_masks[i]

                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()

                    if mask.dtype != bool:
                        mask = mask > 0.5

                    if mask.shape != (height, width):
                        mask_resized = cv2.resize(mask.astype(np.uint8), (width, height),
                                                  interpolation=cv2.INTER_NEAREST)
                        mask = mask_resized.astype(bool)

                    # Parse color
                    color_hex = segment.get('color', '#ff0000')
                    try:
                        color_rgb = tuple(int(color_hex[j:j + 2], 16) for j in (1, 3, 5))
                    except:
                        color_rgb = (255, 0, 0)

                    overlay[mask] = color_rgb

            # Blend with original image
            alpha = 0.4
            blended = cv2.addWeighted(image_array, 1 - alpha, overlay, alpha, 0)

            # Save result
            result_image = Image.fromarray(blended)
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            result_image.save(output_path, 'PNG', quality=95)

            logger.info(f"[ImageSegmenter] Saved segmentation visualization to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[ImageSegmenter] Error creating visualization: {e}")
            return None


class VisionAnalyzer:
    """Main analyzer combining object detection and instance segmentation"""

    def __init__(self, output_dir="static/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "detection").mkdir(exist_ok=True)
        (self.output_dir / "segmentation").mkdir(exist_ok=True)
        (self.output_dir / "video").mkdir(exist_ok=True)

        logger.info(f"[VisionAnalyzer] Initialized with output directory: {self.output_dir}")

    def analyze_complete(self, image_path: str, detection_model: str = "yolo11m",
                         segmentation_model: str = "yolo11m-seg",
                         confidence_threshold: float = 0.5) -> Dict:
        """Perform complete computer vision analysis"""
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
                                 output_dir: str = "static/outputs/video") -> Dict:
    """Process video file with object detection frame by frame"""

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
                             confidence_threshold: float = 0.3) -> Dict:
    """Process single frame from live camera feed - FIXED VERSION"""

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


def _generate_mock_video_results() -> Dict:
    """Generate mock video processing results"""
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


def _generate_mock_live_results() -> Dict:
    """Generate mock live detection results - REMOVED FROM MAIN FLOW"""
    logger.warning("[LiveProcessor] Using mock results - models not available")

    return {
        'status': 'success',
        'objects': [],  # Empty for demo
        'processing_time': 50,
        'model': 'demo_mode',
        'demo_mode': True,
        'message': 'Demo mode - install ultralytics for real detection'
    }


# Global analyzer instance
_vision_analyzer = None


def get_sota_vision_analyzer(output_dir="static/outputs"):
    """Get or create global vision analyzer instance"""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer(output_dir=output_dir)
    return _vision_analyzer


def analyze_vision_complete(image_path: str, detection_model: str = "yolo11m",
                            segmentation_model: str = "yolo11m-seg",
                            confidence_threshold: float = 0.5,
                            output_dir: str = "static/outputs") -> Dict:
    """Computer Vision Analysis Endpoint"""
    analyzer = VisionAnalyzer(output_dir=output_dir)
    return analyzer.analyze_complete(
        image_path=image_path,
        detection_model=detection_model,
        segmentation_model=segmentation_model,
        confidence_threshold=confidence_threshold
    )


if __name__ == "__main__":
    print("Testing Computer Vision Analyzer...")
    try:
        analyzer = VisionAnalyzer()
        print("Vision Analyzer initialized successfully")
        print("Available detection models: YOLO11 (n,s,m,l,x), RT-DETR (l,x)")
        print("Available segmentation models: YOLO11-seg, SAM2.1 (t,s,b,l)")
        print("Analyzer ready for computer vision tasks")
    except Exception as e:
        print(f"Error: {e}")
        print("Running in demo mode - models will generate mock results")