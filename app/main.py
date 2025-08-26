#!/usr/bin/env python3
"""
 CNN Playground - FastAPI Server

FastAPI server for computer vision tasks including image classification,
object detection, instance segmentation, and video processing.
"""
import sys
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
from pathlib import Path
import shutil
from datetime import datetime
import logging
import asyncio
import mimetypes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application instance.

    Returns:
        Configured FastAPI application
    """
    app: FastAPI = FastAPI(
        title="CNN Playground - Computer Vision Platform",
        description="Deep learning platform for image classification, object detection, and video analysis",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def setup_directories() -> None:
    """
    Create necessary application directories with structure.
    """
    directory_structure: List[str] = [
        "static/css",
        "static/js",
        "static/uploads",
        "static/outputs/gradcam",
        "static/outputs/feature_maps",
        "static/outputs/detection",
        "static/outputs/segmentation",
        "static/outputs/video",
        "static/outputs/frames",
        "static/outputs/models",
        "static/outputs/mnist",
        "app/models/saved_models/detection/yolo11",
        "app/models/saved_models/detection/yolo12",
        "app/models/saved_models/detection/rtdetr",
        "app/models/saved_models/segmentation/yolo11",
        "app/models/saved_models/segmentation/yolo12",
        "app/models/saved_models/segmentation/sam2",
        "temp/video_frames",
        "temp/processing"
    ]

    for directory in directory_structure:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("Application directory structure created")


# Initialize FastAPI application
app: FastAPI = create_application()
setup_directories()

# Mount static file handlers
app.mount("/static", StaticFiles(directory="static"), name="static")


class CustomStaticFiles(StaticFiles):
    """Custom static files handler with proper MIME types for video files."""

    def __init__(self, directory: str, name: Optional[str] = None) -> None:
        super().__init__(directory=directory)
        self.name = name

    async def get_response(self, path: str, scope: Dict[str, Any]) -> FileResponse:
        """
        Override to set proper MIME types for video files.

        Args:
            path: File path to serve
            scope: ASGI scope

        Returns:
            FileResponse with proper content type

        Raises:
            HTTPException: If file not found
        """
        try:
            response: FileResponse = await super().get_response(path, scope)

            # Set proper MIME types for video files
            if path.endswith('.mp4'):
                response.headers["content-type"] = "video/mp4"
            elif path.endswith('.avi'):
                response.headers["content-type"] = "video/x-msvideo"
            elif path.endswith('.mov'):
                response.headers["content-type"] = "video/quicktime"
            elif path.endswith('.webm'):
                response.headers["content-type"] = "video/webm"

            return response
        except Exception as e:
            logger.error(f"Error serving file {path}: {e}")
            raise HTTPException(status_code=404, detail="File not found")


app.mount("/outputs", CustomStaticFiles(directory="static/outputs"), name="outputs")

templates: Jinja2Templates = Jinja2Templates(directory="templates")


# ================================
# PAGE ROUTES
# ================================

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request) -> HTMLResponse:
    """
    Main landing page with feature overview.

    Args:
        request: FastAPI request object

    Returns:
        Rendered homepage template
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "home"
    })


@app.get("/imagenet", response_class=HTMLResponse)
async def imagenet_classification_page(request: Request) -> HTMLResponse:
    """
    ImageNet classification interface with CNN analysis.

    Args:
        request: FastAPI request object

    Returns:
        Rendered ImageNet template
    """
    return templates.TemplateResponse("imagenet.html", {
        "request": request,
        "page": "imagenet"
    })


@app.get("/mnist", response_class=HTMLResponse)
async def mnist_playground_page(request: Request) -> HTMLResponse:
    """
    MNIST digit recognition playground with interactive canvas.

    Args:
        request: FastAPI request object

    Returns:
        Rendered MNIST template
    """
    return templates.TemplateResponse("mnist.html", {
        "request": request,
        "page": "mnist"
    })


@app.get("/vision", response_class=HTMLResponse)
async def computer_vision_page(request: Request) -> HTMLResponse:
    """
    Computer vision interface with multi-mode processing.

    Args:
        request: FastAPI request object

    Returns:
        Rendered computer vision template
    """
    return templates.TemplateResponse("computer_vision.html", {
        "request": request,
        "page": "vision"
    })


# ================================
# COMPUTER VISION API ENDPOINTS
# ================================

@app.post("/api/analyze_vision")
async def analyze_vision_endpoint(
        file: UploadFile = File(...),
        detection_model: str = Form("yolo11m"),
        segmentation_model: str = Form("yolo11m-seg"),
        confidence_threshold: float = Form(0.5)
) -> JSONResponse:
    """
    Computer vision analysis with detection and segmentation.

    Args:
        file: Uploaded image file
        detection_model: Object detection model selection
        segmentation_model: Instance segmentation model selection
        confidence_threshold: Minimum confidence for detections

    Returns:
        Analysis results with visualizations

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    # Validate input file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Generate unique filename with timestamp
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_extension: str = Path(file.filename).suffix.lower()
        filename: str = f"{timestamp}_vision{file_extension}"
        file_path: str = f"static/uploads/{filename}"

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Vision analysis started for {filename} with models: {detection_model}, {segmentation_model}")

        try:
            from app.models.vision_analyzer import analyze_vision_complete

            # Perform analysis
            results: Dict[str, Any] = analyze_vision_complete(
                image_path=file_path,
                detection_model=detection_model,
                segmentation_model=segmentation_model,
                confidence_threshold=confidence_threshold,
                output_dir="static/outputs"
            )

            # Enhance results with metadata
            results.update({
                'filename': filename,
                'file_url': f"/static/uploads/{filename}",
                'timestamp': timestamp,
                'configuration': {
                    'detection_model': detection_model,
                    'segmentation_model': segmentation_model,
                    'confidence_threshold': confidence_threshold
                }
            })

            if results['status'] == 'success':
                logger.info(f"Vision analysis completed for {filename}")
                return JSONResponse(results)
            else:
                logger.error(f"Vision analysis failed: {results.get('message', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=results.get('message', 'Analysis failed'))

        except ImportError as e:
            logger.warning(f"Vision analyzer not available: {e}")
            # Return demonstration results
            return await generate_demo_vision_results(filename, file, detection_model, segmentation_model,
                                                      confidence_threshold)

    except Exception as e:
        logger.error(f"Error processing vision analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/process_video")
async def process_video_endpoint(
        video: UploadFile = File(...),
        detection_model: str = Form("yolo11n"),
        frame_skip: int = Form(2),
        confidence_threshold: float = Form(0.4)
) -> JSONResponse:
    """
    Process video file with object detection frame by frame.

    Args:
        video: Uploaded video file
        detection_model: Object detection model for video processing
        frame_skip: Process every Nth frame (1=all frames, 2=every other frame)
        confidence_threshold: Minimum confidence for detections

    Returns:
        Video processing results with download URL

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    # Validate video file
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Check file size (limit to 100MB for processing efficiency)
    if video.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Video file too large (max 100MB)")

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_extension = Path(video.filename).suffix.lower()
        filename = f"{timestamp}_video{file_extension}"
        input_path = f"static/uploads/{filename}"

        # Save uploaded video
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        logger.info(f"Video processing started for {filename} with model: {detection_model}")

        try:
            from app.models.vision_analyzer import process_video_complete

            # Process video with detection
            results = await process_video_complete(
                video_path=input_path,
                detection_model=detection_model,
                frame_skip=frame_skip,
                confidence_threshold=confidence_threshold,
                output_dir="static/outputs/video"
            )

            results.update({
                'filename': filename,
                'input_url': f"/static/uploads/{filename}",
                'timestamp': timestamp,
                'configuration': {
                    'detection_model': detection_model,
                    'frame_skip': frame_skip,
                    'confidence_threshold': confidence_threshold
                }
            })

            if results['status'] == 'success':
                logger.info(f"Video processing completed successfully for {filename}")
                return JSONResponse(results)
            else:
                raise HTTPException(status_code=500, detail=results.get('message', 'Video processing failed'))

        except ImportError as e:
            logger.warning(f"Video processor not available: {e}")
            # Return mock processing results
            return await generate_demo_video_results(filename, video, detection_model, frame_skip, confidence_threshold)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@app.post("/api/process_live_frame")
async def process_live_frame_endpoint(
        frame: UploadFile = File(...),
        detection_model: str = Form("yolo11n"),
        confidence_threshold: float = Form(0.3)
) -> JSONResponse:
    """
    Process single frame from live camera feed for real-time detection.

    Args:
        frame: Camera frame as image file
        detection_model: Object detection model for live processing
        confidence_threshold: Minimum confidence for detections

    Returns:
        Real-time detection results

    Raises:
        HTTPException: If frame processing fails
    """
    if not frame.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Frame must be an image")

    try:
        # Generate unique frame identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{timestamp}.jpg"
        frame_path = f"temp/processing/{filename}"

        # Save frame temporarily
        with open(frame_path, "wb") as buffer:
            shutil.copyfileobj(frame.file, buffer)

        try:
            from app.models.vision_analyzer import process_live_frame

            # Process frame with fast detection
            results = await process_live_frame(
                frame_path=frame_path,
                detection_model=detection_model,
                confidence_threshold=confidence_threshold
            )

            # Clean up temporary frame
            if os.path.exists(frame_path):
                os.remove(frame_path)

            results.update({
                'timestamp': timestamp,
                'model': detection_model
            })

            return JSONResponse(results)

        except ImportError:
            # Return mock live detection for demo
            return JSONResponse({
                'status': 'success',
                'objects': [
                    {
                        'class_name': 'person',
                        'confidence': 0.85,
                        'bbox': [100, 50, 300, 400]
                    }
                ],
                'processing_time': 45,
                'timestamp': timestamp,
                'model': detection_model
            })

    except Exception as e:
        logger.error(f"Error processing live frame: {str(e)}")
        # Clean up on error
        try:
            if 'frame_path' in locals() and os.path.exists(frame_path):
                os.remove(frame_path)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


# ================================
# IMAGENET API ENDPOINTS
# ================================

@app.post("/api/analyze")
async def analyze_imagenet_image(
        file: UploadFile = File(...),
        dataset: str = Form("imagenet"),
        model: str = Form("resnet50")
) -> JSONResponse:
    """
    Analyze image using CNN with feature maps and Grad-CAM visualization.

    Args:
        file: Uploaded image file
        dataset: Target dataset (imagenet, cifar10, etc.)
        model: CNN architecture selection

    Returns:
        Classification results with visualizations

    Raises:
        HTTPException: If file validation fails or analysis errors occur
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_extension = Path(file.filename).suffix.lower()
        filename = f"{timestamp}_imagenet{file_extension}"
        file_path = f"static/uploads/{filename}"

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"ImageNet analysis started for {filename} with model: {model}")

        try:
            from app.models.cnn_analyzer import analyze_image_endpoint

            # Perform CNN analysis with feature visualization
            results = analyze_image_endpoint(file_path, output_dir="static/outputs")

            results.update({
                'filename': filename,
                'file_url': f"/static/uploads/{filename}",
                'dataset': dataset,
                'model': model,
                'timestamp': timestamp
            })

            if results['status'] == 'success':
                logger.info(f"ImageNet analysis completed successfully for {filename}")
                return JSONResponse(results)
            else:
                logger.error(f"ImageNet analysis failed: {results.get('message', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=results.get('message', 'Analysis failed'))

        except ImportError as e:
            logger.warning(f"CNN analyzer not available: {e}")
            # Return demonstration results
            return await generate_demo_imagenet_results(filename, file, dataset, model)

    except Exception as e:
        logger.error(f"Error processing ImageNet analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ================================
# MNIST API ENDPOINTS
# ================================

@app.post("/api/predict_mnist")
async def predict_mnist_digit(request: Request) -> JSONResponse:
    """
    Predict MNIST digit from canvas drawing with confidence analysis.

    Args:
        request: FastAPI request containing image_data in JSON

    Returns:
        Digit prediction with confidence scores

    Raises:
        HTTPException: If prediction fails or invalid data provided
    """
    try:
        body = await request.json()
        image_data = body.get('image_data')

        if not image_data:
            raise HTTPException(status_code=400, detail="Missing image_data field")

        logger.info("Processing MNIST digit prediction request")

        try:
            from app.models.mnist_classifier import predict_mnist_digit
            result = predict_mnist_digit(image_data)

            if result['status'] == 'success':
                logger.info(f"MNIST prediction: digit {result['predicted_digit']} "
                            f"(confidence: {result['confidence']:.3f})")
                return JSONResponse(result)
            else:
                logger.error(f"MNIST prediction failed: {result.get('message', 'Unknown error')}")
                raise HTTPException(status_code=500, detail=result.get('message', 'Prediction failed'))

        except ImportError as e:
            logger.warning(f"MNIST classifier not available: {e}")
            # Return mock prediction for demonstration
            return await generate_demo_mnist_results()

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error in MNIST prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/mnist/model_info")
async def get_mnist_model_information() -> JSONResponse:
    """
    Get MNIST model information and statistics.

    Returns:
        Model architecture and performance details
    """
    try:
        from app.models.mnist_classifier import get_mnist_model_info
        return JSONResponse(get_mnist_model_info())
    except ImportError:
        return JSONResponse({
            'architecture': 'LeNet-5 (Demo)',
            'total_parameters': 62006,
            'trainable_parameters': 62006,
            'input_size': '28x28 grayscale',
            'output_classes': 10,
            'device': 'CPU',
            'preprocessing': 'Standard MNIST normalization',
            'demo_mode': True
        })


# ================================
# UTILITY API ENDPOINTS
# ================================

@app.get("/api/models/{dataset}")
async def get_available_models(dataset: str) -> JSONResponse:
    """
    Get available model configurations for specified dataset.

    Args:
        dataset: Target dataset name

    Returns:
        Available models and default selections

    Raises:
        HTTPException: If dataset not supported
    """
    models_configuration: Dict[str, Dict[str, Any]] = {
        "imagenet": {
            "models": ["resnet50", "vgg16", "efficientnet_b0", "mobilenet_v2"],
            "default": "resnet50",
            "description": "Pre-trained models for ImageNet classification"
        },
        "vision": {
            "detection_models": {
                "yolo11n": "YOLO11-Nano (Fastest)",
                "yolo11s": "YOLO11-Small",
                "yolo11m": "YOLO11-Medium (Recommended)",
                "yolo11l": "YOLO11-Large",
                "yolo11x": "YOLO11-XLarge (Most Accurate)",
                "yolo12n": "YOLO12-Nano (Experimental)",
                "rtdetr-l": "RT-DETR Large"
            },
            "segmentation_models": {
                "yolo11n-seg": "YOLO11-Nano Segmentation",
                "yolo11s-seg": "YOLO11-Small Segmentation",
                "yolo11m-seg": "YOLO11-Medium Segmentation",
                "yolo11l-seg": "YOLO11-Large Segmentation",
                "sam2_b": "SAM 2.1 Base (Universal)",
                "sam2_l": "SAM 2.1 Large (Best Quality)"
            },
            "default_detection": "yolo11m",
            "default_segmentation": "yolo11m-seg"
        },
        "mnist": {
            "models": ["lenet5", "improved_lenet", "custom_cnn"],
            "default": "improved_lenet",
            "description": "Models trained on MNIST handwritten digits"
        }
    }

    if dataset not in models_configuration:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not supported")

    return JSONResponse(models_configuration[dataset])


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Application health check with system status.

    Returns:
        Health status and system information
    """
    return {
        "status": "healthy",
        "message": "CNN Playground is operational",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "ImageNet Classification",
            "MNIST Recognition",
            "Computer Vision Analysis",
            "Video Processing",
            "Live Camera Detection"
        ]
    }


@app.get("/api/status")
async def get_comprehensive_system_status() -> JSONResponse:
    """
    Get system status including model availability.

    Returns:
        System status with model and capability information
    """
    import torch

    # Check directory structure
    directories_status: Dict[str, bool] = {}
    required_directories: List[str] = [
        "static", "templates", "static/outputs",
        "app/models", "static/uploads", "temp"
    ]
    for directory in required_directories:
        directories_status[directory] = os.path.exists(directory)

    # Check model availability
    models_status: Dict[str, bool] = {}

    # MNIST classifier availability
    try:
        from app.models.mnist_classifier import get_mnist_classifier
        models_status["mnist_classifier"] = True
    except (ImportError, Exception):
        models_status["mnist_classifier"] = False

    # CNN analyzer availability
    try:
        from app.models.cnn_analyzer import get_analyzer
        models_status["cnn_analyzer"] = True
    except (ImportError, Exception):
        models_status["cnn_analyzer"] = False

    # Vision analyzer availability
    try:
        from app.models.vision_analyzer import get_sota_vision_analyzer
        models_status["vision_analyzer"] = True
    except (ImportError, Exception):
        models_status["vision_analyzer"] = False

    # System capabilities
    capabilities: Dict[str, Union[bool, int]] = {
        "pytorch_available": True,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "multiprocessing": True,
        "video_processing": models_status.get("vision_analyzer", False),
        "live_camera": models_status.get("vision_analyzer", False)
    }

    status_report: Dict[str, Any] = {
        "server_status": "online",
        "pytorch_version": torch.__version__,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "directories": directories_status,
        "models": models_status,
        "capabilities": capabilities,
        "features": {
            "imagenet_classification": models_status.get("cnn_analyzer", False),
            "mnist_recognition": models_status.get("mnist_classifier", False),
            "feature_visualization": models_status.get("cnn_analyzer", False),
            "gradcam": models_status.get("cnn_analyzer", False),
            "object_detection": models_status.get("vision_analyzer", False),
            "instance_segmentation": models_status.get("vision_analyzer", False),
            "video_processing": models_status.get("vision_analyzer", False),
            "live_camera": models_status.get("vision_analyzer", False)
        }
    }

    return JSONResponse(status_report)


# ================================
# DEMO RESULT GENERATORS
# ================================

async def generate_demo_vision_results(filename: str, file: UploadFile, detection_model: str,
                                       segmentation_model: str, confidence_threshold: float) -> JSONResponse:
    """
    Generate demonstration results for computer vision analysis.

    Args:
        filename: Generated filename for the uploaded file
        file: Original uploaded file
        detection_model: Selected detection model
        segmentation_model: Selected segmentation model
        confidence_threshold: Confidence threshold setting

    Returns:
        Mock analysis results in standard format
    """
    import random

    return JSONResponse({
        "status": "success",
        "filename": filename,
        "file_url": f"/static/uploads/{filename}",
        "image_info": {
            "width": 1024,
            "height": 768,
            "format": "JPEG"
        },
        "detection": {
            "image_url": None,
            "objects": [
                {
                    "class_name": "person",
                    "confidence": 0.92,
                    "bbox": [150, 100, 400, 500],
                    "area": 87500
                },
                {
                    "class_name": "car",
                    "confidence": 0.78,
                    "bbox": [500, 200, 800, 400],
                    "area": 60000
                }
            ],
            "processing_time": random.randint(80, 150),
            "demo_mode": True
        },
        "segmentation": {
            "image_url": None,
            "segments": [
                {
                    "class_name": "person",
                    "coverage": 0.15,
                    "pixel_count": 122880,
                    "color": "#FF6B6B"
                },
                {
                    "class_name": "car",
                    "coverage": 0.08,
                    "pixel_count": 65536,
                    "color": "#4ECDC4"
                }
            ],
            "processing_time": random.randint(200, 400),
            "demo_mode": True
        },
        "model_info": {
            "detection": {
                "architecture": detection_model.upper(),
                "model_size": "Medium",
                "parameters": "~20M",
                "input_size": "640x640"
            },
            "segmentation": {
                "architecture": segmentation_model.upper().replace('-', ' '),
                "parameters": "~22M",
                "classes": "80"
            },
            "device": "CPU (Demo Mode)"
        },
        "configuration": {
            "detection_model": detection_model,
            "segmentation_model": segmentation_model,
            "confidence_threshold": confidence_threshold
        },
        "message": "Demo mode - Vision analyzer models not loaded"
    })


async def generate_demo_video_results(filename: str, video: UploadFile, detection_model: str,
                                      frame_skip: int, confidence_threshold: float) -> JSONResponse:
    """
    Generate demonstration results for video processing.

    Args:
        filename: Generated filename for the uploaded video
        video: Original uploaded video file
        detection_model: Selected detection model
        frame_skip: Frame skip setting
        confidence_threshold: Confidence threshold setting

    Returns:
        Mock video processing results
    """
    import random

    return JSONResponse({
        "status": "success",
        "filename": filename,
        "input_url": f"/static/uploads/{filename}",
        "processed_video_url": None,
        "stats": {
            "total_frames": random.randint(500, 1500),
            "processing_time": random.randint(45, 120),
            "objects_detected": random.randint(50, 200),
            "avg_fps": random.randint(8, 15)
        },
        "configuration": {
            "detection_model": detection_model,
            "frame_skip": frame_skip,
            "confidence_threshold": confidence_threshold
        },
        "demo_mode": True,
        "message": "Demo mode - Video processing not available"
    })


async def generate_demo_imagenet_results(filename: str, file: UploadFile, dataset: str, model: str) -> JSONResponse:
    """
    Generate demonstration results for ImageNet analysis.

    Args:
        filename: Generated filename for the uploaded file
        file: Original uploaded file
        dataset: Selected dataset
        model: Selected model

    Returns:
        Mock ImageNet analysis results
    """
    demo_classes = [
        "Golden Retriever", "Labrador Retriever", "German Shepherd",
        "Bulldog", "Poodle", "Beagle", "Rottweiler", "Border Collie"
    ]

    import random
    random.shuffle(demo_classes)

    predictions = []
    remaining_confidence = 1.0

    for i, class_name in enumerate(demo_classes[:5]):
        if i == 0:
            confidence = random.uniform(0.4, 0.8)
        else:
            max_conf = min(remaining_confidence * 0.8, 0.3)
            confidence = random.uniform(0.01, max_conf)

        remaining_confidence -= confidence
        predictions.append({
            "class_name": class_name,
            "confidence": confidence
        })

    return JSONResponse({
        "status": "success",
        "filename": filename,
        "dataset": dataset,
        "model": model,
        "file_url": f"/static/uploads/{filename}",
        "predictions": predictions,
        "gradcam_url": None,
        "feature_maps": {"overview_url": None, "individual_layers": {}},
        "model_info": {
            "architecture": "ResNet50",
            "dataset": "ImageNet",
            "input_size": "224x224 (with aspect ratio preservation)",
            "device": "CPU (Demo Mode)",
            "parameters": "25.6M"
        },
        "demo_mode": True,
        "message": "Demo mode - CNN analyzer not loaded"
    })


async def generate_demo_mnist_results() -> JSONResponse:
    """
    Generate demonstration results for MNIST prediction.

    Returns:
        Mock MNIST prediction results
    """
    import random

    mock_digit = random.randint(0, 9)
    mock_confidence = random.uniform(0.75, 0.95)

    predictions = []
    for digit in range(10):
        if digit == mock_digit:
            confidence = mock_confidence
        else:
            confidence = random.uniform(0.001, 0.15)

        predictions.append({
            'digit': digit,
            'probability': confidence,
            'percentage': confidence * 100
        })

    predictions.sort(key=lambda x: x['probability'], reverse=True)

    return JSONResponse({
        "status": "success",
        "predicted_digit": mock_digit,
        "confidence": mock_confidence,
        "predictions": predictions,
        "processed_image_url": None,
        "model_info": {
            "architecture": "LeNet-5 (Demo)",
            "total_parameters": 62006,
            "input_size": "28x28 grayscale",
            "device": "CPU (Demo Mode)"
        },
        "demo_mode": True,
        "message": "Demo mode - MNIST classifier not loaded"
    })


# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> Union[JSONResponse, HTMLResponse]:
    """
    Handle 404 errors with appropriate responses.

    Args:
        request: FastAPI request object
        exc: Exception details

    Returns:
        Error response (JSON for API, HTML for pages)
    """
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={
                "error": "API endpoint not found",
                "path": request.url.path,
                "available_endpoints": [
                    "/api/analyze_vision",
                    "/api/process_video",
                    "/api/process_live_frame",
                    "/api/analyze",
                    "/api/predict_mnist",
                    "/api/status"
                ]
            }
        )

    try:
        return templates.TemplateResponse("404.html", {
            "request": request,
            "page": "error"
        }, status_code=404)
    except:
        return HTMLResponse(
            content="<h1>404 - Page Not Found</h1>",
            status_code=404
        )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> Union[JSONResponse, HTMLResponse]:
    """
    Handle 500 errors with logging and appropriate responses.

    Args:
        request: FastAPI request object
        exc: Exception details

    Returns:
        Error response with logging
    """
    logger.error(f"Internal server error on {request.url.path}: {exc}", exc_info=True)

    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )

    try:
        return templates.TemplateResponse("500.html", {
            "request": request,
            "page": "error"
        }, status_code=500)
    except:
        return HTMLResponse(
            content="<h1>500 - Internal Server Error</h1>",
            status_code=500
        )


# ================================
# APPLICATION LIFECYCLE
# ================================

@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize application on startup with setup tasks.
    """
    logger.info("CNN Playground server initializing...")

    # Verify critical directory structure
    required_dirs: List[str] = ["templates", "static", "app/models", "temp"]
    missing_dirs: List[str] = [d for d in required_dirs if not Path(d).exists()]

    if missing_dirs:
        logger.warning(f"Missing critical directories: {missing_dirs}")
        for missing_dir in missing_dirs:
            Path(missing_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created missing directory: {missing_dir}")
    else:
        logger.info("Directory structure verified")

    # Pre-load models for improved first-request performance
    model_load_tasks: List[Any] = []

    # MNIST classifier pre-loading
    async def load_mnist() -> None:
        try:
            from app.models.mnist_classifier import get_mnist_classifier
            get_mnist_classifier()
            logger.info("MNIST classifier pre-loaded")
        except ImportError:
            logger.warning("MNIST classifier not available")
        except Exception as e:
            logger.error(f"Failed to pre-load MNIST classifier: {e}")

    # Vision analyzer pre-loading
    async def load_vision() -> None:
        try:
            from app.models.vision_analyzer import get_sota_vision_analyzer
            get_sota_vision_analyzer()
            logger.info("Computer vision analyzer pre-loaded")
        except ImportError:
            logger.warning("Computer vision analyzer not available")
        except Exception as e:
            logger.error(f"Failed to pre-load vision analyzer: {e}")

    model_load_tasks = [load_mnist(), load_vision()]

    # Load models concurrently
    try:
        await asyncio.gather(*model_load_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error during model pre-loading: {e}")

    logger.info("CNN Playground startup completed")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Cleanup resources on application shutdown.
    """
    logger.info("CNN Playground shutting down...")

    # Cleanup temporary files
    temp_dirs: List[str] = ["temp/processing", "temp/video_frames"]
    for temp_dir in temp_dirs:
        temp_path: Path = Path(temp_dir)
        if temp_path.exists():
            try:
                shutil.rmtree(temp_path)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_dir}: {e}")

    logger.info("CNN Playground shutdown completed")


# ================================
# MAIN APPLICATION ENTRY POINT
# ================================


if __name__ == "__main__":
    """
    Application entry point with environment setup.

    Ensures proper working directory and Python path configuration
    regardless of how the script is executed.
    """
    print("\n" + "=" * 70)
    print("CNN PLAYGROUND")
    print("=" * 70)

    # Resolve current file path and check if running from app subdirectory
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "app":
        # Set working directory to project root for proper resource access
        project_root = current_file.parent.parent
        os.chdir(project_root)

        # Add project root to Python path for module imports
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    print(f"Working directory set to: {os.getcwd()}")


    # System verification
    structure_verification: Dict[str, bool] = {
        "Templates": Path("templates").exists(),
        "Static files": Path("static").exists(),
        "Model modules": Path("app/models").exists(),
        "Output directories": Path("static/outputs").exists(),
        "Temporary storage": Path("temp").exists()
    }

    print("\nSystem verification:")
    for component, exists in structure_verification.items():
        status: str = "✓ OK" if exists else "✗ MISSING"
        print(f"  {component:<20}: {status}")

    print(f"\nServer configuration:")
    print(f"  Host: 127.0.0.1")
    print(f"  Port: 8000")
    print(f"  Reload: Enabled (Development)")
    print(f"  Log level: INFO")

    print(f"\nAccess URLs:")
    print(f"  Main interface:        http://localhost:8000")
    print(f"  ImageNet analysis:     http://localhost:8000/imagenet")
    print(f"  MNIST playground:      http://localhost:8000/mnist")
    print(f"  Computer Vision:       http://localhost:8000/vision")
    print(f"  API documentation:     http://localhost:8000/docs")
    print(f"  Health check:          http://localhost:8000/health")

    print("=" * 70 + "\n")

    # Start server with configuration
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        reload_excludes=["static/outputs/*", "temp/*"]
    )
