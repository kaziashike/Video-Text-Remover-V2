import os
import sys
import tempfile
from typing import Optional, List
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

# Better error handling for imports
try:
    # Import config from backend
    from backend import config
    from backend.main import SubtitleRemover
    BACKEND_AVAILABLE = True
    BACKEND_ERROR = None
except Exception as e:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(e)
    print(f"Backend import error: {e}")

app = FastAPI(title="Video Subtitle Remover API",
              description="API for removing subtitles from videos using AI",
              version="1.1.1" if BACKEND_AVAILABLE else "0.1.0")

class InpaintMode(str, Enum):
    STTN = "sttn"
    LAMA = "lama"
    PROPAINTER = "propainter"

# Store for processing tasks
processing_tasks = {}

@app.get("/")
async def root():
    if not BACKEND_AVAILABLE:
        return {"message": "Video Subtitle Remover API - Backend not available", 
                "version": "1.1.1",
                "error": BACKEND_ERROR,
                "status": "degraded"}
    return {"message": "Video Subtitle Remover API", "version": config.VERSION}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if not BACKEND_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": BACKEND_ERROR,
                "details": "Backend modules not available"
            }
        )
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_info = str(config.device) if hasattr(config, 'device') else "unknown"
        use_dml = getattr(config, 'USE_DML', False)
        onnx_providers = getattr(config, 'ONNX_PROVIDERS', [])
        
        return {
            "status": "healthy",
            "version": config.VERSION,
            "cuda_available": cuda_available,
            "device": device_info,
            "use_dml": use_dml,
            "onnx_providers": onnx_providers
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "details": "Health check failed"
            }
        )

@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = Form(default="sttn"),
    use_h264: bool = Form(default=True),
    threshold_height_width_difference: int = Form(default=10),
    subtitle_area_deviation_pixel: int = Form(default=20),
    threshold_height_difference: int = Form(default=20),
    pixel_tolerance_y: int = Form(default=20),
    pixel_tolerance_x: int = Form(default=20),
    sttn_skip_detection: bool = Form(default=True),
    sttn_neighbor_stride: int = Form(default=5),
    sttn_reference_length: int = Form(default=10),
    sttn_max_load_num: int = Form(default=50),
    propainter_max_load_num: int = Form(default=70),
    lama_super_fast: bool = Form(default=False)
):
    """
    Process a video file to remove subtitles with configurable settings.
    Returns a task ID that can be used to check status and download the result.
    """
    if not BACKEND_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Backend not available: {BACKEND_ERROR}"
        )
    
    try:
        # Convert mode string to enum
        mode_enum = InpaintMode(mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Must be one of: sttn, lama, propainter"
        )
    
    try:
        # Save uploaded video to a temporary file
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1])
        temp_video_path = temp_video_file.name
        temp_video_file.close()
        
        # Write uploaded file to temp location
        with open(temp_video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Create a task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Configure settings based on request
        config.USE_H264 = use_h264
        config.MODE = config.InpaintMode(mode_enum.value)
        config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = threshold_height_width_difference
        config.SUBTITLE_AREA_DEVIATION_PIXEL = subtitle_area_deviation_pixel
        config.THRESHOLD_HEIGHT_DIFFERENCE = threshold_height_difference
        config.PIXEL_TOLERANCE_Y = pixel_tolerance_y
        config.PIXEL_TOLERANCE_X = pixel_tolerance_x
        config.STTN_SKIP_DETECTION = sttn_skip_detection
        config.STTN_NEIGHBOR_STRIDE = sttn_neighbor_stride
        config.STTN_REFERENCE_LENGTH = sttn_reference_length
        config.STTN_MAX_LOAD_NUM = sttn_max_load_num
        config.PROPAINTER_MAX_LOAD_NUM = propainter_max_load_num
        config.LAMA_SUPER_FAST = lama_super_fast
        
        # Create output file path
        video_dir = os.path.dirname(temp_video_path)
        video_name = os.path.splitext(os.path.basename(temp_video_path))[0]
        output_video_path = os.path.join(video_dir, f"{video_name}_no_sub.mp4")
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "processing",
            "input_path": temp_video_path,
            "output_path": output_video_path,
            "progress": 0
        }
        
        # Process video in background
        background_tasks.add_task(process_video_task, task_id, temp_video_path, output_video_path)
        
        return {
            "task_id": task_id,
            "message": "Video processing started",
            "status_url": f"/status/{task_id}",
            "download_url": f"/download/{task_id}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

def process_video_task(task_id: str, input_path: str, output_path: str):
    """
    Background task to process the video
    """
    if not BACKEND_AVAILABLE:
        if task_id in processing_tasks:
            processing_tasks[task_id]["status"] = "failed"
            processing_tasks[task_id]["error"] = f"Backend not available: {BACKEND_ERROR}"
        return
    
    try:
        # Create subtitle remover
        remover = SubtitleRemover(input_path)
        
        # Override the output path
        remover.video_out_name = output_path
        
        # Add progress tracking
        def progress_callback(progress):
            if task_id in processing_tasks:
                processing_tasks[task_id]["progress"] = progress
        
        # Run processing
        remover.run()
        
        # Mark as completed
        if task_id in processing_tasks:
            processing_tasks[task_id]["status"] = "completed"
            processing_tasks[task_id]["progress"] = 100
            
    except Exception as e:
        # Mark as failed
        if task_id in processing_tasks:
            processing_tasks[task_id]["status"] = "failed"
            processing_tasks[task_id]["error"] = str(e)
        print(f"Error processing video: {e}")

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Get the status of a processing task
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = processing_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "error": task.get("error", None) if task["status"] == "failed" else None
    }

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    Download the processed video
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task['status']}")
    
    if not os.path.exists(task["output_path"]):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        task["output_path"],
        media_type='video/mp4',
        filename=os.path.basename(task["output_path"])
    )

if __name__ == "__main__":
    print("Starting Video Subtitle Remover API...")
    print("Access the API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)