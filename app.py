import os
import sys
import tempfile
from typing import Optional, List
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

# Serve static files and templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Create a simple HTML page for the web interface
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Subtitle Remover</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.1); 
        }
        h1 { 
            color: #333; 
            text-align: center; 
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold; 
        }
        input, select { 
            width: 100%; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            box-sizing: border-box; 
        }
        input[type="checkbox"] {
            width: auto;
            margin-right: 10px;
        }
        button { 
            background-color: #4CAF50; 
            color: white; 
            padding: 12px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            width: 100%; 
            font-size: 16px; 
        }
        button:hover { 
            background-color: #45a049; 
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .status { 
            margin-top: 20px; 
            padding: 15px; 
            border-radius: 5px; 
        }
        .status-processing { 
            background-color: #fff3cd; 
            border: 1px solid #ffeaa7; 
        }
        .status-completed { 
            background-color: #d4edda; 
            border: 1px solid #c3e6cb; 
        }
        .status-error { 
            background-color: #f8d7da; 
            border: 1px solid #f5c6cb; 
        }
        .hidden { 
            display: none; 
        }
        .progress-bar { 
            width: 100%; 
            background-color: #f0f0f0; 
            border-radius: 5px; 
            overflow: hidden; 
        }
        .progress { 
            height: 20px; 
            background-color: #4CAF50; 
            width: 0%; 
            transition: width 0.3s; 
        }
        .instructions {
            background-color: #e9f7fe;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .error-details {
            font-family: monospace;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 10px;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }
        .download-link {
            display: block;
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .download-link:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Subtitle Remover</h1>
        
        <div class="instructions">
            <h3>Instructions:</h3>
            <p>1. Select a video file to process</p>
            <p>2. Adjust parameters as needed</p>
            <p>3. Click "Process Video" to start</p>
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="video">Select Video File:</label>
                <input type="file" id="video" name="video" accept="video/*" required>
            </div>
            
            <div class="form-group">
                <label for="mode">Processing Mode:</label>
                <select id="mode" name="mode">
                    <option value="sttn" selected>STTN (Recommended)</option>
                    <option value="lama">LAMA</option>
                    <option value="propainter">PROPAINTER</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="use_h264" name="use_h264" checked> Use H264 Codec
                </label>
            </div>
            
            <div class="form-group">
                <label for="threshold_height_width_difference">Threshold Height Width Difference:</label>
                <input type="number" id="threshold_height_width_difference" name="threshold_height_width_difference" value="10" min="0">
            </div>
            
            <div class="form-group">
                <label for="subtitle_area_deviation_pixel">Subtitle Area Deviation Pixel:</label>
                <input type="number" id="subtitle_area_deviation_pixel" name="subtitle_area_deviation_pixel" value="20" min="0">
            </div>
            
            <div class="form-group">
                <label for="threshold_height_difference">Threshold Height Difference:</label>
                <input type="number" id="threshold_height_difference" name="threshold_height_difference" value="20" min="0">
            </div>
            
            <div class="form-group">
                <label for="pixel_tolerance_y">Pixel Tolerance Y:</label>
                <input type="number" id="pixel_tolerance_y" name="pixel_tolerance_y" value="20" min="0">
            </div>
            
            <div class="form-group">
                <label for="pixel_tolerance_x">Pixel Tolerance X:</label>
                <input type="number" id="pixel_tolerance_x" name="pixel_tolerance_x" value="20" min="0">
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="sttn_skip_detection" name="sttn_skip_detection" checked> STTN Skip Detection
                </label>
            </div>
            
            <div class="form-group">
                <label for="sttn_neighbor_stride">STTN Neighbor Stride:</label>
                <input type="number" id="sttn_neighbor_stride" name="sttn_neighbor_stride" value="5" min="1">
            </div>
            
            <div class="form-group">
                <label for="sttn_reference_length">STTN Reference Length:</label>
                <input type="number" id="sttn_reference_length" name="sttn_reference_length" value="10" min="1">
            </div>
            
            <div class="form-group">
                <label for="sttn_max_load_num">STTN Max Load Num:</label>
                <input type="number" id="sttn_max_load_num" name="sttn_max_load_num" value="50" min="1">
            </div>
            
            <div class="form-group">
                <label for="propainter_max_load_num">PROPAINTER Max Load Num:</label>
                <input type="number" id="propainter_max_load_num" name="propainter_max_load_num" value="70" min="1">
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="lama_super_fast" name="lama_super_fast"> LAMA Super Fast
                </label>
            </div>
            
            <button type="submit" id="submitBtn">Process Video</button>
        </form>
        
        <div id="statusSection" class="status hidden">
            <h2>Processing Status</h2>
            <p id="statusText">Processing...</p>
            <div class="progress-bar">
                <div id="progressBar" class="progress"></div>
            </div>
            <div id="errorDetails" class="error-details hidden"></div>
            <a id="downloadLink" class="download-link hidden" target="_blank">Download Result</a>
        </div>
    </div>
    
    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const statusSection = document.getElementById('statusSection');
            const statusText = document.getElementById('statusText');
            const progressBar = document.getElementById('progressBar');
            const downloadLink = document.getElementById('downloadLink');
            const submitBtn = document.getElementById('submitBtn');
            const errorDetails = document.getElementById('errorDetails');
            
            // Hide error details
            errorDetails.classList.add('hidden');
            
            // Disable submit button during processing
            submitBtn.disabled = true;
            
            statusSection.classList.remove('hidden');
            statusSection.classList.remove('status-completed');
            statusSection.classList.remove('status-error');
            statusSection.classList.add('status-processing');
            statusText.textContent = 'Uploading and processing...';
            progressBar.style.width = '0%';
            downloadLink.classList.add('hidden');
            
            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.detail || 'Unknown error occurred');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Response data:', data);
                if (data.task_id) {
                    // Start checking status
                    checkStatus(data.task_id);
                } else {
                    throw new Error('No task ID received: ' + JSON.stringify(data));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                statusSection.classList.remove('status-processing');
                statusSection.classList.add('status-error');
                statusText.innerHTML = 'Error: ' + error.message.replace(/\\n/g, '<br>');
                
                // Show error details
                if (error.message.includes('Traceback')) {
                    errorDetails.textContent = error.message;
                    errorDetails.classList.remove('hidden');
                }
                
                submitBtn.disabled = false;
            });
        });
        
        function checkStatus(taskId) {
            const statusText = document.getElementById('statusText');
            const progressBar = document.getElementById('progressBar');
            const statusSection = document.getElementById('statusSection');
            const downloadLink = document.getElementById('downloadLink');
            const submitBtn = document.getElementById('submitBtn');
            const errorDetails = document.getElementById('errorDetails');
            
            fetch(`/status/${taskId}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.detail || 'Unknown error occurred');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Status data:', data);
                statusText.textContent = `Status: ${data.status}`;
                if (data.progress !== undefined) {
                    progressBar.style.width = data.progress + '%';
                }
                
                if (data.status === 'completed') {
                    statusSection.classList.remove('status-processing');
                    statusSection.classList.add('status-completed');
                    statusText.textContent = 'Processing completed!';
                    downloadLink.classList.remove('hidden');
                    downloadLink.href = `/download/${taskId}`;
                    downloadLink.textContent = 'Download Result';
                    submitBtn.disabled = false;
                } else if (data.status === 'failed') {
                    statusSection.classList.remove('status-processing');
                    statusSection.classList.add('status-error');
                    const errorMsg = data.error || 'Unknown error';
                    statusText.textContent = 'Processing failed: ' + errorMsg;
                    
                    // Show error details if available
                    if (data.error && typeof data.error === 'string' && data.error.includes('\\n')) {
                        errorDetails.textContent = data.error;
                        errorDetails.classList.remove('hidden');
                    }
                    
                    submitBtn.disabled = false;
                } else {
                    // Continue checking
                    setTimeout(() => checkStatus(taskId), 2000);
                }
            })
            .catch(error => {
                console.error('Status check error:', error);
                statusSection.classList.remove('status-processing');
                statusSection.classList.add('status-error');
                statusText.textContent = 'Error checking status: ' + error.message;
                submitBtn.disabled = false;
            });
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=JSONResponse)
async def root():
    if not BACKEND_AVAILABLE:
        return {"message": "Video Subtitle Remover API - Backend not available", 
                "version": "1.1.1",
                "error": BACKEND_ERROR,
                "status": "degraded"}
    return {"message": "Video Subtitle Remover API", "version": config.VERSION}

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """
    Serve the web interface
    """
    return HTMLResponse(content=html_content, status_code=200)

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
            "progress": 0,
            "config": {
                "mode": mode_enum.value,
                "use_h264": use_h264,
                "threshold_height_width_difference": threshold_height_width_difference,
                "subtitle_area_deviation_pixel": subtitle_area_deviation_pixel,
                "threshold_height_difference": threshold_height_difference,
                "pixel_tolerance_y": pixel_tolerance_y,
                "pixel_tolerance_x": pixel_tolerance_x,
                "sttn_skip_detection": sttn_skip_detection,
                "sttn_neighbor_stride": sttn_neighbor_stride,
                "sttn_reference_length": sttn_reference_length,
                "sttn_max_load_num": sttn_max_load_num,
                "propainter_max_load_num": propainter_max_load_num,
                "lama_super_fast": lama_super_fast
            }
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
        # Get task config
        task_config = processing_tasks[task_id]["config"]
        
        # Save current config values to restore later
        original_mode = config.MODE
        original_use_h264 = config.USE_H264
        original_threshold_height_width_difference = config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE
        original_subtitle_area_deviation_pixel = config.SUBTITLE_AREA_DEVIATION_PIXEL
        original_threshold_height_difference = config.THRESHOLD_HEIGHT_DIFFERENCE
        original_pixel_tolerance_y = config.PIXEL_TOLERANCE_Y
        original_pixel_tolerance_x = config.PIXEL_TOLERANCE_X
        original_sttn_skip_detection = config.STTN_SKIP_DETECTION
        original_sttn_neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        original_sttn_reference_length = config.STTN_REFERENCE_LENGTH
        original_sttn_max_load_num = config.STTN_MAX_LOAD_NUM
        original_propainter_max_load_num = config.PROPAINTER_MAX_LOAD_NUM
        original_lama_super_fast = config.LAMA_SUPER_FAST
        
        # Apply config settings
        config.MODE = config.InpaintMode(task_config["mode"])
        config.USE_H264 = task_config["use_h264"]
        config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = task_config["threshold_height_width_difference"]
        config.SUBTITLE_AREA_DEVIATION_PIXEL = task_config["subtitle_area_deviation_pixel"]
        config.THRESHOLD_HEIGHT_DIFFERENCE = task_config["threshold_height_difference"]
        config.PIXEL_TOLERANCE_Y = task_config["pixel_tolerance_y"]
        config.PIXEL_TOLERANCE_X = task_config["pixel_tolerance_x"]
        config.STTN_SKIP_DETECTION = task_config["sttn_skip_detection"]
        config.STTN_NEIGHBOR_STRIDE = task_config["sttn_neighbor_stride"]
        config.STTN_REFERENCE_LENGTH = task_config["sttn_reference_length"]
        config.STTN_MAX_LOAD_NUM = task_config["sttn_max_load_num"]
        config.PROPAINTER_MAX_LOAD_NUM = task_config["propainter_max_load_num"]
        config.LAMA_SUPER_FAST = task_config["lama_super_fast"]
        
        # Create subtitle remover
        remover = SubtitleRemover(input_path)
        
        # Override the output path
        remover.video_out_name = output_path
        
        # Restore original config values
        config.MODE = original_mode
        config.USE_H264 = original_use_h264
        config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = original_threshold_height_width_difference
        config.SUBTITLE_AREA_DEVIATION_PIXEL = original_subtitle_area_deviation_pixel
        config.THRESHOLD_HEIGHT_DIFFERENCE = original_threshold_height_difference
        config.PIXEL_TOLERANCE_Y = original_pixel_tolerance_y
        config.PIXEL_TOLERANCE_X = original_pixel_tolerance_x
        config.STTN_SKIP_DETECTION = original_sttn_skip_detection
        config.STTN_NEIGHBOR_STRIDE = original_sttn_neighbor_stride
        config.STTN_REFERENCE_LENGTH = original_sttn_reference_length
        config.STTN_MAX_LOAD_NUM = original_sttn_max_load_num
        config.PROPAINTER_MAX_LOAD_NUM = original_propainter_max_load_num
        config.LAMA_SUPER_FAST = original_lama_super_fast
        
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