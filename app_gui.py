from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import requests
import os
import uuid
from typing import Optional
import uvicorn
import json
import traceback

app = FastAPI(title="Video Subtitle Remover GUI",
              description="Frontend GUI for Video Subtitle Remover API",
              version="1.0.0")

# Store for processing tasks
gui_tasks = {}

# API Configuration
BACKEND_API_URL = "http://localhost:8000"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=generate_html_form(), status_code=200)

@app.post("/upload-and-process")
async def upload_and_process(
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
    Upload a video file and process it with the backend API
    """
    try:
        # Check if backend is reachable
        try:
            health_check = requests.get(f"{BACKEND_API_URL}/health", timeout=5)
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Backend API is not reachable. Please make sure the backend API is running on http://localhost:8000")
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="Backend API connection timed out. Please check if the backend API is running correctly.")
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Store task info
        gui_tasks[task_id] = {
            "status": "uploading",
            "filename": video.filename,
            "progress": 0
        }
        
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{task_id}_{video.filename}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Update task status
        gui_tasks[task_id]["status"] = "processing"
        gui_tasks[task_id]["temp_file_path"] = temp_file_path
        
        # Forward to backend API
        with open(temp_file_path, 'rb') as f:
            files = {'video': (video.filename, f, video.content_type)}
            data = {
                'mode': mode,
                'use_h264': use_h264,
                'threshold_height_width_difference': threshold_height_width_difference,
                'subtitle_area_deviation_pixel': subtitle_area_deviation_pixel,
                'threshold_height_difference': threshold_height_difference,
                'pixel_tolerance_y': pixel_tolerance_y,
                'pixel_tolerance_x': pixel_tolerance_x,
                'sttn_skip_detection': sttn_skip_detection,
                'sttn_neighbor_stride': sttn_neighbor_stride,
                'sttn_reference_length': sttn_reference_length,
                'sttn_max_load_num': sttn_max_load_num,
                'propainter_max_load_num': propainter_max_load_num,
                'lama_super_fast': lama_super_fast
            }
            
            response = requests.post(f"{BACKEND_API_URL}/process", files=files, data=data)
        
        if response.status_code == 200:
            try:
                backend_response = response.json()
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail=f"Invalid JSON response from backend: {response.text}")
            
            # Check if backend response has required fields
            if "task_id" not in backend_response:
                raise HTTPException(status_code=500, detail=f"Backend response missing task_id: {backend_response}")
            
            backend_task_id = backend_response["task_id"]
            
            # Update task with backend task ID
            gui_tasks[task_id]["backend_task_id"] = backend_task_id
            gui_tasks[task_id]["status"] = "submitted"
            gui_tasks[task_id]["backend_response"] = backend_response
            
            return {
                "gui_task_id": task_id,
                "message": "Video submitted for processing",
                "backend_response": backend_response
            }
        else:
            detail = f"Backend API error: {response.status_code} - {response.text}"
            raise HTTPException(status_code=response.status_code, detail=detail)
            
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Error processing video: {str(e)}\nTraceback: {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/task-status/{gui_task_id}")
async def get_task_status(gui_task_id: str):
    """
    Get the status of a processing task
    """
    if gui_task_id not in gui_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = gui_tasks[gui_task_id]
    
    # If we have a backend task ID, check the backend status
    if "backend_task_id" in task:
        try:
            backend_response = requests.get(f"{BACKEND_API_URL}/status/{task['backend_task_id']}")
            if backend_response.status_code == 200:
                backend_status = backend_response.json()
                task["status"] = backend_status["status"]
                task["progress"] = backend_status.get("progress", 0)
                
                return {
                    "gui_task_id": gui_task_id,
                    "status": task["status"],
                    "progress": task["progress"],
                    "backend_status": backend_status
                }
            else:
                return {
                    "gui_task_id": gui_task_id,
                    "status": task["status"],
                    "error": f"Backend API error: {backend_response.status_code} - {backend_response.text}"
                }
        except Exception as e:
            return {
                "gui_task_id": gui_task_id,
                "status": task["status"],
                "error": f"Could not fetch backend status: {str(e)}"
            }
    
    return {
        "gui_task_id": gui_task_id,
        "status": task["status"],
        "progress": task.get("progress", 0)
    }

@app.get("/download-result/{gui_task_id}")
async def download_result(gui_task_id: str):
    """
    Download the processed video
    """
    if gui_task_id not in gui_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = gui_tasks[gui_task_id]
    
    if "backend_task_id" not in task:
        raise HTTPException(status_code=400, detail="Task not yet submitted to backend")
    
    try:
        # Get backend status first
        backend_response = requests.get(f"{BACKEND_API_URL}/status/{task['backend_task_id']}")
        if backend_response.status_code != 200:
            raise HTTPException(status_code=backend_response.status_code, detail="Could not fetch backend status")
        
        backend_status = backend_response.json()
        if backend_status["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Backend task not completed. Status: {backend_status['status']}")
        
        # Download the result from backend
        download_response = requests.get(f"{BACKEND_API_URL}/download/{task['backend_task_id']}")
        if download_response.status_code != 200:
            raise HTTPException(status_code=download_response.status_code, detail="Could not download result from backend")
        
        # Save the result temporarily
        temp_dir = "temp_results"
        os.makedirs(temp_dir, exist_ok=True)
        result_file_path = os.path.join(temp_dir, f"{gui_task_id}_result.mp4")
        
        with open(result_file_path, "wb") as f:
            f.write(download_response.content)
        
        task["result_file_path"] = result_file_path
        
        return FileResponse(
            result_file_path,
            media_type='video/mp4',
            filename=f"processed_{task['filename']}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading result: {str(e)}")

def generate_html_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Subtitle Remover</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 40px; 
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Subtitle Remover</h1>
            
            <div class="instructions">
                <h3>Instructions:</h3>
                <p>1. Make sure the backend API is running on <code>http://localhost:8000</code></p>
                <p>2. Select a video file to process</p>
                <p>3. Adjust parameters as needed</p>
                <p>4. Click "Process Video" to start</p>
                <p><strong>Need help? Check the README.md file for detailed instructions.</strong></p>
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
                <button id="downloadButton" class="hidden">Download Result</button>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const statusSection = document.getElementById('statusSection');
                const statusText = document.getElementById('statusText');
                const progressBar = document.getElementById('progressBar');
                const downloadButton = document.getElementById('downloadButton');
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
                downloadButton.classList.add('hidden');
                
                fetch('/upload-and-process', {
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
                    if (data.gui_task_id) {
                        // Start checking status
                        checkStatus(data.gui_task_id);
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
                const downloadButton = document.getElementById('downloadButton');
                const submitBtn = document.getElementById('submitBtn');
                const errorDetails = document.getElementById('errorDetails');
                
                fetch(`/task-status/${taskId}`)
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
                        downloadButton.classList.remove('hidden');
                        downloadButton.onclick = () => {
                            window.location.href = `/download-result/${taskId}`;
                        };
                        submitBtn.disabled = false;
                    } else if (data.status === 'failed') {
                        statusSection.classList.remove('status-processing');
                        statusSection.classList.add('status-error');
                        const errorMsg = data.backend_status?.error || data.error || 'Unknown error';
                        statusText.textContent = 'Processing failed: ' + errorMsg;
                        
                        // Show error details if available
                        if (data.backend_status?.error && typeof data.backend_status.error === 'string' && data.backend_status.error.includes('\\n')) {
                            errorDetails.textContent = data.backend_status.error;
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
    return html_content

if __name__ == "__main__":
    print("Starting Video Subtitle Remover GUI...")
    print("Access the GUI at: http://localhost:8002")
    print("Make sure the API is running at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8002)