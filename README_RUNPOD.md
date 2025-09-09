# Video Subtitle Remover API for RUNPOD

This guide explains how to deploy and use the Video Subtitle Remover API on RUNPOD.

## Prerequisites

- A RUNPOD account
- A GPU-enabled pod (recommended: 16GB+ VRAM for best performance)

## Deployment

1. Create a new pod on RUNPOD with a GPU
2. Select a PyTorch-based template (e.g., "PyTorch 2.0.1 - Python 3.10 - CUDA 11.8")
3. In the "Docker Command" field, add the following:

```bash
git clone https://github.com/your-username/video-subtitle-remover.git && \
cd video-subtitle-remover && \
pip install -r requirements.txt && \
pip install fastapi uvicorn && \
python app.py
```

Note: Replace `your-username` with the actual GitHub username where the repository is hosted.

## API Endpoints

Once deployed, the API will be available on port 8000. The following endpoints are available:

### Root Endpoint
```
GET /
```
Returns basic information about the API.

### Health Check
```
GET /health
```
Returns the health status of the system, including CUDA availability and device information.

### Process Video
```
POST /process
```
Processes a video to remove subtitles. Parameters:

- `video` (file): The video file to process
- `mode` (string): Inpainting mode - "sttn", "lama", or "propainter" (default: "sttn")
- `use_h264` (boolean): Whether to use h264 encoding (default: true)
- `threshold_height_width_difference` (integer): Threshold for height/width difference (default: 10)
- `subtitle_area_deviation_pixel` (integer): Subtitle area deviation in pixels (default: 20)
- `threshold_height_difference` (integer): Threshold for height difference (default: 20)
- `pixel_tolerance_y` (integer): Pixel tolerance on Y axis (default: 20)
- `pixel_tolerance_x` (integer): Pixel tolerance on X axis (default: 20)
- `sttn_skip_detection` (boolean): Whether to skip detection in STTN mode (default: true)
- `sttn_neighbor_stride` (integer): STTN neighbor stride (default: 5)
- `sttn_reference_length` (integer): STTN reference length (default: 10)
- `sttn_max_load_num` (integer): STTN max load number (default: 50)
- `propainter_max_load_num` (integer): PROPAINTER max load number (default: 70)
- `lama_super_fast` (boolean): Whether to use LAMA super fast mode (default: false)

Returns a task ID that can be used to check status and download the result.

### Check Processing Status
```
GET /status/{task_id}
```
Returns the status of a processing task, including progress percentage.

### Download Processed Video
```
GET /download/{task_id}
```
Downloads the processed video once the task is completed.

## Usage Examples

### Using curl

1. Process a video:
```bash
curl -X POST "http://localhost:8000/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@/path/to/video.mp4" \
  -F "mode=sttn"
```

2. Check processing status:
```bash
curl -X GET "http://localhost:8000/status/TASK_ID"
```

3. Download processed video:
```bash
curl -X GET "http://localhost:8000/download/TASK_ID" -o output_video.mp4
```

### Using Python requests

```python
import requests

# Process a video
files = {"video": open("input_video.mp4", "rb")}
data = {"mode": "sttn"}
response = requests.post("http://localhost:8000/process", files=files, data=data)
task_info = response.json()
task_id = task_info["task_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/status/{task_id}")
status = status_response.json()
print(f"Progress: {status['progress']}%")

# Download result
download_response = requests.get(f"http://localhost:8000/download/{task_id}")
with open("output_video.mp4", "wb") as f:
    f.write(download_response.content)
```

## Model Selection Guide

1. **STTN** (default): Best for real-person videos, fastest processing speed, can skip subtitle detection
2. **LAMA**: Good for animated videos, medium processing speed, cannot skip subtitle detection
3. **PROPAINTER**: Requires large VRAM, slow processing speed, best for videos with violent motion

## Performance Tips

- For high VRAM GPUs (16GB+), increase `propainter_max_load_num` for better quality
- For lower VRAM GPUs, decrease `sttn_max_load_num` and `propainter_max_load_num` to avoid OOM errors
- STTN mode with `sttn_skip_detection=true` provides fastest processing but may miss some subtitles
- LAMA mode provides good balance between quality and speed for animations