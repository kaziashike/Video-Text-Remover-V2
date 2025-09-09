# Video Subtitle Remover

Video-subtitle-remover (VSR) is an AI-based video hard subtitle removal software designed to intelligently identify and remove hard subtitles from videos through deep learning models while maintaining the original video resolution.

## Features

- **Lossless resolution subtitle removal**: Preserve original video resolution while removing hard subtitles
- **AI inpainting algorithm**: Use deep learning models for intelligent filling of subtitle areas (not pixel copying or mosaic)
- **Custom subtitle position**: Support specifying subtitle area to only remove content in that area
- **Full video automatic removal**: Automatically detect and remove all text in the entire video
- **Batch image watermark removal**: Support selecting multiple images for watermark text removal

## Core Features

- Support multiple AI algorithms (STTN, LAMA, PROPAINTER)
- Support GPU acceleration (CUDA/NVIDIA, DirectML/AMD/Intel)
- Provide both graphical interface (GUI) and command line (CLI) usage methods
- Support Docker deployment
- Provide FastAPI REST API interface for programmatic calls

## Installation Instructions

### Prerequisites

1. Python 3.7 or higher
2. pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install fastapi uvicorn python-multipart opencv-python torch torchvision paddlepaddle paddleocr
```

### Download Models

The application requires several AI models to function properly. Make sure the following models are in the correct directories:

- LAMA model in `backend/models/big-lama`
- STTN model in `backend/models/sttn/infer_model.pth`
- Video inpaint model in `backend/models/video/ProPainter.pth`
- OCR detection model in `backend/models/V4/ch_det`

## Running the Application

### Run API

There are several ways to run the API:

1. Using the startup script (recommended):
```bash
python start_api.py
```

2. Directly with Python:
```bash
python app.py
```

3. Using uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

After starting, the API will be available at:
- API Root: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Run GUI

```bash
python app_gui.py
```

The GUI will be available at http://localhost:8002

Note: The API must be running before starting the GUI.

### Run CLI

```bash
python ./backend/main.py
```

### Docker Run Command

```bash
docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda11.8
```

### Docker Run API

Build image:

```bash
docker build -t vsr-api .
```

Run using Docker Compose:

```bash
docker-compose up
```

API will be available at `http://localhost:8000`

## API Interface

### Process Video

```
POST /process
```

Parameters:
- `video` (file): The video file to process
- `mode` (string): Inpainting algorithm (sttn, lama, propainter) - Default: sttn
- `use_h264` (boolean): Whether to use H264 codec - Default: True
- `threshold_height_width_difference` (integer): Height/width difference detection threshold - Default: 10
- `subtitle_area_deviation_pixel` (integer): Subtitle area detection pixel deviation - Default: 20
- `threshold_height_difference` (integer): Height difference grouping threshold - Default: 20
- `pixel_tolerance_y` (integer): Vertical tracking pixel tolerance - Default: 20
- `pixel_tolerance_x` (integer): Horizontal tracking pixel tolerance - Default: 20
- `sttn_skip_detection` (boolean): Whether to skip subtitle detection in STTN mode - Default: True
- `sttn_neighbor_stride` (integer): STTN neighbor frame stride - Default: 5
- `sttn_reference_length` (integer): STTN reference frame length - Default: 10
- `sttn_max_load_num` (integer): STTN maximum load number - Default: 50
- `propainter_max_load_num` (integer): PROPAINTER maximum load number - Default: 70
- `lama_super_fast` (boolean): Whether to use LAMA super fast mode - Default: False

### Check Processing Status

```
GET /status/{task_id}
```

### Download Processed Video

```
GET /download/{task_id}
```

### Health Check

```
GET /health
```

## Troubleshooting

### Common Issues

1. **API not starting**: Make sure all dependencies are installed and models are available.

2. **CUDA/cuDNN errors**: Check your GPU driver and CUDA installation. You might need to:
   - Install the correct version of PyTorch for your CUDA version
   - Install the matching cuDNN version
   - Or switch to CPU mode by modifying the configuration

3. **Model loading errors**: Ensure all model files are properly downloaded and placed in the correct directories.

4. **Port conflicts**: If port 8000 or 8002 is already in use, modify the startup scripts to use different ports.

### Running in CPU Mode

If you encounter GPU-related issues, you can switch to CPU mode by modifying the configuration in `backend/config.py`:
```python
# Change this line:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# To force CPU usage:
device = torch.device("cpu")
```