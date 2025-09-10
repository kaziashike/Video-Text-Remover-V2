# Video Subtitle Remover Docker Image

This repository provides a Docker image for the Video Subtitle Remover application, based on `docker.io/eritpchy/video-subtitle-remover:1.1.1-cuda11.8` but with your own code.

## Features

- Based on CUDA 11.8 optimized environment
- Includes all required dependencies (FastAPI, uvicorn, python-multipart)
- Ready for deployment on RunPod
- API access with automatic documentation

## Prerequisites

Before building the Docker image, you need to:

1. Download the required AI models and place them in the `backend/models` directory:
   - LAMA model in `backend/models/big-lama`
   - STTN model in `backend/models/sttn/infer_model.pth`
   - Video model in `backend/models/video/ProPainter.pth`
   - OCR detection model in `backend/models/V4/ch_det`

## Building the Docker Image

To build the Docker image, run:

```bash
docker build -t video-subtitle-remover .
```

## Running the Container

To run the container locally:

```bash
docker run -p 8000:8000 video-subtitle-remover
```

The API will be accessible at `http://localhost:8000` with documentation available at `http://localhost:8000/docs`.

## Deploying to RunPod

To deploy this image to RunPod:

1. Push your image to Docker Hub:
   ```bash
   docker tag video-subtitle-remover your-dockerhub-username/video-subtitle-remover
   docker push your-dockerhub-username/video-subtitle-remover
   ```

2. Create a new pod on RunPod using your custom Docker image.

3. Set the following environment variables in RunPod:
   - `PORT` = `8000`

4. Expose port 8000 when creating the pod.

## API Usage

Once the container is running, you can access the following endpoints:

- `POST /process` - Process a video to remove subtitles
- `GET /` - Web interface for uploading and processing videos

Check `http://localhost:8000/docs` for detailed API documentation.

## Environment Variables

- `PORT` - Port to run the API on (default: 8000)

## Notes

- The container requires significant GPU memory (recommended 16GB+ for best performance)
- Model files are not included in the repository and must be added separately
- For RunPod deployment, ensure you select a GPU-enabled pod template