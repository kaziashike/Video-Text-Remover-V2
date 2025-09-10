FROM docker.io/eritpchy/video-subtitle-remover:1.1.1-cuda11.8

# Remove the existing code but keep the environment
RUN rm -rf /vsr/*

# Copy the current repository code
COPY . /vsr

# Make scripts executable
RUN chmod +x /vsr/update_code.sh

# Install/upgrade only the required packages for FastAPI if missing
RUN pip install --no-cache-dir python-multipart fastapi uvicorn[standard]

# Set the working directory
WORKDIR /vsr

# Expose the port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application
CMD ["python", "app.py"]