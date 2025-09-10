FROM python:3.12

RUN --mount=type=cache,target=/root/.cache,sharing=private \
    apt update && \
    apt install -y libgl1-mesa-glx && \
    true

ADD . /vsr
ARG CUDA_VERSION=11.8
ARG USE_DIRECTML=0

# If it's the CUDA version, perform CUDA-specific setup
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${USE_DIRECTML:-0}" != "1" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | tr -d '.') && \
        pip install -r /vsr/requirements.txt; \
    fi


ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/
WORKDIR /vsr
CMD ["python", "/vsr/backend/main.py"]