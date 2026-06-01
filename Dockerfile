FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Warm-up: download AI models during build so first job is fast
RUN python3 -c "from simple_lama_inpainting import SimpleLama; l=SimpleLama(); import numpy as np; l(np.zeros((256,256,3),dtype=np.uint8), np.zeros((256,256),dtype=np.uint8))"
RUN python3 -c "from paddleocr import PaddleOCR; o=PaddleOCR(use_angle_cls=True,lang='en',show_log=False,use_gpu=False); o.ocr(np.zeros((100,100,3),dtype=np.uint8), cls=True)"

COPY handler.py .

CMD ["python3", "handler.py"]
