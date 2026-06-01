FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download models so worker starts instantly
RUN python3 -c "\
import torch; \
from simple_lama_inpainting import SimpleLama; \
l = SimpleLama(); \
import numpy as np; \
l(np.zeros((256,256,3), dtype=np.uint8), np.zeros((256,256), dtype=np.uint8)); \
from paddleocr import PaddleOCR; \
o = PaddleOCR(lang='en', use_gpu=False); \
o.ocr(np.zeros((100,100,3), dtype=np.uint8)); \
print('Warmup done')"

COPY handler.py .

CMD ["python3", "handler.py"]
