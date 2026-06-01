import runpod, os, uuid, shutil, subprocess, requests, time
import cv2, numpy as np
from simple_lama_inpainting import SimpleLama
from paddleocr import PaddleOCR
import torch

lama = SimpleLama()
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=torch.cuda.is_available())

def detect_text_mask(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    res = ocr.ocr(rgb, cls=True)
    if res and res[0]:
        for line in res[0]:
            pts = np.array(line[0], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "Cannot read image"}
    mask = detect_text_mask(img)
    if np.max(mask) == 0:
        return {"error": "No watermark detected"}
    result = lama(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return {"success": True}

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(f"{tmp}/frames")
    os.makedirs(f"{tmp}/out")
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{tmp}/frames/{idx:06d}.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    cap.release()
    
    if idx == 0:
        return {"error": "Empty video"}
    
    first = cv2.imread(f"{tmp}/frames/000000.jpg")
    mask = detect_text_mask(first)
    
    for i in range(idx):
        frame = cv2.imread(f"{tmp}/frames/{i:06d}.jpg")
        if np.max(mask) > 0:
            result = lama(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mask)
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # Check if original has audio
    has_audio = False
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_path],
            capture_output=True, text=True, check=True
        )
        has_audio = 'audio' in probe.stdout
    except Exception:
        has_audio = False
    
    if has_audio:
        audio_path = os.path.join(tmp, "audio.aac")
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-vn', '-c:a', 'copy', audio_path],
            check=True, capture_output=True
        )
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps), '-i', f"{tmp}/out/%06d.jpg",
            '-i', audio_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k', '-shortest', output_path
        ], check=True, capture_output=True)
    else:
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps), '-i', f"{tmp}/out/%06d.jpg",
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
            output_path
        ], check=True, capture_output=True)
    
    shutil.rmtree(tmp, ignore_errors=True)
    return {"success": True}

def send_webhook(webhook_url, payload, files=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if files:
                requests.post(webhook_url, files=files, data=payload, timeout=120)
            else:
                requests.post(webhook_url, json=payload, timeout=30)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)
    return False

def handler(job):
    input_data = job.get("input", {})
    file_url = input_data.get("file_url")
    file_type = input_data.get("file_type", "image")
    webhook_url = input_data.get("webhook_url")
    job_id = input_data.get("job_id")

    if not file_url or not webhook_url:
        return {"error": "Missing file_url or webhook_url"}

    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp, exist_ok=True)
    ext = ".mp4" if file_type == "video" else ".jpg"
    input_path = os.path.join(tmp, f"input{ext}")
    output_path = os.path.join(tmp, f"result{ext}")

    try:
        # Download with retry
        for attempt in range(3):
            try:
                r = requests.get(file_url, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        
        with open(input_path, "wb") as f:
            f.write(r.content)

        if file_type == "image":
            res = process_image(input_path, output_path)
        else:
            res = process_video(input_path, output_path)

        if res.get("error"):
            send_webhook(webhook_url, {"job_id": job_id, "error": res["error"]})
            return res

        with open(output_path, "rb") as f:
            send_webhook(
                webhook_url,
                {"job_id": job_id, "file_type": file_type},
                files={"file": (f"result{ext}", f)}
            )
        return {"success": True}

    except Exception as e:
        send_webhook(webhook_url, {"job_id": job_id, "error": str(e)})
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

runpod.serverless.start({"handler": handler})
