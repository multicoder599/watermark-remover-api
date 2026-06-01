import runpod, os, uuid, shutil, subprocess, requests
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

def position_mask(h, w, position, custom):
    pm = np.zeros((h, w), dtype=np.uint8)
    if not position or position == 'auto':
        return pm
    positions = {
        'top-left':     (0,       0,       w//4, h//8),
        'top-right':    (w*3//4, 0,       w//4, h//8),
        'bottom-left':  (0,       h*7//8,  w//4, h//8),
        'bottom-right': (w*3//4, h*7//8,  w//4, h//8),
        'center':       (w*3//8, h*3//8,  w//4, h//4),
        'bottom':       (0,       h*7//8,  w,    h//8),
    }
    if position == 'custom' and custom:
        parts = custom.split()
        if len(parts) == 4:
            x, y, mw, mh = map(int, parts)
            pm[y:y+mh, x:x+mw] = 255
    elif position in positions:
        x, y, mw, mh = positions[position]
        pm[y:y+mh, x:x+mw] = 255
    return pm

def process_image(input_path, output_path, position, custom):
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "Cannot read image"}
    text_mask = detect_text_mask(img)
    h, w = img.shape[:2]
    pm = position_mask(h, w, position, custom)
    mask = cv2.bitwise_or(text_mask, pm)
    if np.max(mask) == 0:
        return {"error": "No watermark detected"}
    result = lama(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return {"success": True}

def process_video(input_path, output_path, position, custom):
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
        cv2.imwrite(f"{tmp}/frames/{idx:06d}.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        idx += 1
    cap.release()
    if idx == 0:
        return {"error": "Empty video"}
    first = cv2.imread(f"{tmp}/frames/000000.jpg")
    text_mask = detect_text_mask(first)
    pm = position_mask(h, w, position, custom)
    mask = cv2.bitwise_or(text_mask, pm)
    for i in range(idx):
        frame = cv2.imread(f"{tmp}/frames/{i:06d}.jpg")
        if np.max(mask) > 0:
            result = lama(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mask)
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps), '-i', f"{tmp}/out/%06d.jpg",
        '-i', input_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
        '-c:a', 'copy', '-shortest', output_path
    ], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"success": True}

def handler(job):
    input_data = job.get("input", {})
    file_url = input_data.get("file_url")
    file_type = input_data.get("file_type", "image")
    position = input_data.get("position")
    custom = input_data.get("custom_coords")
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
        # Download
        r = requests.get(file_url, timeout=120)
        with open(input_path, "wb") as f:
            f.write(r.content)

        # Process
        if file_type == "image":
            res = process_image(input_path, output_path, position, custom)
        else:
            res = process_video(input_path, output_path, position, custom)

        if res.get("error"):
            requests.post(webhook_url, json={"job_id": job_id, "error": res["error"]}, timeout=30)
            return res

        # Send result back to your bot
        with open(output_path, "rb") as f:
            requests.post(
                webhook_url,
                files={"file": (f"result{ext}", f)},
                data={"job_id": job_id, "file_type": file_type},
                timeout=120
            )
        return {"success": True}

    except Exception as e:
        requests.post(webhook_url, json={"job_id": job_id, "error": str(e)}, timeout=30)
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

runpod.serverless.start({"handler": handler})