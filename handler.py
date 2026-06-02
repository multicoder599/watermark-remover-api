import runpod, os, uuid, shutil, subprocess, requests, time, traceback
import cv2, numpy as np

def get_lama():
    if not hasattr(get_lama, '_instance'):
        print("Loading LaMa model...")
        from simple_lama_inpainting import SimpleLama
        get_lama._instance = SimpleLama()
    return get_lama._instance

def get_ocr():
    if not hasattr(get_ocr, '_instance'):
        print("Loading PaddleOCR model...")
        from paddleocr import PaddleOCR
        get_ocr._instance = PaddleOCR(lang='en', use_gpu=False)
    return get_ocr._instance

def detect_text_mask(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        ocr = get_ocr()
        res = ocr.ocr(rgb)
        if res and res[0]:
            for line in res[0]:
                pts = np.array(line[0], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            # Aggressive expansion to cover logos next to text (like TikTok icons)
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
    except Exception as e:
        print(f"OCR error: {e}")
    return mask

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    mask = detect_text_mask(img)
    if np.max(mask) == 0:
        return {"error": "AI could not detect any readable watermark in this image."}
    
    lama = get_lama()
    result = lama(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return {"success": True}

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(f"{tmp}/frames")
    os.makedirs(f"{tmp}/out")
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f"{tmp}/frames/{idx:06d}.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    cap.release()
    
    if idx == 0: return {"error": "Empty video"}
    
    # SMARTER AUTO-DETECT: Sample 5 different points in the video
    print("Scanning multiple frames for dynamic watermarks...")
    master_mask = None
    samples = [0, idx//4, idx//2, (idx*3)//4, idx-1]
    
    for s in samples:
        if s >= idx: continue
        frame = cv2.imread(f"{tmp}/frames/{s:06d}.jpg")
        current_mask = detect_text_mask(frame)
        if np.max(current_mask) > 0:
            if master_mask is None:
                master_mask = current_mask
            else:
                # Combine masks so jumping watermarks are fully covered
                master_mask = cv2.bitwise_or(master_mask, current_mask)
    
    if master_mask is None or np.max(master_mask) == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"error": "AI could not auto-detect any watermark in this video."}

    lama = get_lama()
    for i in range(idx):
        frame = cv2.imread(f"{tmp}/frames/{i:06d}.jpg")
        result = lama(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), master_mask)
        cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # Reattach Audio
    has_audio = False
    try:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_path], capture_output=True, text=True)
        has_audio = 'audio' in probe.stdout
    except: pass
    
    if has_audio:
        audio_path = os.path.join(tmp, "audio.aac")
        subprocess.run(['ffmpeg', '-y', '-i', input_path, '-vn', '-c:a', 'copy', audio_path], capture_output=True)
        subprocess.run(['ffmpeg', '-y', '-framerate', str(fps), '-i', f"{tmp}/out/%06d.jpg", '-i', audio_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast', '-c:a', 'aac', '-b:a', '128k', '-shortest', output_path], capture_output=True)
    else:
        subprocess.run(['ffmpeg', '-y', '-framerate', str(fps), '-i', f"{tmp}/out/%06d.jpg", '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast', output_path], capture_output=True)
    
    shutil.rmtree(tmp, ignore_errors=True)
    return {"success": True}

def send_webhook(webhook_url, payload, files=None):
    try:
        if files: requests.post(webhook_url, files=files, data=payload, timeout=120)
        else: requests.post(webhook_url, json=payload, timeout=30)
    except: pass

def handler(job):
    input_data = job.get("input", {})
    file_url = input_data.get("file_url")
    file_type = input_data.get("file_type", "image")
    webhook_url = input_data.get("webhook_url")
    job_id = input_data.get("job_id")

    if not file_url or not webhook_url: return {"error": "Missing params"}

    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp, exist_ok=True)
    ext = ".mp4" if file_type == "video" else ".jpg"
    input_path, output_path = os.path.join(tmp, f"in{ext}"), os.path.join(tmp, f"out{ext}")

    try:
        r = requests.get(file_url, timeout=120)
        with open(input_path, "wb") as f: f.write(r.content)

        if file_type == "image": res = process_image(input_path, output_path)
        else: res = process_video(input_path, output_path)

        if res.get("error"):
            send_webhook(webhook_url, {"job_id": job_id, "error": res["error"]})
            return res

        with open(output_path, "rb") as f:
            send_webhook(webhook_url, {"job_id": job_id, "file_type": file_type}, files={"file": (f"result{ext}", f)})
        return {"success": True}
    except Exception as e:
        err = str(e)
        send_webhook(webhook_url, {"job_id": job_id, "error": err})
        return {"error": err}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

runpod.serverless.start({"handler": handler})
