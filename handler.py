import runpod, os, uuid, shutil, subprocess, requests, time, traceback
import cv2, numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def get_lama():
    if not hasattr(get_lama, '_instance'):
        print("Loading LaMa model...")
        from simple_lama_inpainting import SimpleLama
        get_lama._instance = SimpleLama()
    return get_lama._instance

def get_clipseg():
    if not hasattr(get_clipseg, '_instance'):
        print("Loading CLIPSeg model...")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        get_clipseg._instance = (processor, model)
    return get_clipseg._instance

def create_mask_from_text(img_bgr, prompt):
    if img_bgr is None: return None
    
    processor, model = get_clipseg()
    
    # 1. Convert OpenCV array to PIL Image for HuggingFace
    rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    inputs = processor(text=[prompt], images=[pil_image], padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    preds = outputs.logits.unsqueeze(1)
    mask = torch.sigmoid(preds[0][0]).numpy()
    
    h, w = img_bgr.shape[:2]
    mask = cv2.resize(mask, (w, h))
    
    binary_mask = (mask > 0.4).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
    
    return binary_mask

def process_image(input_path, output_path, prompt):
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "Corrupted image file. AI could not read it."}
        
    mask = create_mask_from_text(img, prompt)
    if mask is None or np.max(mask) == 0:
        return {"error": f"AI could not find '{prompt}' in the image."}
    
    # 2. Convert to strict PIL Images for LaMa Inpainting
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert('L')
    
    lama = get_lama()
    result_pil = lama(img_pil, mask_pil)
    
    # Safety Check
    if not isinstance(result_pil, Image.Image):
        return {"error": "AI model failed to generate a result."}
    
    # 3. Convert back to OpenCV format
    result_np = np.array(result_pil)
    cv2.imwrite(output_path, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
    return {"success": True}

def process_video(input_path, output_path, prompt):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(f"{tmp}/frames", exist_ok=True)
    os.makedirs(f"{tmp}/out", exist_ok=True)
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        success = cv2.imwrite(f"{tmp}/frames/{idx:06d}.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success: idx += 1
        else: break
    cap.release()
    
    if idx == 0: 
        shutil.rmtree(tmp, ignore_errors=True)
        return {"error": "Video codec not supported or corrupted file."}
    
    print("Scanning multiple frames for dynamic objects...")
    master_mask = None
    samples = [0, idx//4, idx//2, (idx*3)//4, idx-1]
    
    for s in samples:
        if s >= idx: continue
        frame = cv2.imread(f"{tmp}/frames/{s:06d}.jpg")
        if frame is None: continue 
        
        current_mask = create_mask_from_text(frame, prompt)
        if current_mask is not None and np.max(current_mask) > 0:
            if master_mask is None:
                master_mask = current_mask
            else:
                master_mask = cv2.bitwise_or(master_mask, current_mask)
        
    if master_mask is None or np.max(master_mask) == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"error": f"AI could not find '{prompt}' to remove."}

    master_mask_pil = Image.fromarray(master_mask).convert('L')
    lama = get_lama()
    
    for i in range(idx):
        frame = cv2.imread(f"{tmp}/frames/{i:06d}.jpg")
        if frame is None: 
            prev = cv2.imread(f"{tmp}/out/{i-1:06d}.jpg") if i > 0 else np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", prev)
            continue
            
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result_pil = lama(img_pil, master_mask_pil)
        
        # Verify AI actually returned a valid image
        if isinstance(result_pil, Image.Image):
            result_np = np.array(result_pil)
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", frame)
    
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

def send_webhook(webhook_url, payload, files=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if files:
                r = requests.post(webhook_url, files=files, data=payload, timeout=120)
            else:
                r = requests.post(webhook_url, json=payload, timeout=30)
            print(f"Webhook attempt {attempt+1}: {r.status_code}")
            return True
        except Exception as e:
            print(f"Webhook attempt {attempt+1} failed: {e}")
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
    prompt = input_data.get("prompt", "watermark text logo") 

    print(f"Job {job_id} started. Type: {file_type}, Prompt: {prompt}")

    if not file_url or not webhook_url:
        return {"error": "Missing file_url or webhook_url"}

    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp, exist_ok=True)
    ext = ".mp4" if file_type == "video" else ".jpg"
    input_path = os.path.join(tmp, f"input{ext}")
    output_path = os.path.join(tmp, f"result{ext}")

    try:
        print(f"Downloading from {file_url}")
        for attempt in range(3):
            try:
                r = requests.get(file_url, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:
                print(f"Download attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(2)
        
        with open(input_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {os.path.getsize(input_path)} bytes")

        if file_type == "image":
            res = process_image(input_path, output_path, prompt)
        else:
            res = process_video(input_path, output_path, prompt)

        if res.get("error"):
            print(f"Processing error: {res['error']}")
            send_webhook(webhook_url, {"job_id": job_id, "error": res["error"]})
            return res

        print(f"Uploading result to webhook")
        with open(output_path, "rb") as f:
            send_webhook(
                webhook_url,
                {"job_id": job_id, "file_type": file_type},
                files={"file": (f"result{ext}", f)}
            )
        print(f"Job {job_id} completed")
        return {"success": True}

    except Exception as e:
        err_msg = str(e)
        print(f"Job {job_id} crashed: {err_msg}")
        print(traceback.format_exc())
        try:
            send_webhook(webhook_url, {"job_id": job_id, "error": err_msg})
        except Exception:
            pass
        return {"error": err_msg}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

runpod.serverless.start({"handler": handler})
