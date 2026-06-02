import runpod, os, uuid, shutil, subprocess, requests, time, traceback
import cv2, numpy as np
import torch
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
    """Uses CLIPSeg to find the object described in the text prompt and creates a mask."""
    processor, model = get_clipseg()
    
    # Convert BGR (OpenCV) to RGB (HuggingFace)
    rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Process image and text
    inputs = processor(text=[prompt], images=[rgb_image], padding="max_length", return_tensors="pt")
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert prediction to an OpenCV-friendly mask
    preds = outputs.logits.unsqueeze(1)
    mask = torch.sigmoid(preds[0][0]).numpy()
    
    # Resize mask back to original image dimensions
    h, w = img_bgr.shape[:2]
    mask = cv2.resize(mask, (w, h))
    
    # Thresholding: Convert probabilities into a hard black/white mask
    # 0.4 is the confidence threshold. Lower it if it misses parts of the object.
    binary_mask = (mask > 0.4).astype(np.uint8) * 255
    
    # Dilate the mask slightly to ensure the edges of the object are fully covered
    kernel = np.ones((15, 15), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
    
    return binary_mask

def process_image(input_path, output_path, prompt):
    img = cv2.imread(input_path)
    
    mask = create_mask_from_text(img, prompt)
    
    if np.max(mask) == 0:
        return {"error": f"AI could not find '{prompt}' in the image."}
    
    lama = get_lama()
    result = lama(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return {"success": True}

def process_video(input_path, output_path, prompt):
    # Video processing remains the same, but passes the prompt to the mask generator
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
    
    # Analyze the middle frame to find the object
    mid_frame = cv2.imread(f"{tmp}/frames/{idx//2:06d}.jpg")
    master_mask = create_mask_from_text(mid_frame, prompt)
        
    if np.max(master_mask) == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"error": f"AI could not find '{prompt}' to remove."}

    lama = get_lama()
    for i in range(idx):
        frame = cv2.imread(f"{tmp}/frames/{i:06d}.jpg")
        result = lama(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), master_mask)
        cv2.imwrite(f"{tmp}/out/{i:06d}.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
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
    prompt = input_data.get("prompt", "watermark text logo") # Default fallback

    if not file_url or not webhook_url: return {"error": "Missing params"}

    tmp = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp, exist_ok=True)
    ext = ".mp4" if file_type == "video" else ".jpg"
    input_path, output_path = os.path.join(tmp, f"in{ext}"), os.path.join(tmp, f"out{ext}")

    try:
        r = requests.get(file_url, timeout=120)
        with open(input_path, "wb") as f: f.write(r.content)

        if file_type == "image": res = process_image(input_path, output_path, prompt)
        else: res = process_video(input_path, output_path, prompt)

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
