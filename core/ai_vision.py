import os
import io
import requests
from PIL import Image

HF_API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_ai_model():
    return None

def analyze_with_deep_learning(image: Image.Image, ai_model=None):
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable is not set.")
        return "NO_TOKEN", 0.0, 50.0

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        # FIX 1: Convert RGBA/transparent or palette images to standard RGB 
        # so they can be safely saved as JPEGs without crashing.
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        # FIX 2: Added a timeout so the server doesn't hang indefinitely
        response = requests.post(
            HF_API_URL, 
            headers=headers, 
            data=img_byte_arr.getvalue(), 
            timeout=20
        )
        
        if response.status_code != 200:
            print(f"HF API Error {response.status_code}: {response.text}")
            # Graceful fallback score so the batch keeps running even if HF throttles
            return "API_THROTTLED", 0.0, 30.0
            
        results = response.json()
        
        if isinstance(results, list) and len(results) > 0:
            top_result = results[0]
            label = str(top_result.get('label', 'REAL')).upper()
            confidence = float(top_result.get('score', 0.85)) * 100.0
            risk_score = confidence if label == "FAKE" else (100.0 - confidence)
            return label, round(confidence, 1), round(risk_score, 1)
        else:
            return "UNKNOWN", 0.0, 30.0
            
    except Exception as e:
        print(f"Inference request error (Handled gracefully): {str(e)}")
        # Graceful fallback fallback so batch processing doesn't throw a 500 error
        return "FALLBACK", 0.0, 30.0