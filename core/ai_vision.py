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
        return "NO_TOKEN", 0.0, 50.0

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    
    try:
        response = requests.post(HF_API_URL, headers=headers, data=img_byte_arr.getvalue())
        results = response.json()
        
        top_result = results[0]
        label = str(top_result.get('label', 'REAL')).upper()
        confidence = float(top_result.get('score', 0.85)) * 100.0
        
        risk_score = confidence if label == "FAKE" else 100.0 - confidence
            
        return label, round(confidence, 1), round(risk_score, 1)
        
    except Exception:
        return "API_ERROR", 0.0, 50.0