import os
import json
import google.generativeai as genai
from PIL import Image

# Initialize Gemini API using Render's environment variable
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def load_ai_model():
    """
    Mock function to replace the old Hugging Face pipeline loader.
    This prevents api.py from crashing on startup and uses 0 RAM!
    """
    return None

def analyze_with_deep_learning(image: Image.Image, ai_model=None):
    """
    Sends the PIL image to Google Gemini and returns the exact 
    variables api.py expects (label, confidence, risk_score).
    """
    if not api_key:
        print("Warning: GEMINI_API_KEY is missing.")
        return "NO_API_KEY", 0.0, 50.0

    try:
        # We use the flash model because it is extremely fast and multimodal
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        You are a digital forensics expert AI. Analyze this image for tampering, deepfakes, or digital manipulation.
        Look for inconsistent fonts, digital noise, or logical errors.
        
        Respond ONLY with a valid JSON object. Do not include markdown formatting or extra text.
        The JSON must have these exact keys:
        {"label": "FAKE" or "REAL", "confidence": <float between 0 and 100>, "risk_score": <float between 0 and 100>}
        """
        
        # Gemini natively accepts PIL Image objects (which api.py already provides!)
        response = model.generate_content([prompt, image])
        
        # Strip potential markdown formatting (e.g., ```json ... ```)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(clean_text)
        
        label = str(data.get("label", "REAL")).upper()
        confidence = float(data.get("confidence", 85.0))
        risk_score = float(data.get("risk_score", 15.0))
        
        return label, round(confidence, 1), round(risk_score, 1)

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        # Safe fallback so the rest of the forensic math in api.py still works
        return "API_ERROR", 0.0, 50.0