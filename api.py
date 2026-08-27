import io
import base64
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from core.ela_engine import generate_ela
from core.laplacian import analyze_laplacian_noise
from core.metadata_scan import extract_metadata
from core.frequency_fft import analyze_frequency_domain
from core.ai_vision import load_ai_model, analyze_with_deep_learning
from core.qr_scanner import scan_payloads

app = FastAPI(title="Document Tamper Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_model = load_ai_model()

def image_to_base64(img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_single_image(image: Image.Image, filename: str = "document.jpg") -> dict:
    ela_heatmap, ela_score = generate_ela(image)
    lap_heatmap, lap_var, lap_score = analyze_laplacian_noise(image)
    fft_heatmap, fft_score = analyze_frequency_domain(image)
    metadata, flags, meta_score = extract_metadata(image)
    qr_data = scan_payloads(image)
    ai_label, ai_conf, ai_risk = analyze_with_deep_learning(image, ai_model)

    is_screenshot = (meta_score == 65.0) and (ela_score < 15.0)
    is_blurry = lap_var < 50.0

    if is_screenshot:
        master_risk = (0.20 * fft_score) + (0.80 * ai_risk)
        doc_type = "Digital Screenshot"
    else:
        master_risk = (0.10 * ela_score) + (0.10 * lap_score) + (0.15 * fft_score) + (0.25 * meta_score) + (0.40 * ai_risk)
        doc_type = "Camera Photograph"

    master_risk = round(master_risk, 1)

    if master_risk >= 45.0:
        verdict = "HIGH_RISK_FORGERY"
        analysis_text = "Deep learning models, Laplacian noise anomalies, or modified metadata indicate high tampering probability."
    elif master_risk >= 35.0:
        verdict = "MEDIUM_RISK_REVIEW"
        analysis_text = "Mixed forensic signals detected. Document may have undergone compression, re-saving, or subtle editing."
    else:
        verdict = "LOW_RISK_AUTHENTIC"
        analysis_text = "Pixel physics, metadata integrity, and neural network inference confirm authentic provenance."

    return {
        "filename": filename,
        "verdict": verdict,
        "master_risk_score": master_risk,
        "document_type": doc_type,
        "analysis_text": analysis_text,
        "quality_alerts": {
            "is_screenshot": is_screenshot,
            "is_blurry": is_blurry,
        },
        "scores": {
            "ela_score": ela_score,
            "laplacian_variance": lap_var,
            "laplacian_score": lap_score,
            "fft_score": fft_score,
            "metadata_score": meta_score,
            "ai_risk_score": ai_risk,
            "ai_confidence": ai_conf,
            "ai_classification": ai_label,
        },
        "metadata": {
            "flags": flags,
            "raw_exif": metadata,
        },
        "qr_payloads": qr_data,
        "heatmaps": {
            "ela_base64": image_to_base64(ela_heatmap),
            "laplacian_base64": image_to_base64(lap_heatmap),
            "fft_base64": image_to_base64(fft_heatmap),
        }
    }

@app.get("/api/v1/health")
async def health_check():
    return {"status": "online"}

@app.post("/api/v1/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return process_single_image(image, file.filename)

@app.post("/api/v1/analyze-batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if file.content_type.startswith("image/"):
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            results.append(process_single_image(image, file.filename))
    return {"total_processed": len(results), "documents": results}