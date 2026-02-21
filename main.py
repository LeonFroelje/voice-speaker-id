import os
from config import settings
import json
import tempfile
from typing import List
import numpy as np
import onnxruntime as ort
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic_settings import BaseSettings

app = FastAPI(title="Speaker Identification API")

# Lade ONNX Modell (Nutzt CUDA wenn verfügbar, sonst CPU)
try:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(settings.model_path, providers=providers)
except Exception as e:
    print(f"Warning: Could not load ONNX model at {settings.model_path}. Error: {e}")
    session = None


def load_db():
    if os.path.exists(settings.db_path):
        with open(settings.db_path, "r") as f:
            return {k: np.array(v) for k, v in json.load(f).items()}
    return {}


def save_db(db):
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    with open(settings.db_path, "w") as f:
        # Konvertiere Numpy Arrays zu Listen für JSON
        json.dump({k: v.tolist() for k, v in db.items()}, f)


def update_profile_ema(speaker_id: str, new_vectors_list: list):
    db = load_db()

    # 1. Average the incoming batch of files first
    new_centroid = np.mean(new_vectors_list, axis=0)

    if speaker_id in db:
        old_vector = db[speaker_id]  # Already a numpy array from load_db
        alpha = settings.ema_alpha

        # 2. Apply Exponential Moving Average
        updated_vector = ((1.0 - alpha) * old_vector) + (alpha * new_centroid)
    else:
        # First time registration
        updated_vector = new_centroid

    # 3. Normalize to keep Cosine Similarity math clean
    updated_vector = updated_vector / np.linalg.norm(updated_vector)

    # 4. Save back to DB (keep as numpy array, save_db handles the conversion)
    db[speaker_id] = updated_vector
    save_db(db)


def extract_embedding(audio_path: str) -> np.ndarray:
    if not session:
        raise HTTPException(status_code=500, detail="ONNX model not loaded.")

    # 1. Lade Audio und konvertiere zu 16kHz
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform)

    # 2. Mono erzwingen
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. WeSpeaker Feature Extraction (Kaldi Fbank)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        energy_floor=0.0,
        sample_frequency=16000.0,
    )

    # 4. Mean Normalization (CMN)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    # 5. Dimension anpassen für ONNX: [Batch, Frames, Mel]
    features = fbank.unsqueeze(0).numpy()

    # 6. Inference
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: features})[0]

    return embedding[0]


def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


@app.post("/enroll")
async def enroll(speaker_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Speichert den Stimmabdruck per EMA-Update. Akzeptiert auch mehrere Dateien."""
    new_embeddings = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            emb = extract_embedding(tmp_path)
            new_embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process a file for {speaker_id}: {e}")
        finally:
            os.remove(tmp_path)

    if not new_embeddings:
        raise HTTPException(
            status_code=400, detail="Keine gültigen Audiodateien gefunden."
        )

    # Check state before updating for the API response
    db = load_db()
    is_update = speaker_id in db

    # Update DB using EMA
    update_profile_ema(speaker_id, new_embeddings)

    return {
        "status": "success",
        "speaker_id": speaker_id,
        "is_update": is_update,
        "files_processed": len(new_embeddings),
    }


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """Identifiziert den Sprecher basierend auf der Audio-Datei."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        emb = extract_embedding(tmp_path)
        db = load_db()

        best_speaker = "Unknown"
        best_score = -1.0

        for speaker_id, ref_emb in db.items():
            score = cosine_similarity(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_speaker = speaker_id

        if best_score >= settings.threshold:
            return {"speaker_id": best_speaker, "score": best_score}
        else:
            return {"speaker_id": "Unknown", "score": best_score}
    finally:
        os.remove(tmp_path)


def start():
    """Entry point for the packaged application"""
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
