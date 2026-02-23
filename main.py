import os
import json
import tempfile
import asyncio
import logging
import boto3
import numpy as np
import onnxruntime as ort
import torchaudio
import aiomqtt
from config import settings

# --- Logging Setup ---
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SpeakerWorker")

# Load ONNX Model
try:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(settings.model_path, providers=providers)
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.error(f"Could not load ONNX model at {settings.model_path}. Error: {e}")
    session = None


# --- Core Logic Functions (Unchanged) ---
def load_db():
    if os.path.exists(settings.db_path):
        with open(settings.db_path, "r") as f:
            return {k: np.array(v) for k, v in json.load(f).items()}
    return {}


def save_db(db):
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    with open(settings.db_path, "w") as f:
        json.dump({k: v.tolist() for k, v in db.items()}, f)


def update_profile_ema(speaker_id: str, new_vectors_list: list):
    db = load_db()
    new_centroid = np.mean(new_vectors_list, axis=0)

    if speaker_id in db:
        old_vector = db[speaker_id]
        alpha = settings.ema_alpha
        updated_vector = ((1.0 - alpha) * old_vector) + (alpha * new_centroid)
    else:
        updated_vector = new_centroid

    updated_vector = updated_vector / np.linalg.norm(updated_vector)
    db[speaker_id] = updated_vector
    save_db(db)


def extract_embedding(audio_path: str) -> np.ndarray:
    if not session:
        raise RuntimeError("ONNX model not loaded.")

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        energy_floor=0.0,
        sample_frequency=16000.0,
    )
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    features = fbank.unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: features})[0]
    return embedding[0]


def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# --- S3 & Worker Logic ---
def download_audio_file(audio_url: str) -> str:
    object_key = audio_url.split("/")[-1]
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        s3_client.download_file(settings.s3_bucket, object_key, temp_audio.name)
        return temp_audio.name


def identify_speaker(audio_path: str) -> tuple[str, float]:
    emb = extract_embedding(audio_path)
    db = load_db()

    best_speaker, best_score = "Unknown", -1.0
    for speaker_id, ref_emb in db.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_speaker = speaker_id

    if best_score >= settings.threshold:
        return best_speaker, best_score
    return "Unknown", best_score


def enroll_speaker(speaker_id: str, audio_paths: list):
    new_embeddings = [extract_embedding(path) for path in audio_paths]
    if new_embeddings:
        update_profile_ema(speaker_id, new_embeddings)
        return len(new_embeddings)
    return 0


# --- MQTT Loop ---
async def main_async():
    try:
        async with aiomqtt.Client(
            settings.mqtt_host, port=settings.mqtt_port
        ) as client:
            logger.info(
                f"Connected to MQTT Broker at {settings.mqtt_host}:{settings.mqtt_port}"
            )

            await client.subscribe("voice/audio/recorded")
            await client.subscribe("voice/speaker/enroll")
            logger.info(
                "Listening for tasks on 'voice/audio/recorded' and 'voice/speaker/enroll'..."
            )

            async for message in client.messages:
                topic = message.topic.value
                payload = json.loads(message.payload.decode())

                # --- Handle Identification ---
                if topic == "voice/audio/recorded":
                    audio_url = payload.get("audio_url")
                    room = payload.get("room")

                    if not audio_url or not room:
                        continue

                    temp_audio = None
                    try:
                        temp_audio = await asyncio.to_thread(
                            download_audio_file, audio_url
                        )
                        speaker, score = await asyncio.to_thread(
                            identify_speaker, temp_audio
                        )

                        logger.info(
                            f"Identified {speaker} (Score: {score:.2f}) in room {room}"
                        )

                        await client.publish(
                            "voice/speaker/identified",
                            payload=json.dumps(
                                {"room": room, "speaker_id": speaker, "score": score}
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Identification failed: {e}")
                    finally:
                        if temp_audio and os.path.exists(temp_audio):
                            os.remove(temp_audio)

                # --- Handle Enrollment ---
                elif topic == "voice/speaker/enroll":
                    speaker_id = payload.get("speaker_id")
                    audio_urls = payload.get("audio_urls", [])  # Expects a list of URLs

                    if not speaker_id or not audio_urls:
                        continue

                    logger.info(f"Enrollment started for {speaker_id}...")
                    temp_files = []
                    try:
                        for url in audio_urls:
                            temp_files.append(
                                await asyncio.to_thread(download_audio_file, url)
                            )

                        processed_count = await asyncio.to_thread(
                            enroll_speaker, speaker_id, temp_files
                        )
                        logger.info(
                            f"Successfully enrolled {speaker_id} using {processed_count} files."
                        )

                        await client.publish(
                            "voice/speaker/enrolled",
                            payload=json.dumps(
                                {
                                    "speaker_id": speaker_id,
                                    "status": "success",
                                    "files_processed": processed_count,
                                }
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Enrollment failed: {e}")
                    finally:
                        for f in temp_files:
                            if os.path.exists(f):
                                os.remove(f)

    except aiomqtt.MqttError as error:
        logger.error(f"MQTT Error: {error}")
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")


def main():
    """Synchronous wrapper for the setuptools entry point."""
    import asyncio

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
