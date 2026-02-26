import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
from transformers import pipeline

WHISPER_MODEL_SIZE = "base"
DEVICE = "cpu"
SAMPLE_RATE = 16000
RECORD_DURATION = 30

print("Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE)

print("Loading Sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def record_audio():
    print("Recording... Speak now!")
    audio = sd.rec(
        int(RECORD_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("Recording finished.\n")
    return audio.flatten()

def speech_to_text(audio_array):
    segments, _ = whisper_model.transcribe(audio_array)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return {"label": result["label"], "confidence": float(result["score"])}

def calculate_confidence(transcription, sentiment_score):
    length_score = min(len(transcription.split()) / 50, 1)
    final_score = (0.6 * sentiment_score) + (0.4 * length_score)
    return round(final_score * 100, 2)

# ✅ NEW: analyze directly from text (for React answer_text)
def analyze_text(text: str):
    start_time = time.time()

    cleaned = (text or "").strip()
    if not cleaned:
        return {"error": "Empty text"}

    sentiment_result = analyze_sentiment(cleaned)
    confidence_score = calculate_confidence(cleaned, sentiment_result["confidence"])

    end_time = time.time()

    return {
        "transcription": cleaned,
        "sentiment": sentiment_result["label"],
        "sentiment_confidence": sentiment_result["confidence"],
        "confidence_score": confidence_score,
        "processing_time_ms": round((end_time - start_time) * 1000, 2)
    }

def ask_question(question: str):
    # Keep for server-side mic mode (optional)
    try:
        audio_array = record_audio()

        if audio_array is None or len(audio_array) == 0:
            return {"error": "No audio recorded"}

        transcription = speech_to_text(audio_array)
        if not transcription:
            return {"error": "Speech could not be transcribed"}

        analysis = analyze_text(transcription)
        if "error" in analysis:
            return analysis

        return {"question": question, **analysis}

    except Exception as e:
        return {"error": str(e)}

def generate_summary(results):
    if not results:
        return {
            "total_questions": 0,
            "average_confidence": 0,
            "dominant_emotion": "N/A",
            "verdict": "No interview data"
        }

    total_confidence = 0
    emotion_count = {}

    for r in results:
        total_confidence += r["confidence_score"]
        emotion = r["sentiment"]
        emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

    average_confidence = total_confidence / len(results)
    dominant_emotion = max(emotion_count, key=emotion_count.get)

    if average_confidence >= 75:
        verdict = "Highly confident candidate"
    elif average_confidence >= 50:
        verdict = "Moderately confident candidate"
    else:
        verdict = "Needs improvement"

    return {
        "total_questions": len(results),
        "average_confidence": round(average_confidence, 2),
        "dominant_emotion": dominant_emotion,
        "verdict": verdict
    }