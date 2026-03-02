import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
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
    model="j-hartmann/emotion-english-distilroberta-base"
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
    
    return {
        "label": result["label"],
        "confidence": float(result["score"])
    }

def calculate_confidence(transcription, sentiment_score):
    length_score = min(len(transcription.split()) / 50, 1)
    final_score = (0.6 * sentiment_score) + (0.4 * length_score)
    
    return round(final_score * 100, 2)

def ask_question(question: str):

    start_time = time.time()

    try:
        audio_array = record_audio()

        if audio_array is None or len(audio_array) == 0:
            return {"error": "No audio recorded"}

        transcription = speech_to_text(audio_array)

        if not transcription:
            return {"error": "Speech could not be transcribed"}

        sentiment_result = analyze_sentiment(transcription)

        confidence_score = calculate_confidence(
            transcription,
            sentiment_result["confidence"]
        )

        end_time = time.time()

        return {
            "question": question,
            "transcription": transcription,
            "sentiment": sentiment_result["label"],
            "sentiment_confidence": sentiment_result["confidence"],
            "confidence_score": confidence_score,
            "processing_time_ms": round((end_time - start_time) * 1000, 2)
        }

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
    
if __name__ == "__main__":

    print("AI Interview Bot (Speech Version)")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Enter AI-generated question: ")

        if question.lower() == "exit":
            break

        result = ask_question(question)

        if result:
            print("\n===== RESULT =====")
            print("Transcription:", result["transcription"])
            print("Sentiment:", result["sentiment"])
            print("Sentiment Confidence:", result["sentiment_confidence"])
            print("Confidence Score:", result["confidence_score"])
            print("\n")