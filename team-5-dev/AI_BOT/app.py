from fastapi import FastAPI
from pydantic import BaseModel
from aiclient import generate_questions
from speech_module import analyze_text, generate_summary  # ✅ analyze_text added
from database_module import init_db, save_response, fetch_responses
from config import MIN_QUESTIONS
import uuid

app = FastAPI(title="AI Interview Backend")

# Initialize database
init_db()

# In-memory session store
sessions = {}

# -----------------------------
# Request Models
# -----------------------------
class InterviewStart(BaseModel):
    candidate_name: str
    domain: str


class QuestionRequest(BaseModel):
    candidate_id: str
    question: str
    answer_text: str  # ✅ NEW: text from React mic/textarea


# -----------------------------
# Routes
# -----------------------------
@app.post("/start-interview")
def start_interview(data: InterviewStart):
    candidate_id = str(uuid.uuid4())[:8]

    questions = generate_questions(data.domain)

    sessions[candidate_id] = {
        "candidate_name": data.candidate_name,
        "domain": data.domain,
        "questions": questions[:MIN_QUESTIONS],
        "results": []
    }

    return {
        "candidate_id": candidate_id,
        "questions": sessions[candidate_id]["questions"]
    }


@app.post("/ask-question")
def process_answer(data: QuestionRequest):
    if data.candidate_id not in sessions:
        return {"error": "Invalid candidate_id"}

    session = sessions[data.candidate_id]

    # ✅ Analyze the text sent from frontend
    analysis = analyze_text(data.answer_text)
    if "error" in analysis:
        return analysis

    # Build result in same shape your UI expects
    result = {
        "question": data.question,
        "transcription": analysis["transcription"],
        "sentiment_confidence": analysis["sentiment_confidence"],
        "sentiment": analysis["sentiment"],
        "confidence_score": analysis["confidence_score"],
        "processing_time_ms": analysis.get("processing_time_ms", None),
    }

    # Save to DB
    save_response(
        session["candidate_name"],
        data.candidate_id,
        data.question,
        result["transcription"],
        result["sentiment_confidence"],
        result["sentiment"],
        result["confidence_score"]
    )

    # Save in memory (for summary)
    session["results"].append(result)

    return result


@app.get("/summary/{candidate_id}")
def get_summary(candidate_id: str):
    if candidate_id not in sessions:
        return {"error": "Invalid candidate_id"}

    summary = generate_summary(sessions[candidate_id]["results"])
    return summary


@app.get("/results/{candidate_id}")
def get_results(candidate_id: str):
    data = fetch_responses(candidate_id)
    return {"database_records": data}