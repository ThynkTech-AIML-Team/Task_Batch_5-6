from aiclient import generate_questions
from speech_module import ask_question, generate_summary
from database_module import init_db, save_response
from config import MIN_QUESTIONS
import uuid
import time


def start_interview():

    # Initialize database
    init_db()

    print("\n===== AI INTERVIEW SESSION STARTED =====\n")

    candidate_name = input("Enter Candidate Name: ")
    candidate_id = str(uuid.uuid4())[:8]

    domain = input("Enter interview domain: ")

    print("\nGenerating questions...\n")
    questions = generate_questions(domain)

    if len(questions) < MIN_QUESTIONS:
        print("Not enough questions generated.")
        return

    question_number = 1
    all_results = []   # <-- Collect results for summary

    for question in questions[:MIN_QUESTIONS]:

        print(f"\nQuestion {question_number}: {question}\n")

        result = ask_question(question)

        if "error" in result:
            print("Error:", result["error"])
            continue

        print("Transcription:", result["transcription"])
        print("Emotion:", result["sentiment"])
        print("Confidence Score:", result["confidence_score"])

        # Save to DB
        save_response(
            candidate_name,
            candidate_id,
            question,
            result["transcription"],
            result["sentiment_confidence"],
            result["sentiment"],
            result["confidence_score"]
        )

        # Store result for summary
        all_results.append(result)

        question_number += 1
        time.sleep(1)

    # ===== Generate Final Summary =====
    summary = generate_summary(all_results)

    print("\n===== INTERVIEW SUMMARY =====")
    print("Total Questions:", summary["total_questions"])
    print("Average Confidence:", summary["average_confidence"])
    print("Dominant Emotion:", summary["dominant_emotion"])
    print("Final Verdict:", summary["verdict"])

    print("\n===== INTERVIEW COMPLETED =====")
    print("All responses saved to database.\n")