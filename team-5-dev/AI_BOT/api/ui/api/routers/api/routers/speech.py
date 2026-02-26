from fastapi import APIRouter

router = APIRouter()

@router.post("/speech/stt")
def stt():
    return {"note": "Speech-to-text endpoint placeholder"}

@router.post("/speech/tts")
def tts():
    return {"note": "Text-to-speech endpoint placeholder"}
