from fastapi import APIRouter

router = APIRouter()

@router.post("/sentiment")
def sentiment(payload: dict):
    text = payload.get("text", "")
    return {"received_text": text, "note": "Teammate sentiment function not connected yet"}
