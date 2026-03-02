from fastapi import FastAPI

app = FastAPI(title="AI BOT API")

@app.get("/health")
def health():
    return {"status": "ok"}
