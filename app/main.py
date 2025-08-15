import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.api.v1 import endpoints
from app.core.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
app = FastAPI(title="Bajaj HackRX Query Retrieval System")
app.state.limiter = limiter
try:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except AttributeError:
    async def rate_limit_handler(request, exc):
        return JSONResponse(status_code=429, content={"detail": "Rate Limit Exceeded"})
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
@app.get("/")
def read_root():
    return JSONResponse(
        content={
            "message": "Bajaj HackRX Query Retrieval System",
            "version": "0.1.0",
            "endpoints": {
                "health": "GET /api/v1",
                "auth": "GET /api/v1/auth",
                "processing": "POST /api/v1/hackrx/run"
            },
            "features": [
                "Multi Format Ingestion",
                "Smart Chunking",
                "Hybrid Retrieval",
                "Disk Caching",
                "AI Powered Answering",
                "LLM Orchestration",
                "Bearer Authentication",
                "Observability",
                "Optional OCR"
            ]
        }
    )
app.include_router(endpoints.router, prefix="/api/v1")