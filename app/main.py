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
file_handler = logging.FileHandler("api.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
app = FastAPI()
app.state.limiter = limiter
try:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except AttributeError:
    async def rate_limit_handler(request, exc):
        return JSONResponse(status_code=429, content={"detail": "Rate Limit Exceeded"})
    app.add_exception_handler(RateLimitExceeded,rate_limit_handler)
@app.get("/")
def read_root():
    return JSONResponse(content={"status": "ok", "message": "API Running Successfully"})
app.include_router(endpoints.router, prefix="/api/v1")