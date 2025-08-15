from fastapi import APIRouter, Depends, Request
from app.models.schemas import HackRXInput, HackRXOutput
from app.core.security import validate_token
from app.services.query_processor import process_query
import logging
logger = logging.getLogger(__name__)
router = APIRouter()
from fastapi.responses import JSONResponse
@router.get("/")
def read_root():
    return JSONResponse(content={"status": "ok", "message": "API Running Successfully"})
@router.get("/auth")
async def auth_check(_: bool = Depends(validate_token)):
    return JSONResponse(content={"status": "ok", "message": "Authentication Successful"})
@router.post("/hackrx/run", response_model=HackRXOutput)
async def run_query(request: Request, input_data: HackRXInput, _: bool = Depends(validate_token)):
    logger.info(f"Received request for URL: {input_data.documents}, Questions: {input_data.questions}")
    result = await process_query(input_data)
    return JSONResponse(content=result)