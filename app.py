from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from models import HackRXRequest, HackRXResponse, HealthResponse
from core.document import DocumentProcessor, get_document_stats
from core.rag import RAGService, create_rag_service
import logging
import uvicorn
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing services...")
    try:
        logger.info("Initializing DocumentProcessor...")
        app.state.document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=80)
        logger.info("DocumentProcessor initialized successfully")
        logger.info("Initializing RAG service...")
        app.state.rag_service = create_rag_service()
        logger.info("RAG service initialized successfully")
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Error during service initialization: {str(e)}")
        raise e
    yield
    logger.info("Application shutdown")
app = FastAPI(
    title="HackRX LLM Query-Retrieval System",
    description="Intelligent document processing and question answering system",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()
VALID_TOKEN = os.getenv("AUTH_TOKEN", " ")
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="healthy",
        message="HackRX API is running",
        version="1.0.0"
    )
@app.get("/simple-health")
async def simple_health_check():
    logger.info("Simple health check endpoint called")
    return {"status": "ok", "message": "API is running"}
@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    req: Request,
    token: str = Depends(verify_token)
):
    try:
        document_processor = req.app.state.document_processor
        rag_service = req.app.state.rag_service
    except AttributeError as e:
        logger.error(f"Services not properly initialized: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Services not properly initialized. Please restart the application."
        )
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        logger.info("Step 1: Processing document...")
        document_chunks = await document_processor.process_document_from_url(request.documents)
        stats = get_document_stats(document_chunks)
        logger.info(f"Document processing stats: {stats}")
        logger.info("Step 2: Answering questions using RAG...")
        answers = await rag_service.answer_multiple_questions(request.questions, document_chunks)
        logger.info("Request processed successfully")
        return HackRXResponse(answers=answers)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
@app.get("/")
async def root():
    return {
        "message": "HackRX LLM Query-Retrieval System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "main": "/api/v1/hackrx/run"
        },
        "features": [
            "PDF document processing",
            "Semantic search with embeddings",
            "Question answering with Gemini"
        ]
    }
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        log_level="info"
    )