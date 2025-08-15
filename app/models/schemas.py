from pydantic import BaseModel, Field
from typing import List, Dict, Any
class HackRXInput(BaseModel):
    documents: str
    questions: List[str]
class Source(BaseModel):
    chunks: List[str] = Field(description="The source text chunks used for the answer.")
    pages: List[int] = Field(None, description="The page numbers of the source chunks.")
    clauses: List[str] = Field(None, description="The specific clauses from the source document.")
class LLMResponse(BaseModel):
    question: str
    answer: str
    source: Source = Field(description="Source metadata for the answer.")
    confidence: float = Field(description="Confidence score of the answer (1-5).")
class HackRXOutput(BaseModel):
    answers: List[LLMResponse]