from typing import Optional
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    gemini_api_key: str
    gemini_api_key_2: Optional[str] = None
    gemini_api_key_3: Optional[str] = None
    gemini_api_key_4: Optional[str] = None
    auth_token: str
    embedding_model: str = "sentence-transformers/static-retrieval-mrl-en-v1"
    tesseract_cmd: Optional[str] = None
    llm_model: str = "gemini-2.5-flash"
    llm_model_2: str = "gemini-2.5-flash"
    llm_model_3: str = "gemini-2.5-flash"
    llm_model_4: str = "gemini-2.5-flash"
    chunk_size: int = 2048
    chunk_overlap: int = 256    
    class Config:
        env_file = ".env"
settings = Settings()