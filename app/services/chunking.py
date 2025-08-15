from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
import re
import logging
SECTION_HEADER_PATTERNS = [
    r'\n\nArticle \d+\.?',
    r'\n\nSection \d+\.?',
    r'\n\nClause \d+\.?',
    r'\n\nPolicy Terms',
    r'\n\nDefinitions',
    r'\n\nI\.\s',
    r'\n\nII\.\s',
    r'\n\nIII\.\s',
]
COMPILED_HEADER_REGEX = re.compile(f"({'|'.join(SECTION_HEADER_PATTERNS)})", flags=re.IGNORECASE)
async def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=lambda x: num_tokens_from_string(x, "cl100k_base")
    )
    major_sections = COMPILED_HEADER_REGEX.split(text)
    if len(major_sections) > 1:
        logging.info(f"Found {len(major_sections)} major sections. Chunking within sections.")
        final_chunks = []
        for section in major_sections:
            if section.strip():
                final_chunks.extend(splitter.split_text(section))
        return final_chunks
    else:
        logging.info("No section headers found. Using default chunking strategy.")
        return splitter.split_text(text)