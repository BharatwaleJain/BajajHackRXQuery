import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    embeddings_model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model
    )
    embedding_dim = len(embeddings_model.embed_query("test"))
except Exception as e:
    logger.error(f"Failed to initialize embeddings model: {e}")
    embeddings_model = None
    embedding_dim = 0
def embed_batch(batch: List[str]) -> List[List[float]]:
    if not embeddings_model:
        raise RuntimeError("Embeddings model is not initialized.")
    return embeddings_model.embed_documents(batch)
async def build_faiss_index(chunks: List[str]) -> FAISS:    
    if not embeddings_model:
        raise RuntimeError("Embeddings model is not initialized.")
    MIN_CHUNK_LENGTH = 10
    filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > MIN_CHUNK_LENGTH]
    if not filtered_chunks:
        raise ValueError("No valid chunks to process after filtering.")
    start_time_embedding = time.time()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=50) as executor:
        batch_size = 100
        futures = [
            loop.run_in_executor(
                executor,
                embed_batch,
                filtered_chunks[i:i + batch_size]
            )
            for i in range(0, len(filtered_chunks), batch_size)
        ]
        embedding_batches = await asyncio.gather(*futures)
    embeddings_list = [embedding for batch in embedding_batches for embedding in batch]
    end_time_embedding = time.time()
    logger.info(f"Embedding time: {end_time_embedding - start_time_embedding:.2f} seconds")
    if not embeddings_list:
        raise ValueError("No embeddings were generated.")
    embeddings_np = np.array(embeddings_list, dtype='float32')
    start_time_indexing = time.time()
    index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 40
    index.add(embeddings_np)
    documents = [Document(page_content=chunk) for chunk in filtered_chunks]
    docstore = InMemoryDocstore(
        {str(i): doc for i, doc in enumerate(documents)}
    )
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    faiss_index = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    end_time_indexing = time.time()
    logger.info(f"FAISS index building time with HNSW: {end_time_indexing - start_time_indexing:.2f} seconds")
    return faiss_index