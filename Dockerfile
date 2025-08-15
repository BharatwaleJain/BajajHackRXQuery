FROM python:3.9-slim
ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/cache/sent
RUN mkdir -p /cache/hf /cache/sent /code/faiss_cache && \
    chmod -R 777 /cache /code/faiss_cache
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
RUN chmod -R 755 /code && chmod -R 777 /code/faiss_cache /cache
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]