import json
import re
import asyncio
import itertools
import numpy as np
import tiktoken
import logging
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions
from app.core.config import settings
from app.services.embedding import embeddings_model
def extract_json_from_string(s: str) -> dict:
    try:
        start_index = s.find('{')
        end_index = s.rfind('}') + 1
        json_str = s[start_index:end_index]
        return json.loads(json_str)
    except (ValueError, IndexError) as e:
        logging.error(f"Error extracting JSON: {e}\nOriginal string: {s}")
        return None
def extract_json_array_from_string(s: str) -> list:
    try:
        start_index = s.find('[')
        end_index = s.rfind(']') + 1
        if start_index == -1 or end_index == 0:
            return None
        json_str = s[start_index:end_index]
        return json.loads(json_str)
    except (ValueError, IndexError) as e:
        logging.error(f"Error extracting JSON array: {e}\nOriginal string: {s}")
        return None
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
def _create_wait_strategy(retry_state):
    exception = retry_state.outcome.exception()
    if exception and hasattr(exception, 'response') and hasattr(exception.response, 'headers') and 'Retry-After' in exception.response.headers:
        retry_after_header = exception.response.headers['Retry-After']
        if retry_after_header.isdigit():
            retry_after = int(retry_after_header)
        else:
            try:
                retry_after_dt = parsedate_to_datetime(retry_after_header)
                retry_after = (retry_after_dt - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                retry_after = 0
        if retry_after > 0:
            logging.warning(json.dumps({
                "event": "rate_limit_retry",
                "retry_after": retry_after
            }))
            return retry_after
    return wait_exponential(multiplier=1, min=2, max=60)(retry_state)
def log_before_retry(retry_state):
    logging.info(json.dumps({
        "event": "retry_attempt",
        "attempt": retry_state.attempt_number,
        "exception": str(retry_state.outcome.exception())
    }))
async def _process_sub_batch(llm, sub_questions, faiss_index):
    retriever = faiss_index.as_retriever(search_kwargs={"k": 8})
    question_contexts = []
    for i, question in enumerate(sub_questions):
        top_chunks = await retriever.aget_relevant_documents(question)
        if not top_chunks:
            question_contexts.append(f"Question {i+1}: {question}\nContext for Question {i+1}:\nNo relevant context found.")
            continue
        query_keywords = set(re.findall(r'\b\w+\b', question.lower()))
        max_keyword_score = 0
        for chunk in top_chunks:
            chunk_keywords = set(re.findall(r'\b\w+\b', chunk.page_content.lower()))
            score = len(query_keywords.intersection(chunk_keywords))
            chunk.metadata["keyword_score"] = score
            if score > max_keyword_score:
                max_keyword_score = score
        if embeddings_model:
            query_embedding = embeddings_model.embed_query(question)
            chunk_contents = [chunk.page_content for chunk in top_chunks]
            chunk_embeddings = embeddings_model.embed_documents(chunk_contents)
            max_semantic_score = -1
            for j, chunk in enumerate(top_chunks):
                semantic_score = np.dot(query_embedding, chunk_embeddings[j])
                chunk.metadata["semantic_score"] = semantic_score
                if semantic_score > max_semantic_score:
                    max_semantic_score = semantic_score
        else:
            logging.warning("Embeddings model not initialized. Skipping semantic reranking.")
            for chunk in top_chunks:
                chunk.metadata["semantic_score"] = 0
        for chunk in top_chunks:
            normalized_keyword_score = chunk.metadata["keyword_score"] / (max_keyword_score if max_keyword_score > 0 else 1)
            normalized_semantic_score = (chunk.metadata["semantic_score"] + 1) / 2
            chunk.metadata["hybrid_score"] = normalized_keyword_score + normalized_semantic_score
        top_chunks.sort(key=lambda x: x.metadata["hybrid_score"], reverse=True)
        final_chunks_for_question = top_chunks[:5]
        context_for_question = "\n\n".join([doc.page_content for doc in final_chunks_for_question])
        question_contexts.append(f"Question {i+1}: {question}\nContext for Question {i+1}:\n{context_for_question}")
    combined_context_for_batch = "\n\n---\n\n".join(question_contexts)
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(combined_context_for_batch))
    logging.info(f"Number of tokens sent to LLM: {num_tokens}")
    prompt_template = """
You are a helpful and empathetic AI legal assistant. Your goal is to provide accurate, easy-to-understand answers to legal questions.

**Instructions:**
1.  **Analyze Each Question's Context:** For each "Question X" provided, carefully read its corresponding "Context for Question X".
2.  **Answer Each Question:**
    *   If the context directly answers the question, use that information.
    *   If the context is insufficient or irrelevant, use your general knowledge to provide a comprehensive answer.
3.   **Concise Answers:** Keep your answers brief and to the point. Do not provide long explanations unless necessary.
4.  **Yes/No Questions:** If the question can be answered with a "Yes" or "No," start your answer with "Yes," or "No," followed by a brief explanation.
5.  **Irrelevant Questions:** If a question is completely irrelevant to the provided context (e.g., a coding or miscellaneous question), you MUST state: 'The answer could not be found in the provided document.'
6.  **JSON Output:** Your final output must be a single JSON object with a key "answers", which is a list of strings. Each string is an answer to the corresponding question, in the same order as the input questions.
7. **THINK AGGRESSIVELY:** Use all the context provided to you. Do not skip any information that could be relevant to answering the questions. for eg: if a question asks about future investments, and the context mentions a company's future plans, use that information to answer the question.
8. **Follow The Steps Provided to Find the Answer:**  Some documents describe a multi-step reasoning process (like legal checklists, procedures, or decision trees). In such cases:
    * Simulate the entire step-by-step process described in the document to arrive at the answer.
    * Do the API Calls to the Links if provided to get the answer.
    * Don’t say “The answer is not in the document” unless there is truly no rule or process you can follow.
    * Apply mappings (e.g., "Action A → Rule B → Outcome C") using logic and reasoning, not literal extraction.
    
**Questions and Contexts:**
{combined_context_for_batch}

**Example JSON Output:**
```json
{{
  "answers": [
    "Answer to question 1 based on its context.",
    "Based on general legal principles, [your answer here].",
    "Yes, [Explanation based on context].",
    "No, [Explanation based on context].",
    "The answer could not be found in the provided context."
  ]
}}
```

**Your JSON output:**
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["combined_context_for_batch"],
    )
    chain = PROMPT | llm | StrOutputParser()
    llm_output = await chain.ainvoke({"combined_context_for_batch": combined_context_for_batch})
    json_response = extract_json_from_string(llm_output)
    if json_response and 'answers' in json_response and isinstance(json_response['answers'], list):
        return json_response['answers']
    else:
        logging.error(json.dumps({
            "event": "batch_json_parsing_error",
            "llm_output": llm_output,
        }))
        return ["Error processing sub-batch."] * len(sub_questions)
@retry(
    wait=_create_wait_strategy,
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    before_sleep=log_before_retry
)
def _build_llm_clients() -> List[GoogleGenerativeAI]:
    clients: List[GoogleGenerativeAI] = []
    def maybe_add(api_key: Optional[str], model: Optional[str]):
        if api_key:
            clients.append(
                GoogleGenerativeAI(
                    model=(model or settings.llm_model),
                    temperature=0.1,
                    google_api_key=api_key,
                )
            )
    if not settings.gemini_api_key:
        raise RuntimeError("gemini_api_key is required but not set.")
    maybe_add(
        settings.gemini_api_key,
        settings.llm_model
    )
    maybe_add(
        settings.gemini_api_key_2,
        getattr(settings, "llm_model_2", None)
    )
    maybe_add(
        settings.gemini_api_key_3,
        getattr(settings, "llm_model_3", None)
    )
    maybe_add(
        settings.gemini_api_key_4,
        getattr(settings, "llm_model_4", None)
    )
    return clients
async def ask_llm_batch(faiss_index: FAISS, questions: List[str]) -> List[str]:
    # llm1 = GoogleGenerativeAI(
    #     model=settings.llm_model,
    #     temperature=0.1,
    #     google_api_key=settings.gemini_api_key,
    # )
    # llm2 = GoogleGenerativeAI(
    #     model=settings.llm_model_2,
    #     temperature=0.1,
    #     google_api_key=settings.gemini_api_key_2,
    # )
    # llm3 = GoogleGenerativeAI(
    #     model=settings.llm_model_3,
    #     temperature=0.1,
    #     google_api_key=settings.gemini_api_key_3,
    # )
    # llm4 = GoogleGenerativeAI(
    #     model=settings.llm_model_4,
    #     temperature=0.1,
    #     google_api_key=settings.gemini_api_key_4,
    # )
    # llms = itertools.cycle([llm1, llm2, llm3, llm4])
    clients = _build_llm_clients()
    if not clients:
        raise RuntimeError("No LLM clients available. Provide at least gemini_api_key.")
    llms = itertools.cycle(clients)
    n_clients = max(1, len(clients))
    if len(questions) > n_clients:
        sub_batches = np.array_split(questions, n_clients)
        sub_batches = [list(sb) for sb in sub_batches]
    else:
        sub_batches = [questions]
    tasks = [_process_sub_batch(next(llms), sub_batch, faiss_index) for sub_batch in sub_batches]
    results_from_sub_batches = await asyncio.gather(*tasks)
    all_answers = [answer for sub_list in results_from_sub_batches for answer in sub_list]
    return all_answers