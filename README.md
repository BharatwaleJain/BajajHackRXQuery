# Bajaj HackRX Query Retrieval System

An API backend designed for intelligent document processing and question answering, optimized for insurance policy documents

Built with FastAPI, it leverages semantic search via sentence-transformer embeddings and Google Gemini to deliver precise, context-aware answers

Supports multi-format ingestion, hybrid retrieval, caching, resilient LLM orchestration and simple bearer authentication

---

## Features

- **Multi Format Ingestion**: Process PDF, DOCX, EML, PPTX, XLSX, PNG/JPG, HTML and raw website URLs
- **Smart Chunking**: Configurable chunk size/overlap with domain-aware section header hints
- **Hybrid Retrieval**: FAISS HNSW vector index combined with keyword and semantic re-ranking
- **Disk Caching**: FAISS indices cached under `faiss_cache/` per document base URL for fast subsequent queries
- **AI Powered Answering**: Answer questions using Google Gemini 2.5 Flash for contextual responses
- **LLM Orchestration**: Round-robin across up to 4 Gemini API keys with exponential backoff honoring Retry-After
- **Bearer Authentication**: Simple HTTP bearer token guard on protected endpoints
- **Observability**: Structured logs to `stdout`
- **Optional OCR**: Image and PPTX image OCR via Tesseract if valid `TESSERACT_CMD`

---

## Setup Instructions

### 0. System Requirements
- [Python 3.9+](https://www.python.org/downloads)
- [Tesseract OCR Binary](https://github.com/UB-Mannheim/tesseract/wiki) for OCR on Images

### 1. Clone the Repository
```bash
git clone https://github.com/BharatwaleJain/BajajHackRXQuery.git
cd BajajHackRXQuery
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory
```env
GEMINI_API_KEY=your_gemini_api_key (primary)
GEMINI_API_KEY_2=your_gemini_api_key_2 (optional)
GEMINI_API_KEY_3=your_gemini_api_key_3 (optional)
GEMINI_API_KEY_4=your_gemini_api_key_4 (optional)
AUTH_TOKEN=your_hackrx_bearer_token
TESSERACT_CMD=your_tesseract_path (optional)
```

### 4. Run App Locally
```bash
uvicorn app.main:app --reload
```
The server listens by default at `http://localhost:8000` or `http://127.0.0.1:8000`

---

## API Endpoints

#### API Service Status
```http
GET /api/v1
```

**Response:**
```json
{
    "status": "ok",
    "message": "API Running Successfully"
}
```

### Authentication Check
```http
GET /api/v1/auth
```

**Headers:**
```http
Authorization: Bearer your_token_here
```

**Response:**
```json
{
    "status": "ok",
    "message": "Authentication Successful"
}
```

#### Main Processing
```http
POST /api/v1/hackrx/run
```

**Request Body:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
```

**Headers:**
```http
Authorization: Bearer your_token_here
Content-Type: application/json
```

**Response:**
```json
{
    "answers": [
        "The grace period for premium payment under the National Parivar Mediclaim Plus Policy is 30 days.",
        "The waiting period for pre-existing diseases (PED) to be covered is 36 months of continuous coverage after the date of inception of the first policy.",
        "Yes, the policy covers maternity expenses, including pre-natal and post-natal hospitalization and new born baby vaccination. The female insured person must have been continuously covered for at least 24 months, and coverage is limited to two deliveries or terminations. The waiting period may be waived only in case of delivery, miscarriage or abortion induced by accident.",
        "The waiting period for cataract surgery is two years of continuous coverage after the date of inception of the first policy.",
        "Yes, medical expenses incurred for an organ donor's hospitalization for harvesting the organ donated to an insured person are covered, provided the organ donation conforms to the Transplantation of Human Organs Act 1994, the organ is used for an insured person medically advised to undergo transplant, the donor is an in-patient, and the claim for the recipient is admitted under the In-patient Treatment Section. Exclusions apply to pre/post-hospitalization expenses of the donor, acquisition costs, experimental treatments, complications post-harvesting, and organ transportation/preservation.",
        "A flat 5% No Claim Discount (NCD) is offered on the base premium upon renewal of one-year policies if no claims were reported. For policies exceeding one year, the NCD from each claim-free year is aggregated, not exceeding 5% of the total base premium for the policy term.",
        "Yes, expenses for health check-ups are reimbursed at the end of a block of two continuous policy years, provided the policy has been continuously renewed without a break. The expenses are subject to limits specified in the Table of Benefits (e.g., up to INR 5,000, INR 7,500, or INR 10,000 depending on the plan).",
        "A 'Hospital' is defined as any institution established for in-patient care and day care treatment of disease/injuries, registered with local authorities or complying with specific criteria: having qualified nursing staff round the clock, at least ten (or fifteen in towns with population over ten lacs) inpatient beds, qualified medical practitioners round the clock, a fully equipped operation theatre, and maintaining daily patient records accessible to the Company.",
        "The policy covers medical expenses for Inpatient Care treatment under Ayurveda, Yoga and Naturopathy, Unani, Siddha, and Homeopathy systems of medicine, up to the Sum Insured as specified in the Policy Schedule, when taken in an AYUSH Hospital.",
        "Yes, for Plan A, room charges are limited to up to 1% of the Sum Insured (SI) or actual, whichever is lower, and ICU charges are limited to up to 2% of the SI or actual, whichever is lower, per day per insured person. These limits do not apply if the treatment is undergone for a listed procedure in a Preferred Provider Network (PPN) as a package."
    ]
}
```

**Notes:**
- If documents is a website URL (no recognized file extension), the HTML content is fetched and processed without caching.
- If documents is a file URL, the file is downloaded and its index is cached in faiss_cache/ keyed by the base URL (query params ignored).
- Special question: including exactly what is my flight number?" invokes an internal multi-API flow and returns that answer in place of standard RAG results.

**Supported Inputs:**
- Files: .pdf, .docx, .eml, .pptx, .xlsx, .png, .jpeg, .jpg, .html
- Websites: any `http(s)://` URL without a supported file extension
- Unsupported file types like .zip or .bin are skipped with an error

---

## Project Layout

- app/main.py: FastAPI app factory, logging, limiter wiring, router include
- app/api/v1/endpoints.py: Health, auth check, RAG execution endpoint
- app/core/
  - config.py: Environment driven settings (keys, models, chunking, Tesseract path)
  - security.py: HTTP bearer validation
  - limiter.py: SlowAPI limiter setup
- app/models/schemas.py: Pydantic models for requests/responses
- app/services/
  - document_loader.py: Text extraction per file type & optional OCR
  - chunking.py: Token aware recursive chunking with header hints
  - embedding.py: Sentence transformers embeddings; FAISS HNSW build
  - llm.py: Gemini batching, round-robin keys, retry/backoff
  - query_processor.py: Orchestration, caching, website/file handling
- app/utils/downloader.py: Async downloads & HTML fetch helpers
- faiss_cache/: On-disk FAISS indices
- Dockerfile: Slim base, runs uvicorn
- railway.json: Docker build and start command
- requirements.txt: Dependencies
- setup.py: Packaging entry point

---

## Working Architecture

### Core Components

- Ingestion: `app/services/document_loader.py`  
  - Extracts text per file type (PDF, DOCX, PPTX, XLSX, EML, HTML, images)
  - For images and images embedded in PPTX, applies OCR via Tesseract when `TESSERACT_CMD` is valid otherwise OCR is skipped with a warning

- Chunking: `app/services/chunking.py`  
  - Uses a token-aware, domain-friendly splitter with configurable chunk_size/chunk_overlap and optional section-header detection to preserve structure

- Embeddings + Index: `app/services/embedding.py`  
  - Builds embeddings with `sentence-transformers/static-retrieval-mrl-en-v1` and constructs an HNSW FAISS index
  - Indices are cached under `faiss_cache/`

- Retrieval + Re-ranking: `app/services/llm.py`  
  - Performs top‑k retrieval from FAISS then hybrid re‑ranking combining keyword overlap and semantic similarity scores to choose the final context

- LLM: `app/services/llm.py`  
  - Queries Google Gemini 2.5 Flash with prompt templating and batching includes retry/backoff that respects Retry‑After and rotates across up to 4 API keys when provided

- Orchestration: `app/services/query_processor.py`  
  - Ties the pipeline together, manages caching and concurrency, routes website vs file flows and handles the special query "what is my flight number?"

### Query Retrieval Flow

```
Input URL
├─ Website (no recognized extension)
│  └─ Fetch HTML
│  └─ Extract plain text
└─ File (recognized extension)
   └─ Download
   └─ Type-specific extraction
   └─ OCR for images (if TESSERACT_CMD)
   └─ Chunk (token-aware, section-aware)
   └─ Embed
   └─ Build/Load FAISS HNSW (cached by base URL)
   └─ For each question
   │   └─ FAISS top‑k
   │   └─ Hybrid re‑rank (keywords + semantics)
   │   └─ Final context
   └─ Batch to Gemini (round‑robin, retry/backoff)
   └─ Parse JSON
   └─ Return {"answers": [string, ...]}
```

### Concurrency and Rate Management:
- Semaphore caps concurrent LLM calls at 5
- Backoff respects `Retry‑After` headers on rate limits
- Global rate limiter is initialized in the app
- Per‑route limits can be added via decorators if needed

### Caching
- FAISS indices are stored under `faiss_cache/`
- Cache key is the base URL (scheme, host & path; query string ignored)
- Delete the corresponding hashed directory to force a rebuild

### Known Limitations
- Declared Pydantic response model (HackRXOutput) is richer than the current runtime output which returns `{ "answers": [string, ...] }`
- First-time runs load the embedding model and build the FAISS index which can take noticeable time
- Large documents may require tuning `CHUNK_SIZE`/`CHUNK_OVERLAP` and container resources
- OCR requires `TESSERACT_CMD` to point to a valid Tesseract executable otherwise image OCR is skipped with a warning

---

## Contributors

- [Naincy Jain](https://www.linkedin.com/in/naincy-jain-38a20a283)
- [Aarjav Jain](https://www.linkedin.com/in/bharatwalejain)