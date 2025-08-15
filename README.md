# Bajaj HackRX LLM Query Retrieval System
An backend API designed for intelligent document processing and question answering, optimized for insurance policy documents
Built with FastAPI, it leverages semantic search via embeddings and Google's Gemini LLM to deliver precise context-aware answers

---

## Features

- **PDF Document Processing**: Download and extract text directly from PDF URL
- **Intelligent Text Chunking**: Split document into overlapping chunks for better contextual understanding
- **Semantic Search**: Utilize sentence transformers embeddings and FAISS for efficient document retrieval
- **AI Powered Answering**: Answer questions using Google Gemini 1.5 Flash for contextual responses
- **Answer Post Processing**: Intelligent cleaning and formatting of AI-generated response
- **Secure API Access**: Bearer token authentication protects the API endpoints
- **Insurance Domain Optimized**: Specifically tuned for insurance policy document analysis

---

## Technology Stack

- **Backend:** FastAPI (Python 3.8+)
- **Document Parsing:** PyMuPDF (`fitz`) for extracting PDF text
- **Embeddings:** SentenceTransformers with `all-MiniLM-L6-v2`
- **Vector Search:** FAISS for fast similarity queries
- **LLM:** Google Gemini 1.5 Flash for question answering

---

## Setup Instructions

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
GEMINI_API_KEY=your_gemini_api_key
HF_API_TOKEN=your_huggingface_token
AUTH_TOKEN=your_bearer_auth_token
```

### 4. Run the API server
```bash
python main.py
```
The server listens by default at `http://localhost:7860`

---

## API Endpoints

#### Health Check
```http
GET /health
```
Returns the service status

#### Main Processing Endpoint
```http
POST /hackrx/run
```
**Request Body:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
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
    "The grace period for premium payment is thirty (30) days after the due date.",
    "There is a waiting period of thirty-six (36) months for pre-existing diseases."
  ]
}
```

---

## Architecture

### Core Components

- **DocumentProcessor (`core/document.py`)**  
  Handles downloading PDF files, extracting text, cleaning and creating overlapping chunks for semantic search

- **RAGService (`core/rag.py`)**  
  Generates embeddings for text chunks, manages FAISS vector store, retrieves relevant context and generates answers using Gemini LLM

- **API Layer (`app.py`)**  
  FastAPI instance providing routing, CORS support, bearer token authentication and request/response validation with Pydantic models

- **Data Models (`models.py`)**
   Defines Pydantic request and response models for validating and structuring API inputs and outputs

### Data Process Flow

```
PDF URL → Download & Extract Text → Clean & Chunk → Create Embeddings → FAISS Vector Store → Question → Embed → Search for Relevant Chunks → Context → Gemini LLM → Clean & Return Answer
```

---

## Contributors

- [Naincy Jain](https://www.linkedin.com/in/naincy-jain-38a20a283)
- [Aarjav Jain](https://www.linkedin.com/in/bharatwalejain)