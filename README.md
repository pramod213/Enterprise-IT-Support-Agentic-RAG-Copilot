# Enterprise-IT-Support-Agentic-RAG-Copilot

An end-to-end **Forward Deployed Engineer (FDE) project** that turns an
Agentic RAG workflow into a deployable internal IT support product using
**LangGraph, FastAPI, Pinecone, Groq, Tavily, Hugging Face embeddings,
HTML, CSS, and JavaScript**.

The application is designed around a fictional enterprise customer,
**NovaRetail**, and demonstrates private knowledge retrieval, evidence
grading, web fallback, query rewriting, document ingestion, audit
logging, and a user-facing chat interface.

------------------------------------------------------------------------

## 1. Business Problem

### Customer

**NovaRetail**, a fictional 3,000-employee retail company.

### Problem

The internal IT team maintains documents covering VPN access, password
policies, MFA, software installation, laptop troubleshooting, security
procedures, and service-desk runbooks.

Employees still create repetitive support tickets because:

-   They do not know where the correct document is.
-   Traditional keyword search can return too many results.
-   A normal chatbot may hallucinate an answer.
-   Internal documents may be incomplete or outdated.
-   Some questions require current information from the public web.

### Example

An employee asks:

> **How do I connect to the company VPN from home?**

If the answer exists in the private company knowledge base, the system
should use the private KB rather than search the public internet.

Another employee asks:

> **What is the latest Microsoft Teams outage guidance?**

If the internal KB does not contain sufficient current information, the
system can fall back to external web search and clearly identify the
answer as web-based information.

### Business Goal

Build an IT Support Copilot that:

1.  Searches trusted private knowledge first.
2.  Evaluates whether retrieved evidence is sufficient.
3.  Uses web search only when private knowledge is insufficient.
4.  Rewrites weak queries and retries retrieval.
5.  Generates grounded answers.
6.  Shows the execution path for transparency and debugging.
7.  Allows authorized staff to add new company documents.

------------------------------------------------------------------------

## 2. Why This Is an FDE Project

A Forward Deployed Engineer does more than create an LLM notebook. The
FDE must turn a customer problem into a usable product.

``` text
Customer Problem
      ↓
Discovery & Requirements
      ↓
Solution Architecture
      ↓
Knowledge Integration
      ↓
Agentic RAG Development
      ↓
API Development
      ↓
User Interface
      ↓
Security + Audit + Testing
      ↓
Deployment
      ↓
Observe + Improve
```

This repository demonstrates the progression from an Agentic RAG
workflow to a functional internal application.

------------------------------------------------------------------------

## 3. Architecture

``` text
                   ┌─────────────────────┐
                   │   Employee / User   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   HTML/CSS/JS UI    │
                   │   Chat + Ingestion  │
                   └──────────┬──────────┘
                              │
                              │ POST /api/chat
                              ▼
                   ┌─────────────────────┐
                   │       FastAPI       │
                   │ REST API + Security │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │     LangGraph       │
                   │  Agentic RAG Graph  │
                   └──────────┬──────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       ┌───────────────┐              ┌──────────────┐
       │ Private KB    │              │ Tavily Web   │
       │ Pinecone      │              │ Search       │
       │ + HF Embeds   │              └──────┬───────┘
       └───────┬───────┘                     │
               │                             │
               └──────────────┬──────────────┘
                              ▼
                     ┌────────────────┐
                     │    Groq LLM    │
                     │ Routing / Grade │
                     │ Rewrite / Gen   │
                     └────────────────┘
```

------------------------------------------------------------------------

## 4. Agentic RAG Workflow

``` text
Question
   ↓
[1] Route Question
   ├── Greeting / simple chat ─────────────→ Direct Answer
   │
   └── IT support question
                ↓
[2] Retrieve from Private Pinecone KB
                ↓
[3] Grade Private Evidence
       ┌────────┴────────┐
       │                 │
     GOOD               WEAK
       │                 │
       ▼                 ▼
Generate from KB    [4] Tavily Web Search
                         ↓
                  [5] Grade Web Evidence
                    ┌────┴─────┐
                    │          │
                  GOOD        WEAK
                    │          │
                    ▼          ▼
              Generate Web  [6] Rewrite Query
                               ↓
                         Retry Private KB
                               ↓
                        Max retry reached?
                               ↓
                    Insufficient Evidence
```

### Why Agentic?

Traditional RAG is approximately:

``` text
Question → Retrieve → Generate
```

This project makes state-dependent decisions:

``` text
Question
 → Route
 → Retrieve
 → Evaluate evidence
 → Choose KB or Web
 → Rewrite if needed
 → Retry
 → Generate grounded answer
```

The workflow therefore determines **what to do next** based on the
current state.

------------------------------------------------------------------------

## 5. Technology Stack

  --------------------------------------------------------------------------
  Layer                   Technology                 Purpose
  ----------------------- -------------------------- -----------------------
  Agent workflow          LangGraph                  Stateful routing and
                                                     conditional decisions

  LLM                     Groq                       Routing, grading,
                                                     rewriting, and answer
                                                     generation

  Embeddings              Hugging Face               Local 384-dimensional
                          `BAAI/bge-small-en-v1.5`   embeddings

  Vector database         Pinecone                   Private enterprise
                                                     knowledge base

  External search         Tavily                     Fallback when private
                                                     KB evidence is
                                                     insufficient

  API                     FastAPI                    Backend REST API

  Frontend                HTML/CSS/JavaScript        Employee-facing chat
                                                     and document UI

  Audit                   SQLite                     Basic decision-path
                                                     logging

  Document parsing        pypdf / python-docx        PDF and DOCX text
                                                     extraction

  Text splitting          LangChain text splitters   Recursive document
                                                     chunking

  Packaging               Docker                     Reproducible deployment

  Environment management  uv                         Python 3.11 environment
                                                     and dependency
                                                     management
  --------------------------------------------------------------------------

> **Note:** The current implementation uses Hugging Face embeddings and
> Groq. OpenAI embeddings and OpenAI chat are not part of the current
> application path.

------------------------------------------------------------------------

## 6. Project Structure

``` text
Enterprise-IT-Support-Agentic-RAG-Copilot/
│
├── app/
│   ├── api/
│   │   └── routes.py              # Health, chat and document ingestion APIs
│   │
│   ├── core/
│   │   ├── config.py              # Environment configuration
│   │   └── logging.py             # Logging configuration
│   │
│   ├── rag/
│   │   ├── state.py               # LangGraph state and structured decisions
│   │   ├── vectorstore.py         # Hugging Face embeddings + Pinecone
│   │   └── workflow.py            # Complete Agentic RAG graph
│   │
│   ├── services/
│   │   ├── audit.py               # SQLite query audit
│   │   └── ingestion.py           # PDF/TXT/MD/DOCX loading + chunking
│   │
│   └── main.py                    # FastAPI application
│
├── data/
│   └── sample_kb/
│       ├── company_it_handbook.md
│       └── service_desk_runbook.md
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── templates/
│   └── index.html
│
├── tests/
│   └── test_ingestion.py
│
├── uploads/
├── .env.example
├── Dockerfile
├── ingest_sample_kb.py
├── requirements.txt
├── run.py
└── README.md
```

------------------------------------------------------------------------

## 7. Prerequisites

-   Python **3.11**
-   `uv`
-   A Groq API key
-   A Pinecone API key
-   A Tavily API key
-   Internet access for API calls and initial Hugging Face model
    download

------------------------------------------------------------------------

## 8. Setup with uv

### Step 1 --- Create the environment

From the project root:

``` bash
uv venv --python 3.11 fde-project
```

Activate it on Windows:

``` bash
fde-project\Scripts\activate
```

For macOS/Linux:

``` bash
source fde-project/bin/activate
```

### Step 2 --- Install dependencies

If using the existing requirements file:

``` bash
uv pip install -r requirements.txt
```

------------------------------------------------------------------------

## 9. Environment Configuration

Create a `.env` file in the project root.

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

TAVILY_API_KEY=your_tavily_api_key

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=fde-it-support-rag
PINECONE_NAMESPACE=company-it-kb

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

ADMIN_API_KEY=change-me-in-production

APP_ENV=development
```

### Environment Variable Reference

  Variable                Description                         Required
  ----------------------- ----------------------------------- ----------
  `GROQ_API_KEY`          API key for Groq                    Yes
  `GROQ_MODEL`            Groq model identifier               No
  `TAVILY_API_KEY`        API key for Tavily search           Yes
  `PINECONE_API_KEY`      API key for Pinecone                Yes
  `PINECONE_INDEX_NAME`   Pinecone index name                 No
  `PINECONE_NAMESPACE`    Pinecone namespace                  No
  `EMBEDDING_MODEL`       Hugging Face embedding model        No
  `ADMIN_API_KEY`         Secret used by document ingestion   No
  `APP_ENV`               Application environment             No

### Security Notes

-   Never commit `.env` to version control.
-   Change `ADMIN_API_KEY` before production use.
-   Keep API keys in environment variables.
-   Restrict credentials to the services that require them.

------------------------------------------------------------------------

## 10. Knowledge Base Ingestion

The project includes sample company knowledge under:

``` text
data/sample_kb/
```

Load the sample knowledge base with:

``` bash
python ingest_sample_kb.py
```

The ingestion pipeline is:

``` text
Company Documents
       ↓
Load Documents
       ↓
Chunk Text
       ↓
Hugging Face Embeddings
       ↓
Pinecone Index
```

The current embedding model is:

``` text
BAAI/bge-small-en-v1.5
```

It produces **384-dimensional embeddings**.

> When changing embedding models, the Pinecone index dimension and
> stored vectors must be compatible with the selected embedding model.
> Existing documents may need to be re-indexed after an embedding-model
> change.

------------------------------------------------------------------------

## 11. Running the Application

Start the application with:

``` bash
python run.py
```

Open the web interface:

``` text
http://127.0.0.1:8000
```

FastAPI documentation:

``` text
http://127.0.0.1:8000/docs
```

The application provides:

-   IT support chat
-   Private KB grounded answers
-   Web fallback answers
-   Execution trace visibility
-   Source/citation information
-   Suggested questions
-   Authorized document ingestion
-   Health check endpoint

------------------------------------------------------------------------

## 12. API Endpoints

### `GET /api/health`

Returns application health information.

### `POST /api/chat`

Request:

``` json
{
  "question": "How do I reset my company password?"
}
```

Response structure:

``` json
{
  "answer": "...",
  "source_used": "private_kb",
  "trace": [
    "Router → KB",
    "Private KB retrieval → 4 chunks",
    "KB evidence grade → GOOD",
    "Answer generation → PRIVATE KB"
  ],
  "citations": [],
  "rewritten_query": "..."
}
```

### `POST /api/ingest`

Uploads a company document and adds its chunks to Pinecone.

Supported file types:

-   `.pdf`
-   `.txt`
-   `.md`
-   `.docx`

The endpoint requires the `X-Admin-Key` header.

Example:

``` text
X-Admin-Key: your-admin-key
```

The ingestion flow is:

``` text
Upload
  ↓
Validate File Type
  ↓
Load Text
  ↓
Recursive Chunking
  ↓
Hugging Face Embeddings
  ↓
Pinecone Indexing
  ↓
Available for Retrieval
```

------------------------------------------------------------------------

## 13. Classroom / Portfolio Demo Scenarios

### Demo A --- Private KB Success

Ask:

> How do I connect to the company VPN from home?

Expected path:

``` text
Router → KB
Private KB Retrieval
KB Grade → GOOD
Generate from Private KB
```

Teaching point:

> Trusted internal company knowledge is preferred. Public web search is
> not required when the private KB contains sufficient evidence.

------------------------------------------------------------------------

### Demo B --- Company Policy Question

Ask:

> Can IT support ask me to share my MFA code?

If the company handbook contains the relevant policy, the expected path
is:

``` text
Router → KB
Private KB Retrieval
KB Grade → GOOD
Private KB Answer
```

Teaching point:

> RAG allows the application to answer using company-specific
> information that is not part of the model's general knowledge.

------------------------------------------------------------------------

### Demo C --- External / Current Information

Ask:

> What is the latest Microsoft Teams outage guidance?

When the private KB does not contain sufficient information:

``` text
Router → KB
Private KB Retrieval
KB Grade → WEAK
Tavily Search
Web Grade → GOOD
Web Answer
```

Teaching point:

> Agentic RAG can select another information source when private
> evidence is insufficient.

------------------------------------------------------------------------

### Demo D --- Query Rewrite

Ask a deliberately vague question such as:

> My work communication app is acting strange after the new update. What
> should I do?

If the available evidence is weak, the workflow can rewrite the query
and retry retrieval.

Teaching point:

> Retrieval failure does not immediately terminate the workflow. The
> agent can improve the query and retry while using a retry guard to
> prevent loops.

------------------------------------------------------------------------

### Demo E --- Direct Conversation

Ask:

> Hello!

Expected path:

``` text
Router → DIRECT
Direct Answer
```

Teaching point:

> Not every message should trigger vector retrieval or web search.

------------------------------------------------------------------------

## 14. Document Upload Demo

The web interface includes an **Add Company Document** capability for
authorized users.

Supported documents:

-   PDF
-   TXT
-   Markdown
-   DOCX

The backend validates the file type, loads the document, chunks the
content, creates embeddings, and indexes the chunks in Pinecone.

This demonstrates an important FDE workflow: customer knowledge can be
updated without modifying application source code for every new policy
or runbook.

------------------------------------------------------------------------

## 15. Security and Reliability Considerations

The project includes several safeguards:

-   Private KB is preferred for company-specific IT questions.
-   Web search is used as a controlled fallback.
-   Evidence is graded before answer generation.
-   Query rewriting has a retry limit.
-   Document ingestion requires an admin API key.
-   Uploaded filenames are sanitized before storage.
-   File types are validated before ingestion.
-   API secrets are loaded from environment variables.
-   Audit information is stored in SQLite.
-   Answer-generation prompts instruct the model to remain grounded in
    the supplied evidence.
-   Generated answers are requested as plain text rather than Markdown.

This is a demonstration application and should receive additional
enterprise security hardening before handling real confidential company
data.

------------------------------------------------------------------------

## 16. Testing and Validation

The project includes automated tests under:

``` text
tests/
```

Run the test suite with:

``` bash
pytest
```

Basic application validation can also be performed through:

``` text
GET /api/health
POST /api/chat
POST /api/ingest
```

The application has been tested locally with:

-   FastAPI startup
-   Frontend page loading
-   Groq API requests
-   Hugging Face embedding model initialization
-   Private KB retrieval
-   `/api/chat` response generation

------------------------------------------------------------------------

## 17. Docker

The project includes a `Dockerfile` for containerized execution.

Build:

``` bash
docker build -t enterprise-it-support-copilot .
```

Run:

``` bash
docker run --env-file .env -p 8000:8000 enterprise-it-support-copilot
```

Then open:

``` text
http://127.0.0.1:8000
```

------------------------------------------------------------------------

## 18. Current Project Status

The project currently includes:

-   Agentic RAG workflow implemented with LangGraph
-   Groq-based routing, grading, query rewriting, and generation
-   Hugging Face local embeddings
-   Pinecone private knowledge retrieval
-   Tavily web fallback
-   FastAPI REST endpoints
-   PDF/TXT/MD/DOCX ingestion
-   SQLite audit logging
-   HTML/CSS/JavaScript frontend
-   Chat response and execution-trace display
-   Fixed viewport with conversation-only scrolling
-   Plain-text answer-generation instructions
-   Docker support
-   Sample company knowledge base

------------------------------------------------------------------------

## 19. Future Improvements

Potential next steps include:

-   Authentication and role-based access control
-   Production-grade secrets management
-   More comprehensive automated tests
-   Evaluation datasets and retrieval metrics
-   Observability and tracing
-   Feedback collection for answer quality
-   Document versioning
-   Knowledge-base access controls
-   Background ingestion jobs
-   Production deployment and monitoring
-   Enterprise SSO integration

------------------------------------------------------------------------

## 20. License / Project Context

This is a portfolio and educational FDE project built around a fictional
company and fictional internal IT documentation.

**NovaRetail is fictional.** Company policies, documents, and support
scenarios included in the repository are demonstration data and should
not be treated as real corporate policies.
