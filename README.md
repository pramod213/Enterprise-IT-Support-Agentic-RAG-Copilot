# Enterprise-IT-Support-Agentic-RAG-Copilot

An end-to-end **Forward Deployed Engineer (FDE) project** that turns a notebook-style Agentic RAG workflow into a deployable internal product using **LangGraph, FastAPI, Pinecone, Groq, Tavily, HTML, CSS, and JavaScript**.

## 1. Setup

### Step 1 — Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment

Create a `.env` file in the project root directory with the following variables:

```env
# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# External Search
TAVILY_API_KEY=your_tavily_api_key_here

# Vector Database (Pinecone)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=fde-it-support-rag
PINECONE_NAMESPACE=company-it-kb

# LLM Models
GROQ_MODEL=openai/gpt-oss-20b
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Security
ADMIN_API_KEY=change-me-in-production

# Application Settings
APP_ENV=development
```