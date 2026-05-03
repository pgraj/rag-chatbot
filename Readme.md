#  RAG Chatbot — Course Catalog AI Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that ingests a course catalog PDF, stores vector embeddings in ChromaDB, and lets users query course information through an interactive Streamlit chat interface. Includes a full pipeline monitoring dashboard and vector database explorer.

---

##  Project Structure

```
rag-chatbot/
├── app.py                  # Application entry point / launcher
├── chat_ui.py              # Streamlit chat interface for querying the RAG pipeline
├── pipeline_monitor.py     # Streamlit dashboard for monitoring pipeline health & performance
├── rag_pipeline.py         # Core RAG logic: embedding, retrieval, and LLM response generation
├── ingest.py               # PDF ingestion: loads and chunks the course catalog into ChromaDB
├── inspect_db.py           # CLI utility to inspect ChromaDB collection contents
├── inspect_ui.py           # Streamlit UI for browsing ChromaDB collections
├── vector_explorer.py      # Visual explorer for vector embeddings and similarity search
├── course_catalog.pdf      # Source document ingested into the vector store
├── requirements.txt        # All Python dependencies with pinned versions
├── .env                    # Your private API keys (NOT committed — see setup below)
├── .gitignore              # Excludes .env, chroma_db/, and runtime files
└── README.md               # This file
```

> **Auto-generated at runtime (not committed):**
> `chroma_db/` — ChromaDB vector store, rebuilt by running `ingest.py`
> `query_bridge.json` — runtime data bridge between pipeline components

---

##  What Each File Does

| File | Description |
|------|-------------|
| `app.py` | Main launcher. Initialises the application and routes to the appropriate Streamlit interface. |
| `chat_ui.py` | Full Streamlit chat interface. Accepts user questions, passes them through the RAG pipeline, and displays LLM-generated answers with retrieved source context. **This is the main user-facing app.** |
| `pipeline_monitor.py` | Streamlit monitoring dashboard. Displays pipeline metrics, query logs, response times, retrieval scores, and system health indicators in real time. |
| `rag_pipeline.py` | The core engine. Handles document chunking, embedding generation, ChromaDB vector search, prompt construction, and LLM API calls. All UI components call into this module. |
| `ingest.py` | One-time setup script. Reads `course_catalog.pdf`, splits it into chunks, generates embeddings, and persists them in ChromaDB. Must be run before launching the chat UI. |
| `inspect_db.py` | Command-line utility for directly querying and printing ChromaDB collection contents. Useful for debugging the vector store without a UI. |
| `inspect_ui.py` | Streamlit companion to `inspect_db.py`. Browse ChromaDB collections visually, view stored document chunks, and check metadata. |
| `vector_explorer.py` | Advanced Streamlit tool for exploring the embedding space. Visualises vector similarity, allows manual similarity searches, and shows how document chunks are distributed. |

---

##  Requirements

### Python Version
```
Python 3.11.9
```

### Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.57.0 | Web UI framework for all interactive dashboards |
| `chromadb` | 1.5.8 | Local persistent vector database |
| `langchain` | 1.2.15 | RAG chain orchestration and document splitting |
| `langchain-core` | 1.3.0 | Core LangChain primitives |
| `langchain-anthropic` | 1.4.1 | Anthropic Claude integration via LangChain |
| `langchain-google-genai` | 4.2.2 | Google Gemini integration via LangChain |
| `anthropic` | 0.96.0 | Direct Anthropic Claude API client |
| `google-genai` | 1.73.1 | Google Generative AI SDK |
| `sentence-transformers` | 5.4.1 | Local embedding model support |
| `transformers` | 5.6.2 | Hugging Face model loading |
| `torch` | 2.11.0 | PyTorch backend for local models |
| `pypdf` | 6.10.2 | PDF reading and text extraction |
| `python-dotenv` | 1.2.2 | Loads API keys from `.env` file |
| `pandas` | 3.0.2 | Data handling in monitoring dashboard |
| `plotly` | 6.7.0 | Charts and visualisations in pipeline monitor |
| `numpy` | 2.4.4 | Numerical operations |
| `scikit-learn` | 1.8.0 | ML utilities (dimensionality reduction, metrics) |
| `langgraph` | 1.1.8 | Stateful multi-step LLM pipeline graphs |

Full pinned dependency list: see `requirements.txt`

---

##  API Key Setup (Required)

This project uses LLM APIs (Anthropic Claude and/or Google Gemini). You must supply your own API keys.

**The `.env` file is excluded from this repository for security. You must create it yourself.**

### 1. Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

Add only the keys relevant to the LLM provider you are using.

### 2. Where to get your API keys:

| Provider | URL |
|----------|-----|
| Anthropic (Claude) | https://console.anthropic.com/settings/keys |
| Google Gemini | https://aistudio.google.com/app/apikey |

### 3. Important — never share this file.
It is listed in `.gitignore` and will never be pushed to GitHub.

---

##  How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
# source venv/bin/activate   # Mac / Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> `torch==2.11.0` is a large download (~2 GB). Allow extra time for this step.

### Step 4 — Create your `.env` file
Create `.env` in the project root and add your API keys (see API Key Setup above).

### Step 5 — Ingest the course catalog *(first time only)*
Reads the PDF and populates ChromaDB. Only needed once, or when the source PDF changes.
```bash
python ingest.py
```

### Step 6 — Launch the Chat UI
```bash
streamlit run chat_ui.py
```
Open your browser at: **http://localhost:8501**

Type any question about the course catalog — the RAG pipeline retrieves relevant content and generates a grounded answer.

---

##  Running the Pipeline Monitor

```bash
streamlit run pipeline_monitor.py
```
Open your browser at: **http://localhost:8501**

The pipeline monitor displays:
- Live query history and response latency
- Retrieval relevance scores per query
- ChromaDB collection statistics
- Pipeline health indicators
- Token usage and LLM call metrics

---

##  Other Utilities

**Inspect ChromaDB contents (command line):**
```bash
python inspect_db.py
```

**Inspect ChromaDB contents (browser UI):**
```bash
streamlit run inspect_ui.py
```

**Explore vector embeddings visually:**
```bash
streamlit run vector_explorer.py
```

---

##  Notes

- `chroma_db/` is excluded from the repository and is auto-generated when you run `ingest.py`. You do not need to commit it.
- `query_bridge.json` is a runtime file created automatically during pipeline execution. It does not need to be committed.
- If you replace `course_catalog.pdf` with a new document, re-run `ingest.py` to rebuild the vector store.

---

##  Licence

MIT Licence — free to use, modify, and distribute.

---

##  Questions or Issues?

Open a GitHub Issue or start a Discussion in this repository.
