# Production Document RAG System

This project implements a production-ready RAG pipeline with local document loading, chunking, local or OpenAI embeddings, FAISS or Chroma Cloud vector search, grounded LLM generation, structured logging, Pydantic validation, and LangSmith tracing.

## Features

- Loads local `PDF`, `TXT`, and `JSON` files.
- Splits documents with `chunk_size=1000` and `chunk_overlap=200`.
- Generates embeddings locally by default, or with OpenAI when configured.
- Stores and loads a FAISS vector index or a Chroma Cloud collection.
- Retrieves top-k semantic matches, default `top_k=5`.
- Supports four retrieval strategies:
  - **Adaptive RAG** (default): classifies each question and routes to naive, advanced, or corrective retrieval.
  - **Advanced RAG**: fetches extra candidates, reranks for relevance, and diversifies chunks.
  - **Corrective RAG**: checks first-pass overlap, rewrites weak queries, and retries search.
  - **Naive RAG**: direct semantic top-k search.
- Builds grounded prompts from retrieved context.
- Generates answers with OpenAI Chat, Groq, or the local extractive fallback.
- Traces indexing, retrieval, and generation as custom LangSmith spans.
- Logs retrieval results, prompts, latency, token usage, and errors.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set at least `OPENAI_API_KEY` when using the default OpenAI providers. Set `LANGSMITH_API_KEY` to enable remote LangSmith traces.

## Configuration

Defaults live in `config.yaml`.

```yaml
documents:
  input_dir: data/documents
  chunk_size: 1000
  chunk_overlap: 200
vector_store:
  provider: chroma
  path: storage/faiss_index
  top_k: 5
  chroma_collection: rag_documents
  chroma_host: api.trychroma.com
retrieval:
  mode: adaptive
  advanced_fetch_multiplier: 4
  diversity_lambda: 0.7
  corrective_min_overlap: 1
embeddings:
  provider: local
  model: text-embedding-3-small
  local_dimensions: 384
llm:
  provider: extractive
  model: extractive-local
langsmith:
  enabled: true
```

The checked-in config uses local deterministic embeddings for indexing and Groq for chatbot generation. Set `GROQ_API_KEY` in `.env`. To use OpenAI embeddings or chat:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small
llm:
  provider: openai
  model: gpt-4o-mini
```

The Groq integration uses LangChain's `langchain-groq` package and defaults to `llama-3.3-70b-versatile`.

The checked-in vector store config uses Chroma Cloud. Put these values in `.env`:

```powershell
CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=rag
```

Install the Chroma packages before indexing with Chroma:

```powershell
pip install chromadb langchain-chroma
```

To switch back to the local FAISS index:

```yaml
vector_store:
  provider: faiss
  path: storage/faiss_index
```

## Usage

Put documents under `data/documents`, then build the vector index:

```powershell
rag index --config config.yaml --force
```

Ask a question:

```powershell
rag ask "What does the policy say about refunds?" --config config.yaml
```

Use a specific retrieval strategy for a question:

```powershell
rag ask "Compare refund and exchange policies" --retrieval-mode advanced
rag ask "What does the policy say about refunds?" --retrieval-mode corrective
rag ask "What does the policy say about refunds?" --retrieval-mode adaptive
rag ask "Who is the CEO?" --retrieval-mode naive
```

Return structured JSON:

```powershell
rag ask "What does the policy say about refunds?" --json
```

## Run Backend And Frontend

Start the API:

```powershell
.\.venv\Scripts\rag-api.exe
```

Start the browser UI in another terminal:

```powershell
cd frontend
python -m http.server 5173
```

Open `http://127.0.0.1:5173`. The UI can upload documents, build the vector index, chat over your documents, choose Adaptive/Advanced/Corrective/Naive RAG from the sidebar, and show retrieved sources (including corrective query rewrites).

You can also run the CLI without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m rag_app.cli index --force
python -m rag_app.cli ask "What does the policy say about refunds?"
```

## LangSmith

Tracing is configured with standard LangSmith environment variables:

```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_API_KEY = "..."
$env:LANGSMITH_PROJECT = "naive-rag-production"
```

The code creates custom spans for:

- `rag_index_documents`
- `load_local_documents`
- `build_faiss_index`
- `load_faiss_index`
- `rag_answer_question`
- `semantic_search`
- `generate_grounded_answer`

LangChain model calls include provider/model metadata so LangSmith can track prompts, inputs, outputs, latency, and token usage when the provider exposes it.

## Production Notes

- FAISS loading uses pickle metadata. The sample config sets `allow_dangerous_deserialization=true` because this app creates and reloads its own local index. Set it to `false` for untrusted indexes.
- Advanced retrieval fetches extra candidates and reranks for relevance plus diversity, which helps when several near-duplicate chunks crowd out useful context.
- Corrective retrieval checks whether the first pass overlaps the question terms. If the match is weak, it rewrites the query into key terms and retries with the advanced reranker.
- The default prompt forces factual grounding and instructs the model to say when retrieved context is insufficient.
- JSON files are parsed with Python's standard JSON parser and ingested as formatted JSON text. Customize `DocumentLoader._load_file` if your JSON files need field-specific extraction.
- OpenAI's current embeddings guide documents `text-embedding-3-small` and `text-embedding-3-large` for semantic search use cases.
"# rag-application" 
