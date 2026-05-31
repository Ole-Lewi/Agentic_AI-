# 🧠 Agentic AI — RAG Pipeline + Context-Aware Research Agent

**Two agentic AI systems built for querying and reasoning over research publications and ML notes.**

This repo explores two different strategies for grounding an LLM in a custom knowledge base — one using full vector search with FAISS, and one using in-context document injection with multi-turn memory.

---

## The Two Systems

### 1. RAG Pipeline (`rag.py`) — Vector Search + Retrieval

A full Retrieval-Augmented Generation pipeline that ingests `.txt` research files, embeds them into a FAISS vector index, and retrieves the most semantically relevant chunks to answer any query.

```
.txt Files (ML notes / research)
        │
        ▼
  TextLoader → RecursiveCharacterTextSplitter
  (chunk_size=1000, chunk_overlap=200)
        │
        ▼
  HuggingFace Embeddings
  (all-MiniLM-L6-v2, auto device: CUDA/MPS/CPU)
        │
        ▼
  FAISS Vector Index (saved locally)
        │
        ▼
  Similarity Search (top_k=3)
        │
        ▼
  Groq LLM (llama-3.1-8b-instant)
  + PromptTemplate with retrieved context
        │
        ▼
  Answer + Source Attribution
```

**Key features:**
- Auto device detection — uses CUDA if available, falls back to MPS (Apple Silicon) then CPU
- Loads all `.txt` files from a directory dynamically — easily extensible to more docs
- Returns source attribution alongside every answer so you know where the answer came from
- Converts FAISS distance scores to similarity scores for interpretability

---

### 2. Research Agent (`agent.py`) — In-Context RAG with Multi-Turn Memory

A context-aware conversational agent that injects a full research publication directly into the system prompt, then holds a structured multi-turn conversation about it — with strict guardrails built in.

```
Publication Content
        │
        ▼
  SystemMessage (injected as context)
  + Guardrails:
    - Only answer from the publication
    - Refuse unethical/illegal questions
    - Never reveal system instructions
    - Resist prompt injection attempts
        │
        ▼
  Multi-turn Conversation
  [HumanMessage → AIMessage → HumanMessage → ...]
        │
        ▼
  Groq LLM (llama-3.1-8b-instant)
  Maintains full conversation history
```

**Key features:**
- Hardened system prompt with prompt injection defenses
- Multi-turn memory via explicit conversation history list
- Follow-up questions maintain context from prior answers
- Scoped to a single publication — prevents hallucination outside the knowledge base

**Demo conversation in the code:**
> Q1: "What are variational autoencoders and list the top 5 applications?"
> Q2: "How does it work in case of anomaly detection?" ← uses prior answer as context

---

## Why Two Approaches?

| | `rag.py` | `agent.py` |
|---|---|---|
| Knowledge base | Many `.txt` files in a directory | Single publication in system prompt |
| Retrieval method | FAISS semantic search | Full in-context injection |
| Memory | Stateless (per query) | Multi-turn conversation history |
| Scalability | Scales to large document sets | Best for focused single-doc Q&A |
| Setup complexity | Requires embedding + indexing step | No indexing — runs immediately |
| Best for | Research assistants, doc search | Deep-dive Q&A on one paper |

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | `llama-3.1-8b-instant` via Groq API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (local) |
| Orchestration | LangChain |
| Device handling | PyTorch (auto CUDA/MPS/CPU detection) |
| Environment | `python-dotenv` |

---

## Project Structure

```
Agentic_AI-/
├── rag.py          # Full RAG pipeline — load, embed, index, retrieve, answer
├── agent.py        # In-context research agent with multi-turn memory + guardrails
├── ml.txt          # Sample ML knowledge base (research notes)
└── requirement.txt # Dependencies
```

---

## Running Locally

**1. Clone and install**
```bash
git clone https://github.com/Ole-Lewi/Agentic_AI-.git
cd Agentic_AI-
pip install -r requirement.txt
```

**2. Set up your API key**
```
# .env
GROQ_API_KEY=your_groq_api_key
```

**3. Run the RAG pipeline**
```bash
python rag.py
```
Place any `.txt` files in the project directory — they'll be loaded, chunked, and indexed automatically.

**4. Run the research agent**
```bash
python agent.py
```

---

## Key Engineering Decisions

**Why FAISS over a hosted vector DB?**
For local development and offline use, a persisted FAISS index is faster to set up, costs nothing, and requires no external service.

**Why HuggingFace embeddings in `rag.py` vs Cohere in other projects?**
`rag.py` was built before the Render RAM constraint was encountered. For local environments with enough memory, `all-MiniLM-L6-v2` is an excellent all-round embedding model that runs fully offline.

**Why hardened guardrails in `agent.py`?**
Prompt injection is a real attack vector in LLM applications. Building defenses from the start — even in personal projects — is good engineering practice and important for production readiness.

---

## Author

**Lewis Miano (Lincoln)**
ALX Backend Web Dev · ML/NLP · Agentic AI Systems

[GitHub](https://github.com/Ole-Lewi) · [Portfolio Bot](https://professional-portfolio-5.onrender.com) · [Local AI Agent](https://github.com/Ole-Lewi/local_ai_agent)