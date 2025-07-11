# QueryBot - An AI-Powered Document Analyst

QueryBot is a GenAI-driven chatbot that allows users to query, summarize, and explore complex PDF documents such as investment reports, financial statements, product briefs, or research papers using Retrieval-Augmented Generation (RAG) techniques and locally hosted LLMs.

---

## What It Does

Upload any PDF and ask InsightBot:
- "What are the key takeaways from this report?"
- "Summarise the risk factors mentioned."
- "Rewrite this in layman's terms."
- "When is the next product milestone?"

QueryBot performs:
- Semantic chunking + embedding
- Fast vector search using FAISS
- LLM-based answer generation with source citations
---

##  Key Features

| Feature                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
|  PDF Upload               | Accepts user documents for ingestion                                        |
|  RAG (Retrieval-Augmented Generation) | Finds answers grounded in your own content using FAISS + LLM            |
|  Local LLM Support        | Works with LLaMA 3 / Mistral via Ollama for private, fast inference         |
|  Document Summarization   | Get TL;DRs or section-level summaries                                       |
|  Source Citations         | Responses include source text snippets for transparency                    |
|  GenAI Rewriting          | Ask the bot to simplify or rephrase content                                |
|  Easy UI                  | Built with Streamlit for a clean, responsive chat experience                |

---

##  Tech Stack

-  LLM: [LLaMA 3](https://llama.meta.com/), [Mistral](https://mistral.ai/)
-  RAG: [LlamaIndex](https://github.com/jerryjliu/llama_index) / [LangChain](https://www.langchain.com/)
-  PDF Parsing: PyMuPDF / pdfminer
-  Vector DB: FAISS or ChromaDB
-  UI: Streamlit
-  Local LLM: [Ollama](https://ollama.com/)
-  Embedding Model: sentence-transformers or Ollama-embedded LLM

---

## Demo 

##  Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/shimonadas/QueryBot.git
cd QueryBot
```

### 2. Set Up Environment

```bash
pip install -r requirements.txt
```

### 3. Run Ollama with Mistral

```bash
ollama run mistral
```

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

## Folder Structure

```graphql
QueryBot/
├── app.py              # Streamlit frontend
├── ingest.py           # PDF chunking and FAISS embedding
├── query_engine.py     # RAG + LLM interaction logic
├── prompts/
│   └── prompt_template.txt
├── data/               # Upload folder (can be .gitignored)
├── vector_store/       # Saved FAISS vector index (auto-generated)
├── requirements.txt
├── .gitignore
└── README.md
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
