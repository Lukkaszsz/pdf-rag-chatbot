# 📄 PDF RAG Chatbot AI

**Intelligent assistant for analyzing PDF documents using Retrieval-Augmented Generation (RAG), hybrid search, and multimodal processing**

---

## ✨ Features

- **Multi-PDF Upload** - Analyze multiple documents simultaneously
- **Intelligent Chat** - Q&A based on document content
- **Hybrid Search** - BM25 + Vector Similarity
- **Multimodal OCR** - Support for scanned PDFs
- **Automatic Charts** - Data visualization from PDFs
- **History Export** - CSV/TXT
- **Feedback System** - Response quality rating

---

## 🛠️ Technologies

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.1, Llama 3.3, Llama 4, Qwen3)
- **Embeddings**: Sentence-Transformers
- **Vector DB**: ChromaDB
- **RAG**: LangChain
- **OCR**: Tesseract + pdf2image

---

## 📋 Installation

### System Requirements

**Tesseract OCR:**
```bash
# Windows
https://github.com/UB-Mannheim/tesseract/wiki

# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

**Poppler:**
```bash
# Windows
https://github.com/oschwartz10612/poppler-windows/releases

# Linux
sudo apt install poppler-utils

# macOS
brew install poppler
```

### Project Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and add key
cp .env.example .env

# Edit .env and insert GROQ_API_KEY
```

### API Configuration

**Get free Groq key:**
1. https://console.groq.com
2. Generate API key
3. Add to `.env`:
```
GROQ_API_KEY=gsk_...your_key
```

---

## 🚀 Usage

1. **Upload PDF** - Upload documents
2. **Index** - Click index button
3. **Ask questions** - Chat with documents
4. **Rate responses** - ✅ Accurate / ❌ Inaccurate

---

## ☁️ Deployment on Streamlit Cloud

1. Push to GitHub
2. https://share.streamlit.io → "New app"
3. In **Secrets** add:
```toml
GROQ_API_KEY = "gsk_...your_key"
```

---

## ⚙️ Settings: Chunking and Models

### Chunking (document splitting)

You can adjust two parameters in the side panel:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Chunk size** | 1000 characters | Larger chunk = more context, but more noise in response |
| **Overlap** | 200 characters | Overlap of adjacent chunks prevents loss of meaning at fragment boundaries |

**When to change?**
- Short documents, invoices, notes → chunk 500-800, overlap 100
- Long reports, books → chunk 1500-2000, overlap 300-400
- Questions about technical details → smaller chunk = more precision
- General questions/summaries → larger chunk = more context

### LLM Models

| UI selection | Actual Groq model | Characteristics |
|--------------|-------------------|-----------------|
| **llama-3.1-8b (fast)** | llama-3.1-8b-instant | Fastest, for simple questions |
| **qwen3-32b (balanced)** | qwen/qwen3-32b | Good balance quality/speed |
| **llama-3.3-70b (best)** | llama-3.3-70b-versatile | Best answer quality |
| **llama-3.2-90b (vision)** | llama-3.2-90b-vision-preview | Supports images and tables |

All models run through Groq API (require `GROQ_API_KEY`).

---

## 👤 Author

**Lukkaszsz**

**Stack:** Python · Streamlit · LangChain · RAG · Vector DB · NLP · OCR

---
