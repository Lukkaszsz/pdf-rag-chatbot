# 📚 PDF RAG Chatbot AI

Inteligentny asystent do analizy dokumentów PDF z wykorzystaniem Retrieval-Augmented Generation (RAG), hybrid search i multimodalnego przetwarzania.

## 🚀 Funkcje

- 📄 Multi-PDF Upload - Analiza wielu dokumentów jednocześnie
- 💬 Inteligentny Czat - Q&A na podstawie treści dokumentów
- 🔍 Hybrid Search - BM25 + Vector Similarity
- 🖼️ Multimodal OCR - Obsługa skanowanych PDF
- 📊 Automatyczne Wykresy - Wizualizacja danych z PDF
- 📥 Export Historii - CSV/TXT
- ✅ Feedback System - Ocena jakości odpowiedzi

## 🛠️ Technologie

- **Frontend:** Streamlit
- **LLM:** Groq API (Llama 3.1, Llama 3.3, Llama 4, Qwen3)
- **Embeddings:** Sentence-Transformers
- **Vector DB:** ChromaDB
- **RAG:** LangChain
- **OCR:** Tesseract + pdf2image

## 📦 Instalacja

### Wymagania systemowe

**Tesseract OCR:**
```bash
Windows: https://github.com/UB-Mannheim/tesseract/wiki
Linux: sudo apt install tesseract-ocr
macOS: brew install tesseract
```

**Poppler:**
```bash
Windows: https://github.com/oschwartz10612/poppler-windows/releases/
Linux: sudo apt install poppler-utils
macOS: brew install poppler
```

### Instalacja projektu

```bash
# Zainstaluj zależności
pip install -r requirements.txt

# Skopiuj .env.example do .env i dodaj klucz
cp .env.example .env

# Edytuj .env i wstaw GROQ_API_KEY
# Uruchom aplikację
streamlit run app.py
```

### Konfiguracja API

Uzyskaj darmowy klucz Groq:
1. [console.groq.com](https://console.groq.com)
2. Wygeneruj API key
3. Dodaj do `.env`:
```
GROQ_API_KEY=gsk_twój_klucz
```

## 📖 Użycie

1. **Upload PDF** - Wgraj dokumenty
2. **Indeksuj** - Kliknij przycisk indeksowania
3. **Zadawaj pytania** - Czatuj z dokumentami
4. **Oceń odpowiedzi** - ✅ Trafione / ❌ Chybione

## 🚀 Deployment na Streamlit Cloud

1. Push do GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. W Secrets dodaj:
```toml
GROQ_API_KEY = "gsk_twój_klucz"
```

## 👨‍💻 Autor

**[Lukkaszsz](https://github.com/Lukkaszsz/pdf-rag-chatbot)**

**Stack:** Python • Streamlit • LangChain • RAG • Vector DB • NLP • OCR

---

## ⚙️ Ustawienia – Chunkowanie i Modele

### Chunkowanie (podział dokumentu)

W panelu bocznym możesz dostosować dwa parametry:

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| **Rozmiar chunku** | 1000 znaków | Większy chunk = więcej kontekstu, ale większy "szum" w odpowiedzi |
| **Overlap** | 200 znaków | Nakładanie sąsiednich chunków – zapobiega utracie sensu na granicy fragmentów |

**Kiedy zmieniać?**
- 📄 Krótkie dokumenty, faktury, notatki → chunk **500–800**, overlap **100**
- 📚 Długie raporty, książki → chunk **1500–2000**, overlap **300–400**
- 🔍 Pytania o szczegóły techniczne → mniejszy chunk (więcej precyzji)
- 💬 Pytania ogólne/podsumowania → większy chunk (więcej kontekstu)

### Modele LLM

| Wybór w UI | Rzeczywisty model (Groq) | Charakterystyka |
|------------|--------------------------|-----------------|
| ⚡ `llama-3.1-8b (szybki)` | `llama-3.1-8b-instant` | Najszybszy, do prostych pytań |
| ⚖️ `qwen3-32b (zbalansowany)` | `qwen/qwen3-32b` | Dobry balans jakość/szybkość |
| 🏆 `llama-3.3-70b (najlepszy)` | `llama-3.3-70b-versatile` | Najlepsza jakość odpowiedzi |
| 🖼️ `llama-3.2-90b (vision)` | `llama-3.2-90b-vision-preview` | Obsługa obrazów i tabel |

> Wszystkie modele działają przez **Groq API** – wymagają klucza `GROQ_API_KEY`.
