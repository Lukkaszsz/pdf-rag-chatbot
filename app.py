import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import fitz  #type: ignore
import io
import pytesseract
import platform
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from rank_bm25 import BM25Okapi
from PIL import Image
from pdf2image import convert_from_path

# Automatyczna konfiguracja Tesseract/Poppler dla różnych systemów
if platform.system() == "Windows":
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    poppler_path = r'C:\poppler\Library\bin'
    
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    if os.path.exists(poppler_path):
        os.environ['PATH'] += f';{poppler_path}'

# Opcjonalnie: użyj zmiennych środowiskowych z .env
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

POPPLER_PATH = os.getenv("POPPLER_PATH", "")
if POPPLER_PATH:
    sep = ';' if platform.system() == "Windows" else ':'
    os.environ['PATH'] += f'{sep}{POPPLER_PATH}'

# Ekstrahuje wszystkie obrazy z każdej strony PDF do plików
def extract_images_per_page(pdf_path: str, output_dir: str):
    """
    Zwraca mapę {(page_index): [ścieżki_do_obrazów]} dla danego PDF.
    Obrazy są zapisywane jako pliki PNG w output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_to_imgs = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        paths = []

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")

            img_filename = f"pdfimg_p{page_index+1}_{img_index}.{ext}"
            img_path = os.path.join(output_dir, img_filename)

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            paths.append(img_path)

        if paths:
            page_to_imgs[page_index + 1] = paths

    doc.close()   
    return page_to_imgs

def extract_text_with_ocr(pdf_path: str):
    """
    Wyciąga tekst ze skanowanego PDF za pomocą Tesseract OCR.
    Zwraca listę (page_index, text) dla każdej strony.
    """
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        ocr_results = []

        for page_index, image in enumerate(images):
            # OCR na obrazku strony
            text = pytesseract.image_to_string(image, lang="pol+eng")
            ocr_results.append((page_index + 1, text))

        return ocr_results
    except Exception as e:
        st.warning(f"⚠️ Błąd OCR: {str(e)}")
        return []

# =============================== KONFIGURACJA STRONY I STYL
st.set_page_config(
    page_title="PDF RAG Chatbot AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================== RATE LIMIT / SOFT LIMIT PER SESSION
if "llm_calls_count" not in st.session_state:
    st.session_state["llm_calls_count"] = 0

LLM_CALLS_LIMIT = 50  # <<<< MUSI być nad sidebar

# =============================== FEEDBACK NA ODPOWIEDZI (trafione/chybione)
if "llm_correct_count" not in st.session_state:
    st.session_state["llm_correct_count"] = 0
if "llm_incorrect_count" not in st.session_state:
    st.session_state["llm_incorrect_count"] = 0

# --- SIDEBAR: ustawienia modeli i RAG ---
with st.sidebar:
    st.header("⚙️ Ustawienia")

    model_key = st.selectbox(
    "Wybierz model:",
    [
        "⚡ llama-3.1-8b (szybki)",
        "⚖️ qwen3-32b (zbalansowany)",
        "🏆 llama-3.3-70b (najlepszy)",
        "🖼️ llama-4-maverick (multimodalny)",
    ],
    index=0,
    help="Wszystkie modele działają przez Groq API.\n"
         "⚡ Szybki – proste pytania\n"
         "⚖️ Zbalansowany – dobry do większości zadań\n"
         "🏆 Najlepszy – złożone analizy\n"
         "🖼️ Multimodalny – dokumenty z obrazami/tabelami",
)

    top_k_context = st.selectbox(
        "Ile dokumentów w kontekście?",
        [2, 3, 4, 5, 6],
        index=1,
        help="Liczba fragmentów PDF przekazywanych jako kontekst do modelu.",
    )

    st.markdown("### 🔧 Chunkowanie (tunable)")

    chunk_size = st.slider(
        "Rozmiar chunku (znaki):",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Większy chunk = więcej kontekstu w jednym fragmencie, ale też więcej szumu.",
    )

    chunk_overlap = st.slider(
        "Overlap między chunkami (znaki):",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="10–20% chunk size to zwykle dobry punkt startowy.",
    )

    st.markdown("---")
    used = st.session_state.get("llm_calls_count", 0)
    st.write(f"🔐 Zapytania w tej sesji: **{used}/{LLM_CALLS_LIMIT}**")
    correct = st.session_state.get("llm_correct_count", 0)
    incorrect = st.session_state.get("llm_incorrect_count", 0)
    total_labeled = correct + incorrect
    if total_labeled > 0:
        acc = 100 * correct / total_labeled
        st.write(f"✅ Trafione: **{correct}**")
        st.write(f"❌ Chybione: **{incorrect}**")
        st.write(f"🎯 Skuteczność odpowiedzi: **{acc:.1f}%**")
    else:
        st.write("Brak ocenionych odpowiedzi.")
    if used >= LLM_CALLS_LIMIT:
        st.warning(
            "Osiągnąłeś maksymalny limit zapytań w tej sesji. "
            "Odśwież stronę, aby rozpocząć nową sesję."
        )

st.session_state["selected_model"] = model_key
st.session_state["top_k_context"] = top_k_context
st.session_state["chunk_size"] = chunk_size
st.session_state["chunk_overlap"] = chunk_overlap

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stSidebarNav"] {background: #f1f5fa;}
[data-testid="stSidebar"] {display: block !important; visibility: visible !important;}
[data-testid="collapsedControl"] {display: none !important;}
.stTitle {color: #1f77b4;}
.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 20px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #0d5aa7;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📚 PDF RAG Chatbot AI")
st.markdown(
    """
🚀 **Inteligny asystent do analizy dokumentów PDF** z wykorzystaniem RAG (Retrieval-Augmented Generation)

**Funkcje:**
- 📄 Upload i automatyczne indeksowanie PDF (multi-PDF!)
- 💬 Pytania i odpowiedzi na podstawie treści wybranych dokumentów  
- 📋 Automatyczne podsumowania rozdziałów i całych dokumentów
- 📊 Generowanie wykresów z danych tekstowych
- 🧠 Insights: liczba plików, fragmentów, szacunkowe strony, główne tematy
- 🤖 Zasilany przez najnowsze modele LLM (Hugging Face / OpenAI-compatible)

*Projekt do portfolio AI/ML — [GitHub](https://github.com/TWOJ-USERNAME/pdf-rag-chatbot)*
"""
)

# =============================== KONFIGURACJA API
load_dotenv()
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =============================== ROUTER MODELI

# ========================================
# NA GŁÓWNYM POZIOMIE (wklej PRZED funkcją)
# ========================================

# Sprawdzenie czy GROQ_API_KEY jest ustawiony
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error(
        "❌ **Brak GROQ_API_KEY!**\n\n"
        "Ustaw klucz w pliku `.env` lub Streamlit Cloud Secrets:\n"
        "```\n"
        "GROQ_API_KEY=twój_klucz_tutaj\n"
        "```\n\n"
        "🔑 Uzyskaj darmowy klucz: https://console.groq.com"
    )
    st.stop()

# Teraz dopiero twórz klienta
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# FUNKCJA (wklej ZAMIAST starej funkcji)

def get_llm_client():
    key = st.session_state.get("selected_model", "⚡ llama-3.1-8b (szybki)")
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    if key == "⚡ llama-3.1-8b (szybki)":
        model_name = "llama-3.1-8b-instant"
    elif key == "⚖️ qwen3-32b (zbalansowany)":
        model_name = "qwen/qwen3-32b"
    elif key == "🏆 llama-3.3-70b (najlepszy)":
        model_name = "llama-3.3-70b-versatile"
    elif key == "🖼️ llama-4-maverick (multimodalny)":
        model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"
    else:
        model_name = "llama-3.1-8b-instant"
    return client, model_name

# FUNKCJE POMOCNICZE


def rerank_by_keyword_overlap(question, docs):
    """Prosty reranking: liczy overlap słów pytania z każdym chunkiem i wybiera najlepsze."""
    if not docs:
        return []

    import re

    top_k = st.session_state.get("top_k_context", 3)

    q_tokens = re.findall(r"\w+", question.lower())
    q_tokens = [t for t in q_tokens if len(t) > 2]

    scored = []
    for doc in docs:
        text = doc.page_content.lower()
        score = sum(1 for t in q_tokens if t in text)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked_docs = [d for s, d in scored]

    return reranked_docs[:top_k]


def process_multiple_pdfs(file_streams):
    """Wczytanie wielu PDF, pocięcie na chunki i zbudowanie wektorowej bazy Chroma."""
    all_chunks = []
    pdf_names = []
    images_map = {}  # np. {(pdf_name, page): [paths...]}
    for pdf_file in file_streams:
        pdf_names.append(pdf_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(pdf_file.read())
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        chunk_size = st.session_state.get("chunk_size", 1000)
        chunk_overlap = st.session_state.get("chunk_overlap", 200)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["source_pdf"] = pdf_file.name

        all_chunks.extend(chunks)
        os.unlink(tmp_path)

    st.info(f"✅ Przetworzono: {len(file_streams)} plików, {len(all_chunks)} fragmentów")

    db_dir = tempfile.mkdtemp()
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )
    return vector_store, all_chunks, pdf_names

def process_multiple_pdfs_multimodal(file_streams):
    """
    Multimodalne przetwarzanie skanowanych PDF:
    - OCR wyciąga tekst ze stron (Tesseract)
    - PyMuPDF wyciąga obrazy stron
    - tekst trafia do chunków → Chroma
    - obrazy trafiają do metadanych i session_state
    """
    all_chunks = []
    pdf_names = []

    for pdf_file in file_streams:
        pdf_names.append(pdf_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(pdf_file.read())

        # 1) OCR – wyciągnij tekst ze skanu
        ocr_results = extract_text_with_ocr(tmp_path)
        ocr_dict = {page: text for page, text in ocr_results}  # {1: "...", 2: "..."}

        # 2) Wyciągnij obrazy stron
        images_dir = os.path.join("pdf_images", os.path.splitext(pdf_file.name)[0])
        page_to_imgs = extract_images_per_page(tmp_path, images_dir)

        # 3) Zamiast PyPDFLoader, zbuduj dokumenty z OCR'u
        # (bo PyPDFLoader nic nie daje ze skanów)
        documents = []
        for page_num in sorted(ocr_dict.keys()):
            ocr_text = ocr_dict[page_num]
            if ocr_text.strip():  # jeśli OCR coś znalazł
                doc = type('Document', (), {})()  # dummy Document
                doc.page_content = ocr_text
                doc.metadata = {"page": page_num, "source": pdf_file.name}
                documents.append(doc)

        # 4) Chunkuj tekst z OCR
        chunk_size = st.session_state.get("chunk_size", 1000)
        chunk_overlap = st.session_state.get("chunk_overlap", 200)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        # 5) Podpinaj obrazy do metadanych (bezpieczne dla Chroma)
        for chunk in chunks:
            chunk.metadata["source_pdf"] = pdf_file.name
            page = chunk.metadata.get("page", None)
            imgs = page_to_imgs.get(page, []) if page is not None else []

            if "page_images" not in st.session_state:
                st.session_state["page_images"] = {}
            st.session_state["page_images"][(pdf_file.name, page)] = imgs

            chunk.metadata["images_on_page_count"] = len(imgs)
            if imgs:
                chunk.metadata["image_preview"] = imgs[0]
            else:
                chunk.metadata["image_preview"] = None

        all_chunks.extend(chunks)
        os.unlink(tmp_path)

    st.info(f"✅ [MULTIMODAL + OCR] Przetworzono: {len(file_streams)} plików, {len(all_chunks)} fragmentów")

    db_dir = tempfile.mkdtemp()
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )
    return vector_store, all_chunks, pdf_names

def bm25_scores_for_docs(question, docs):
    """Liczy BM25 score dla listy doc.page_content względem pytania."""
    if not docs:
        return []

    import re

    corpus = []
    for doc in docs:
        tokens = re.findall(r"\w+", doc.page_content.lower())
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)

    q_tokens = re.findall(r"\w+", question.lower())
    q_tokens = [t for t in q_tokens if len(t) > 2]

    scores = bm25.get_scores(q_tokens)
    return list(scores)


def hybrid_bm25_vector_rerank(question, docs, return_scores=False):
    """Łączy kolejność wektorową (Chroma) z BM25 i zwraca posortowane docy + score'y."""
    if not docs:
        if return_scores:
            return [], [], [], []
        return []

    bm25_scores = bm25_scores_for_docs(question, docs)
    n = len(docs)

    vector_scores = [n - i for i in range(n)]

    alpha = 0.5
    combined = []
    for idx, (doc, bm, vec) in enumerate(zip(docs, bm25_scores, vector_scores)):
        h_score = alpha * bm + (1 - alpha) * vec
        combined.append((h_score, bm, vec, doc))

    combined.sort(key=lambda x: x[0], reverse=True)

    docs_sorted = [c[3] for c in combined]
    hybrid_scores_sorted = [c[0] for c in combined]
    bm25_scores_sorted = [c[1] for c in combined]
    vector_scores_sorted = [c[2] for c in combined]

    if return_scores:
        return docs_sorted, hybrid_scores_sorted, bm25_scores_sorted, vector_scores_sorted
    return docs_sorted


def ask_llm_RAG(question, vector_store, selected_pdfs, detail_level="Standard"):
    """Zapytanie do LLM z kontekstem z wybranych PDF (RAG + historia czatu)."""

    if st.session_state.get("llm_calls_count", 0) >= LLM_CALLS_LIMIT:
        info_msg = (
            f"Osiągnięto limit {LLM_CALLS_LIMIT} zapytań do LLM w tej sesji. "
            "Odśwież stronę, aby rozpocząć nową sesję lub zmniejsz liczbę pytań."
        )
        return info_msg, [], {
            "top_hybrid": 0.0,
            "threshold": 3.0,
            "is_low_quality": True,
        }

    chat_history = st.session_state.get("chat_history", [])
    n = 3
    history_block = ""
    for turn in chat_history[-n:]:
        history_block += f"User: {turn['question']}\nAsystent: {turn['answer']}\n"

    top_k = st.session_state.get("top_k_context", 3)

    k_search = top_k * 3
    all_docs = vector_store.similarity_search(question, k=k_search)

    if selected_pdfs:
        filtered_docs = [
            doc for doc in all_docs if doc.metadata.get("source_pdf") in selected_pdfs
        ]
    else:
        filtered_docs = all_docs

    hybrid_docs, hybrid_scores, bm25_scores, vector_scores = hybrid_bm25_vector_rerank(
        question, filtered_docs, return_scores=True
    )

    relevant_docs = rerank_by_keyword_overlap(question, hybrid_docs)

    if hybrid_scores:
        top_hybrid = hybrid_scores[0]
    else:
        top_hybrid = 0.0

    QUALITY_THRESHOLD = 3.0
    retrieval_quality = {
        "top_hybrid": float(top_hybrid),
        "threshold": QUALITY_THRESHOLD,
        "is_low_quality": bool(top_hybrid < QUALITY_THRESHOLD or len(relevant_docs) == 0),
    }

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    # dodatkowa reguła dla pytań typu "czy ..."
    is_yes_no = question.strip().lower().startswith("czy ")
    extra_rule = ""
    if is_yes_no:
        extra_rule = (
            "Dla pytań zaczynających się od 'czy' odpowiedź ma mieć maksymalnie 3 zdania.\n"
        )
    # styl wyjaśnienia zależnie od poziomu
    base_language_rule = (
    "Pisz po polsku, poprawnie językowo, bez błędów ortograficznych i gramatycznych.\n"
    "Unikaj potocznych, dziwnych lub sztucznie brzmiących sformułowań.\n"
    "Nie dopowiadaj faktów spoza dokumentów – jeśli czegoś nie ma w treści, powiedz wprost, że tego nie widzisz.\n"
    )

    # styl wyjaśnienia zależnie od poziomu
    if detail_level.startswith("ELI5"):
        style_rule = (
            base_language_rule +
            "Wyjaśniaj prosto, krótkimi zdaniami, ale zachowaj naturalny, dorosły język.\n"
        )
    elif detail_level.startswith("Expert"):
        style_rule = (
            base_language_rule +
            "Wyjaśniaj precyzyjnie, używaj poprawnej terminologii, ale unikaj nadmiernie złożonych zdań.\n"
        )
    else:
        style_rule = (
            base_language_rule +
            "Wyjaśniaj jasno i zwięźle, z umiarkowanym poziomem szczegółowości.\n"
        )

    prompt = f"""Jesteś pomocnym asystentem analizującym dokumenty PDF.

{history_block}
Pytanie użytkownika:
{question}

Poniżej znajdziesz fragmenty dokumentów PDF do wykorzystania jako kontekst:
{context}

{style_rule}{extra_rule}Zasady odpowiedzi:
1. Najpierw odpowiedz jednym, maksymalnie dwoma zdaniami na główne pytanie.
2. Następnie podaj maksymalnie 2–3 krótkie punkty z odwołaniem do dokumentu (jeśli są istotne).
3. Nie wklejaj długich cytatów z artefaktami OCR; jeśli trzeba, parafrazuj.
4. Jeśli w dokumentach nie ma jednoznacznej odpowiedzi (np. nie widać wykresów lub brak danej informacji), powiedz to wprost i nie zgaduj.
5. Używaj poprawnego, naturalnego języka polskiego (bez sztucznie uproszczonych, błędnych form).
6. Dbaj o poprawną interpunkcję i naturalny szyk zdania.

Odpowiedź po polsku:
"""
    client, model_name = get_llm_client()
    # krótszy limit tokenów specjalnie dla quizu
    if "Learning quiz" in question:
        max_tokens = 700
    else:
        max_tokens = 1024
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
        st.session_state["llm_calls_count"] += 1
    except Exception as e:
        answer = f"Błąd API: {str(e)}"
        retrieval_quality = {
            "top_hybrid": 0.0,
            "threshold": QUALITY_THRESHOLD,
            "is_low_quality": True,
        }

    return answer, relevant_docs, retrieval_quality

def extract_numerical_data(answer):
    """Wyciąga pary 'nazwa - wartość' z odpowiedzi modelu."""
    lines = answer.splitlines()
    data = []
    for line in lines:
        if " - " in line and any(char.isdigit() for char in line):
            try:
                parts = line.split(" - ")
                if len(parts) == 2:
                    name = parts[0].strip()
                    value_str = parts[1].strip()
                    import re

                    numbers = re.findall(r"-?\d+(?:\.\d+)?", value_str)
                    if numbers:
                        value = float(numbers[0])
                        data.append((name, value))
            except Exception:
                continue
    return data


def generate_chat_history_txt(chat_history):
    import io

    output = io.StringIO()
    for i, chat in enumerate(chat_history):
        output.write(f"--- Pytanie {i+1} ---\n")
        output.write(f"Użytkownik: {chat['question']}\n")
        output.write(f"Asystent: {chat['answer']}\n")
        output.write("Źródła:\n")
        for j, doc in enumerate(chat["docs"]):
            pdfname = doc.metadata.get("source_pdf", "Nieznany PDF")
            output.write(
                f"    {pdfname} – Fragment {j+1}: {doc.page_content[:300]}...\n"
            )
        output.write("\n")
    return output.getvalue().encode("utf-8")


def generate_chat_history_csv(chat_history):
    rows = []
    for chat in chat_history:
        sources = "; ".join(
            [
                f"{doc.metadata.get('source_pdf', 'Nieznany PDF')} – {doc.page_content[:120]}..."
                for doc in chat["docs"]
            ]
        )
        rows.append(
            {
                "Pytanie": chat["question"],
                "Odpowiedź": chat["answer"],
                "Źródła": sources,
            }
        )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf_insights(chunks, pdf_names):
    """Generuje statystyki i insights z zaindeksowanych dokumentów."""
    insights = {
        "total_pdfs": len(pdf_names),
        "total_chunks": len(chunks),
        "pdfs_info": {},
    }

    for pdf_name in pdf_names:
        pdf_chunks = [c for c in chunks if c.metadata.get("source_pdf") == pdf_name]
        insights["pdfs_info"][pdf_name] = {
            "chunk_count": len(pdf_chunks),
            "estimated_pages": len(pdf_chunks) // 3 if len(pdf_chunks) > 0 else 1,
            "content_sample": pdf_chunks[0].page_content[:200] if pdf_chunks else "",
        }

    return insights


def display_insights_dashboard(insights):
    """Wyświetla dashboard z insights w Streamlit."""
    st.subheader("📊 Podsumowanie zaindeksowanych dokumentów")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 Liczba plików", insights["total_pdfs"])
    with col2:
        st.metric("📄 Liczba fragmentów", insights["total_chunks"])
    with col3:
        avg_chunks = (
            insights["total_chunks"] // insights["total_pdfs"]
            if insights["total_pdfs"] > 0
            else 0
        )
        st.metric("📊 Średnio fragmentów/plik", avg_chunks)

    st.markdown("---")
    st.subheader("📋 Szczegóły każdego dokumentu:")

    for pdf_name, info in insights["pdfs_info"].items():
        with st.expander(f"📖 {pdf_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Fragmentów:** {info['chunk_count']}")
            with col2:
                st.write(f"**Szacunkowe strony:** ~{info['estimated_pages']}")
            st.write("**Pierwsze 200 znaków:**")
            st.text(info["content_sample"])


def extract_main_topics(chunks):
    """Używa LLM do wyodrębnienia głównych tematów z dokumentów."""
    if not chunks:
        return "Brak danych do analizy tematów."

    sample_chunks = random.sample(chunks, min(5, len(chunks)))
    sample_text = "\n\n".join([c.page_content for c in sample_chunks])

    client, model_name = get_llm_client()

    prompt = f"""Na podstawie poniższych fragmentów dokumentów wypisz 5 głównych tematów/zagadnień
poruszanych w dokumentach w jednej linii, oddzielone przecinkami:

{sample_text}

Główne tematy:"""

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        topics = completion.choices[0].message.content
    except Exception as e:
        topics = f"Błąd przy ekstrakcji tematów: {str(e)}"

    return topics


def generate_document_summary(chunks):
    """Generuje krótkie podsumowanie całego korpusu dokumentów."""
    if not chunks:
        return "Brak danych do podsumowania."

    step = max(1, len(chunks) // 3)
    sample_chunks = chunks[::step][:5]
    sample_text = "\n\n".join([c.page_content[:300] for c in sample_chunks])

    client, model_name = get_llm_client()

    prompt = f"""Stwórz bardzo krótkie (2-3 zdania) podsumowanie zawartości tych dokumentów:

{sample_text}

Podsumowanie:"""

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        summary = completion.choices[0].message.content
    except Exception as e:
        summary = f"Błąd przy generowaniu podsumowania: {str(e)}"

    return summary


# =============================== ZAKŁADKI
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📂 Upload PDF", "💬 Czat i Podsumowania", "📊 Wykresy", "🔍 Debug RAG", "🌐 Inne źródła"]
)

# =============================== Tab1: Upload PDF + INSIGHTS
with tab1:
    st.header("📂 Upload i indeksowanie dokumentów PDF")
    uploaded_files = st.file_uploader(
        "Przeciągnij i upuść pliki PDF lub kliknij aby wybrać (możesz kilka!)",
        type="pdf",
        accept_multiple_files=True,
        help="Obsługiwane formaty: PDF (do 200MB każdy)",
    )

    if uploaded_files:
        st.success(
            f"✅ Załadowano {len(uploaded_files)} plików: "
            + ", ".join(f.name for f in uploaded_files)
        )

        if st.button("🚀 Indeksuj dokumenty", type="primary", key="index_docs"):
            with st.spinner("🔄 Przetwarzam dokumenty..."):
                try:
                    vector_store, chunks, pdf_names = process_multiple_pdfs(
                        uploaded_files
                    )
                    st.session_state["vector_store"] = vector_store
                    st.session_state["chunks"] = chunks
                    st.session_state["pdf_names"] = pdf_names

                    st.success("🎉 Dokumenty gotowe do analizy!")

                    insights = generate_pdf_insights(chunks, pdf_names)
                    st.session_state["pdf_insights"] = insights
                    display_insights_dashboard(insights)

                    st.markdown("---")
                    st.subheader("🎯 Główne tematy w dokumentach:")
                    with st.spinner("Ekstraguję główne tematy..."):
                        topics = extract_main_topics(chunks)
                        st.info(f"**Tematy:** {topics}")

                    st.markdown("---")
                    st.subheader("📝 Szybkie podsumowanie korpusu dokumentów:")
                    with st.spinner("Generuję podsumowanie..."):
                        summary = generate_document_summary(chunks)
                        st.write(summary)

                except Exception as e:
                    st.error(f"❌ Błąd podczas przetwarzania: {str(e)}")
        if st.button("🚀 Indeksuj dokumenty (multimodalnie)", type="secondary", key="index_docs_mm"):
            with st.spinner("🔄 Przetwarzam dokumenty (multimodalnie)..."):
                try:
                    vector_store, chunks, pdf_names = process_multiple_pdfs_multimodal(
                    uploaded_files
                    )
                    st.session_state["vector_store"] = vector_store
                    st.session_state["chunks"] = chunks
                    st.session_state["pdf_names"] = pdf_names

                    st.success("🎉 Dokumenty multimodalne gotowe do analizy!")

                    insights = generate_pdf_insights(chunks, pdf_names)
                    st.session_state["pdf_insights"] = insights
                    display_insights_dashboard(insights)

                    st.markdown("---")
                    st.subheader("🎯 Główne tematy w dokumentach:")
                    with st.spinner("Ekstraguję główne tematy..."):
                        topics = extract_main_topics(chunks)
                        st.info(f"**Tematy:** {topics}")

                    st.markdown("---")
                    st.subheader("📝 Szybkie podsumowanie korpusu dokumentów:")
                    with st.spinner("Generuję podsumowanie..."):
                        summary = generate_document_summary(chunks)
                        st.write(summary)

                except Exception as e:
                    st.error(f"❌ Błąd podczas przetwarzania (multimodalnie): {str(e)}")
    if "pdf_insights" in st.session_state:
        st.markdown("---")
        display_insights_dashboard(st.session_state["pdf_insights"])

# =============================== Tab2: Czat, podsumowania, eksport historii
with tab2:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = ""
    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = "Ask (nowe pytanie)"
    # jeśli nie ma jeszcze vector_store, spróbuj zainicjować pusty
    if "vector_store" not in st.session_state:
        st.warning("⚠️ Najpierw załaduj i zindeksuj dokumenty w zakładce 'Upload PDF'"
                   "PDF w zakładce 'Upload PDF' lub URL/tekst w zakładce 'Inne źródła'."
        )
    else:
        selected_pdfs = st.multiselect(
            "Wybierz źródła do analizy (PDF/URL/tekst):",
            options=st.session_state.get("pdf_names", []),
            default=st.session_state.get("pdf_names", []),
            help="Możesz zadać pytanie dla wybranego PDF, URL lub surowego tekstu (albo kilku jednocześnie).",
        )

        mode = st.radio(
            "Tryb zapytania:",
            [
                "Ask (nowe pytanie)",
                "Refine (doprecyzuj poprzednią odpowiedź)",
                "Tasks (lista zadań z dokumentu)",
                "Learning quiz (pytania testowe)",
            ],
            index=0,
            help=(
                "Ask – normalne pytanie. Refine – doprecyzowanie ostatniej "
                "odpowiedzi. Tasks – lista zadań. Learning quiz – wygeneruj pytania testowe z dokumentów."
            ),
        )

        # POZIOM SZCZEGÓŁOWOŚCI ODPOWIEDZI
        detail_level = st.radio(
            "Poziom wyjaśnienia:",
            ["ELI5 (jak dla 5‑latka)", "Standard", "Expert"],
            index=1,
            help="Określ, jak prostego lub szczegółowego wyjaśnienia oczekujesz.",
        )
        st.session_state["detail_level"] = detail_level

        question = st.text_input("Twoje pytanie:", key="current_question")

        def handle_ask_refine_click(mode_label):
            st.session_state["last_question"] = st.session_state.get(
                "current_question", ""
            )
            st.session_state["last_mode"] = mode_label
            st.session_state["current_question"] = ""

        col1, col2 = st.columns([1, 4])
        with col1:
            st.button(
                "🤖 Zapytaj (Ask)",
                type="primary",
                disabled=not st.session_state.get("current_question", "").strip(),
                key="ask_button",
                on_click=handle_ask_refine_click,
                args=("Ask (nowe pytanie)",),
            )
        with col2:
            st.button(
                "🔁 Refine / Tasks",
                type="secondary",
                disabled=not st.session_state.get("current_question", "").strip(),
                key="refine_button",
                on_click=handle_ask_refine_click,
                args=(mode,),
            )

        if st.session_state.get("last_question"):
            last_mode = st.session_state.get("last_mode", "Ask (nowe pytanie)")
            user_question = st.session_state["last_question"]

            if last_mode.startswith("Refine"):
                if st.session_state["chat_history"]:
                    prev_answer = st.session_state["chat_history"][-1]["answer"]
                else:
                    prev_answer = ""
                composed_question = (
                    "Doprecyzuj poprzednią odpowiedź poniżej, biorąc pod uwagę nowe pytanie.\n\n"
                    f"[Poprzednia odpowiedź asystenta]:\n{prev_answer}\n\n"
                    f"[Nowe pytanie użytkownika]: {user_question}"
                )

            elif last_mode.startswith("Tasks"):
                composed_question = (
                    "Na podstawie dokumentów przygotuj listę konkretnych zadań do wykonania "
                    "(w punktach), odpowiadając na pytanie:\n\n"
                    f"{user_question}"
                )

            elif last_mode.startswith("Learning quiz"):
                composed_question = (
                    "Na podstawie przekazanych fragmentów dokumentów przygotuj quiz do nauki.\n"
                    "Wygeneruj 5–10 pytań testowych (np. jednokrotnego wyboru), "
                    "z poprawną odpowiedzią i 2–3 mylącymi odpowiedziami.\n"
                    "Pytania mają być po polsku, jasno sformułowane.\n\n"
                    f"Zakres / wskazówki użytkownika: {user_question}"
                )

            else:
                composed_question = user_question

            with st.spinner("🔍 Szukam odpowiedzi..."):
                answer, docs, retrieval_quality = ask_llm_RAG(
                    composed_question,
                    st.session_state["vector_store"],
                    selected_pdfs,
                    detail_level=st.session_state.get("detail_level", "Standard"),
                )

            st.session_state["chat_history"].append(
                {
                    "question": user_question,
                    "answer": answer,
                    "docs": docs,
                    "mode": last_mode,
                    "retrieval_quality": retrieval_quality,
                }
            )

            st.session_state["last_question"] = ""

        if st.session_state["chat_history"]:
            chat_txt = generate_chat_history_txt(st.session_state["chat_history"])
            chat_csv = generate_chat_history_csv(st.session_state["chat_history"])

            st.download_button(
                "📥 Eksportuj historię czatu (.txt)",
                data=chat_txt,
                file_name="historia_czatu.txt",
                mime="text/plain",
            )

            st.download_button(
                "📥 Eksportuj historię czatu (.csv)",
                data=chat_csv,
                file_name="historia_czatu.csv",
                mime="text/csv",
            )

        for i, chat in enumerate(reversed(st.session_state["chat_history"])):
            idx = len(st.session_state["chat_history"]) - i
            mode_label = chat.get("mode", "Ask (nowe pytanie)")

            with st.chat_message("user"):
                st.markdown(f"**[{mode_label}]** {chat['question']}")

            with st.chat_message("assistant"):
                st.markdown(chat["answer"])

                rq = chat.get("retrieval_quality")
                if rq and rq.get("is_low_quality"):
                    top_h = rq.get("top_hybrid", 0.0)
                    thr = rq.get("threshold", 0.0)
                    st.warning(
                        f"⚠️ Uwaga: kontekst dla tej odpowiedzi może być słabo dopasowany "
                        f"(hybrid score top chunku = {top_h:.2f}, próg jakości = {thr:.2f}). "
                        "Zweryfikuj w źródłach, zanim zaufasz odpowiedzi.",
                    )
                if rq:
                    st.caption(
                        f"Retrieval: top hybrid score = {rq.get('top_hybrid', 0.0):.2f}, "
                        f"próg = {rq.get('threshold', 0.0):.2f}"
                    )

                col_ok, col_bad = st.columns(2)
                with col_ok:
                    if st.button(
                        "✔️ Trafione",
                        key=f"ok_{idx}",
                    ):
                        st.session_state["llm_correct_count"] += 1
                with col_bad:
                    if st.button(
                        "❌ Chybione",
                        key=f"bad_{idx}",
                    ):
                        st.session_state["llm_incorrect_count"] += 1

                with st.expander(f"📚 Źródła dla pytania {idx}"):
                    for j, doc in enumerate(chat["docs"]):
                        pdfname = doc.metadata.get("source_pdf", "Nieznany PDF")
                        page_num = doc.metadata.get("page", "?")
                    
                        # Blok z tłem (highlight effect)
                        st.markdown(
                            f"""
                            <div style="background-color: #fff9e6; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0;">
                            <strong>📄 {pdfname}</strong><br>
                            <span style="color: #666;">Strona: <strong>{page_num}</strong></span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                        st.text_area(
                            f"Fragment {j+1}:",
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content,
                            height=100,
                            key=f"source_{i}_{j}",
                            disabled=True,
                        )
                        # WYŚWIETLANIE OBRAZÓW
                        img_path = doc.metadata.get("image_preview")
                        if img_path:
                            st.write("Podgląd obrazu z tej strony:")
                            st.image(img_path, use_column_width=True)

# =============================== Tab3: Wykresy
with tab3:
    if "vector_store" not in st.session_state:
        st.warning(
            "⚠️ Najpierw załaduj i zindeksuj dokumenty PDF w zakładce 'Upload PDF'"
        )
    else:
        st.header("📊 Automatyczne wykresy z danych")
        st.markdown(
            "Asystent spróbuje wyciągnąć dane liczbowe z dokumentów i przedstawić je w formie wykresu."
        )

        chart_type = st.selectbox(
            "Wybierz typ wykresu:",
            ["Wykres słupkowy", "Wykres kołowy", "Wykres liniowy"],
            key="chart_type_select",
        )

        if st.button("📊 Generuj wykres", type="primary"):
            with st.spinner("📊 Analizuję dane i tworzę wykres..."):
                data_question = """Znajdź wszystkie dane liczbowe w dokumentach i przedstaw je w formacie:
nazwa parametru - wartość liczbowa
Np.:
Sprzedaż Q1 - 1500
Koszty - 800
Zysk - 700"""
                answer, docs_tmp, rq_tmp = ask_llm_RAG(
                    data_question,
                    st.session_state["vector_store"],
                    st.session_state.get("pdf_names", []),
                )

                st.markdown("### 🔍 Znalezione dane:")
                st.text(answer)

                data = extract_numerical_data(answer)

                if data and len(data) > 0:
                    labels, values = zip(*data)
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if chart_type == "Wykres słupkowy":
                        ax.bar(
                            labels,
                            values,
                            color="skyblue",
                            edgecolor="navy",
                            alpha=0.7,
                        )
                        ax.set_ylabel("Wartości")
                        plt.xticks(rotation=45, ha="right")

                    elif chart_type == "Wykres kołowy":
                        ax.pie(
                            values,
                            labels=labels,
                            autopct="%1.1f%%",
                            startangle=90,
                        )
                        ax.axis("equal")

                    elif chart_type == "Wykres liniowy":
                        ax.plot(
                            labels,
                            values,
                            marker="o",
                            linewidth=2,
                            markersize=8,
                        )
                        ax.set_ylabel("Wartości")
                        plt.xticks(rotation=45, ha="right")

                    ax.set_title(
                        "Dane wyciągnięte z dokumentów PDF",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.tight_layout()
                    st.pyplot(fig)

                    df = pd.DataFrame(data, columns=["Parametr", "Wartość"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning(
                        "⚠️ Nie udało się wyciągnąć danych liczbowych z dokumentów. "
                        "Spróbuj z dokumentami zawierającymi tabele lub statystyki."
                    )

# =============================== Tab4: Debug RAG
with tab4:
    st.header("🔍 Debug RAG – top kontekst")

    if "vector_store" not in st.session_state:
        st.warning("⚠️ Najpierw załaduj i zindeksuj dokumenty w zakładce 'Upload PDF'.")
    else:
        pdf_options = st.session_state.get("pdf_names", [])
        selected_pdfs_debug = st.multiselect(
            "Wybierz dokument(y) do debugowania:",
            options=pdf_options,
            default=pdf_options,
            help="Ogranicz debug do wybranych PDF-ów lub pozostaw wszystkie.",
        )

        debug_question = st.text_input(
            "Pytanie do debugowania retrieval:",
            value=st.session_state.get("last_debug_question", ""),
            help="Pytanie, dla którego chcesz zobaczyć top-N chunków i score'y.",
        )

        debug_n = st.number_input(
            "Ile top chunków pokazać?",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )

        show_post_rerank = st.checkbox(
            "Pokaż także chunki po keyword-rerankingu (finalny kontekst do LLM)",
            value=True,
            help="Jeśli zaznaczone, zobaczysz też fragmenty, które faktycznie trafiają do promptu LLM.",
        )

        if st.button("🔁 Zresetuj licznik trafień dla tej sesji"):
            st.session_state["llm_correct_count"] = 0
            st.session_state["llm_incorrect_count"] = 0

        if st.button("🔎 Pokaż top chunków"):
            if not debug_question.strip():
                st.warning("Wpisz pytanie, aby zobaczyć wyniki.")
            else:
                st.session_state["last_debug_question"] = debug_question

                top_k = st.session_state.get("top_k_context", 3)
                k_search = max(debug_n, top_k * 3)

                all_docs = st.session_state["vector_store"].similarity_search(
                    debug_question, k=k_search
                )

                if selected_pdfs_debug:
                    filtered_docs = [
                        doc
                        for doc in all_docs
                        if doc.metadata.get("source_pdf") in selected_pdfs_debug
                    ]
                else:
                    filtered_docs = all_docs

                (
                    hybrid_docs,
                    hybrid_scores,
                    bm25_scores,
                    vector_scores,
                ) = hybrid_bm25_vector_rerank(
                    debug_question, filtered_docs, return_scores=True
                )

                st.subheader("📄 Top chunków wg hybrid score (PRZED keyword-rerankiem)")
                for i, (doc, h, bm, vec) in enumerate(
                    zip(
                        hybrid_docs[:debug_n],
                        hybrid_scores[:debug_n],
                        bm25_scores[:debug_n],
                        vector_scores[:debug_n],
                    ),
                    start=1,
                ):
                    pdfname = doc.metadata.get("source_pdf", "Nieznany PDF")
                    st.markdown(
                        f"**#{i} – {pdfname}**  \n"
                        f"Hybrid score: `{h:.3f}`  \n"
                        f"BM25 score: `{bm:.3f}`  \n"
                        f"Vector rank score: `{vec:.3f}`"
                    )
                    st.text_area(
                        f"[PRZED] Fragment {i}:",
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content,
                        height=120,
                        key=f"debug_before_{i}",
                    )
                    st.markdown("---")

                st.markdown("---")
                st.subheader("📊 Podsumowanie sesji (hyperparam + trafienia)")

                chunk_size = st.session_state.get("chunk_size", 1000)
                chunk_overlap = st.session_state.get("chunk_overlap", 200)
                top_k = st.session_state.get("top_k_context", 3)
                correct = st.session_state.get("llm_correct_count", 0)
                incorrect = st.session_state.get("llm_incorrect_count", 0)
                total = correct + incorrect
                accuracy = (correct / total) * 100 if total > 0 else 0.0

                summary_df = pd.DataFrame(
                    [
                        {
                            "chunk_size": chunk_size,
                            "overlap": chunk_overlap,
                            "top_k_context": top_k,
                            "trafione": correct,
                            "chybione": incorrect,
                            "accuracy_%": round(accuracy, 1),
                        }
                    ]
                )

                st.dataframe(summary_df, use_container_width=True)

                if show_post_rerank:
                    st.subheader(
                        "✅ Chunki, które trafiły do finalnego kontekstu (PO keyword-rerankingu)"
                    )

                    top_k_final = st.session_state.get("top_k_context", 3)

                    final_docs = rerank_by_keyword_overlap(
                        debug_question,
                        hybrid_docs,
                    )[:top_k_final]

                    indices_map = {id(doc): idx for idx, doc in enumerate(hybrid_docs)}

                    for j, doc in enumerate(final_docs, start=1):
                        pdfname = doc.metadata.get("source_pdf", "Nieznany PDF")
                        idx_in_hybrid = indices_map.get(id(doc), -1)
                        h = hybrid_scores[idx_in_hybrid] if idx_in_hybrid >= 0 else 0.0
                        bm = bm25_scores[idx_in_hybrid] if idx_in_hybrid >= 0 else 0.0
                        vec = (
                            vector_scores[idx_in_hybrid] if idx_in_hybrid >= 0 else 0.0
                        )

                        st.markdown(
                            f"**#{j} – {pdfname}**  \n"
                            f"Hybrid score: `{h:.3f}`  \n"
                            f"BM25 score: `{bm:.3f}`  \n"
                            f"Vector rank score: `{vec:.3f}`  \n"
                            f"Pozycja w liście hybrid: `{idx_in_hybrid + 1}`"
                        )
                        st.text_area(
                            f"[PO] Fragment {j}:",
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content,
                            height=120,
                            key=f"debug_after_{j}",
                        )
                        st.markdown("---")
# =============================== Tab5: Inne źródła (URL / tekst)
with tab5:
    st.header("🌐 Inne źródła wiedzy (URL / tekst)")

    url_input = st.text_input(
        "Podaj URL do zaindeksowania:",
        help="Np. artykuł, dokument online, wpis na blogu.",
        key="extra_url_input",
    )

    raw_text = st.text_area(
        "Lub wklej surowy tekst do zaindeksowania:",
        height=200,
        help="Tekst zostanie potraktowany jak dodatkowy dokument.",
        key="extra_text_input",
    )

    if st.button("➕ Dodaj źródła do bazy", type="primary"):
        # Upewnij się, że istnieje vector_store
        if "vector_store" not in st.session_state:
            db_dir = tempfile.mkdtemp()
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            st.session_state["vector_store"] = Chroma(
                embedding_function=embeddings,
                persist_directory=db_dir,
            )
            st.session_state["chunks"] = []
            st.session_state["pdf_names"] = []

        new_docs = []

        # ---- URL ----
        if url_input.strip():
            from langchain_community.document_loaders import WebBaseLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            loader = WebBaseLoader([url_input.strip()])  # lista URL
            url_docs = loader.load()
            for d in url_docs:
                d.metadata["source_pdf"] = url_input.strip()
                d.metadata["source_url"] = url_input.strip()

            # CHUNKOWANIE tak jak dla PDF
            chunk_size = st.session_state.get("chunk_size", 1000)
            chunk_overlap = st.session_state.get("chunk_overlap", 200)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            url_chunks = splitter.split_documents(url_docs)
            new_docs.extend(url_chunks)
            st.session_state["chunks"].extend(url_chunks)

        # ---- Surowy tekst ----
        if raw_text.strip():
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_doc = Document(
                page_content=raw_text.strip(),
                metadata={
                    "source_pdf": "Surowy tekst użytkownika",
                    "source_text": "user_paste",
                },
            )

            # opcjonalne chunkowanie, jak dla PDF
            chunk_size = st.session_state.get("chunk_size", 1000)
            chunk_overlap = st.session_state.get("chunk_overlap", 200)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            text_chunks = splitter.split_documents([text_doc])
            new_docs.extend(text_chunks)
            st.session_state["chunks"].extend(text_chunks)

        # ---- Dodanie do vector_store + aktualizacja pdf_names ----
        if new_docs:
            st.session_state["vector_store"].add_documents(new_docs)

            extra_names = set(
                d.metadata.get("source_pdf", "Inne źródło") for d in new_docs
            )
            pdf_names = st.session_state.get("pdf_names", [])
            for name in extra_names:
                if name not in pdf_names:
                    pdf_names.append(name)
            st.session_state["pdf_names"] = pdf_names

            st.success(f"Dodano {len(new_docs)} nowych fragmentów do bazy.")
        else:
            st.info("Nie podano ani URL, ani tekstu – brak nowych źródeł.")

# =============================== STOPKA PORTFOLIO
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>PDF RAG Chatbot AI</strong> - Projekt portfolio wykorzystujący najnowsze technologie AI</p>
    <p>🛠️ Technologie: Streamlit • LangChain • Hugging Face • OpenAI-compatible • RAG • Vector Database</p>
    <p>👨‍💻 Stworzony przez: <a href="https://github.com/TWOJ-USERNAME" target="_blank">GitHub</a></p>
</div>
""",
    unsafe_allow_html=True,
)
