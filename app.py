import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import fitz  # type: ignore
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

# ============================================
# Automatyczna konfiguracja Tesseract/Poppler dla różnych systemów
# ============================================
if platform.system() == "Windows":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    if os.path.exists(poppler_path):
        os.environ["PATH"] += f";{poppler_path}"

# Opcjonalnie użyj zmiennych środowiskowych z .env
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

POPPLER_PATH = os.getenv("POPPLER_PATH", "")
if POPPLER_PATH:
    sep = ";" if platform.system() == "Windows" else ":"
    os.environ["PATH"] += f"{sep}{POPPLER_PATH}"


def extract_images_per_page(pdf_path: str, output_dir: str):
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
            img_filename = f"pdf_img_p{page_index+1}_i{img_index}.{ext}"
            img_path = os.path.join(output_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            paths.append(img_path)
        if paths:
            page_to_imgs[page_index + 1] = paths
    doc.close()
    return page_to_imgs


def extract_text_with_ocr(pdf_path: str):
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        ocr_results = []
        for page_index, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='pol+eng')
            ocr_results.append((page_index + 1, text))
        return ocr_results
    except Exception as e:
        st.warning(f"OCR error: {str(e)}")
        return []


st.set_page_config(
    page_title="PDF RAG Chatbot AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "llm_calls_count" not in st.session_state:
    st.session_state["llm_calls_count"] = 0
LLM_CALLS_LIMIT = 50

if "llm_correct_count" not in st.session_state:
    st.session_state["llm_correct_count"] = 0
if "llm_incorrect_count" not in st.session_state:
    st.session_state["llm_incorrect_count"] = 0

with st.sidebar:
    st.header("⚙️ Settings")
    model_key = st.selectbox(
        "Select model",
        ["🚀 llama-3.1-8b (fast)", "⚖️ qwen3-32b (balanced)", "🎯 llama-3.3-70b (best)", "🖼️ llama-3.2-90b (vision)"],
        index=0,
        help="All models run through Groq API."
    )
    top_k_context = st.selectbox("How many documents in context?", [1, 2, 3, 4, 5, 6], index=2)
    st.markdown("**Chunking (tunable)**")
    chunk_size = st.slider("Chunk size (characters)", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Overlap between chunks (characters)", min_value=0, max_value=500, value=200, step=50)
    st.markdown("---")
    used = st.session_state.get("llm_calls_count", 0)
    st.write(f"**Queries in this session:** {used}/{LLM_CALLS_LIMIT}")
    correct = st.session_state.get("llm_correct_count", 0)
    incorrect = st.session_state.get("llm_incorrect_count", 0)
    total_labeled = correct + incorrect
    if total_labeled > 0:
        acc = 100 * (correct / total_labeled)
        st.write(f"✅ Accurate: {correct}")
        st.write(f"❌ Inaccurate: {incorrect}")
        st.write(f"📊 Response accuracy: {acc:.1f}%")
    else:
        st.write("No rated responses yet.")
    if used >= LLM_CALLS_LIMIT:
        st.warning("You've reached the maximum query limit for this session. Refresh the page to start a new session.")

st.session_state["selected_model"] = model_key
st.session_state["top_k_context"] = top_k_context
st.session_state["chunk_size"] = chunk_size
st.session_state["chunk_overlap"] = chunk_overlap

st.markdown("""<style>
    .MainMenu, footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stSidebarNav"] {background: #f1f5fa;}
    [data-testid="stSidebar"] {display: block !important; visibility: visible !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    .stTitle {color: #1f77b4;}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 20px; border: none; padding: 0.5rem 1rem; font-weight: bold;}
    .stButton>button:hover {background-color: #0d5aa7; color: white;}
</style>""", unsafe_allow_html=True)

st.title("📄 PDF RAG Chatbot AI")
st.markdown("""
**Intelligent assistant for analyzing PDF documents using RAG (Retrieval-Augmented Generation)**

**Features:**
- Upload and automatic indexing of PDFs (multi-PDF!)
- Q&A based on content of selected documents
- Automatic chapter summaries and full document summaries
- Chart generation from text data
- Insights: number of files, chunks, estimated pages, main topics
- Powered by latest LLM models

**Portfolio AI/ML project** → [GitHub](https://github.com/Lukkaszsz/pdf-rag-chatbot)
""")

load_dotenv()
# Ustaw USER_AGENT dla web scraperów
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; PDFRagChatbot/1.0)"
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Sprawdzenie czy GROQ_API_KEY jest ustawiony
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("**Missing GROQ_API_KEY!** Set the key in `.env` file or Streamlit Cloud Secrets.\nGet a free key: https://console.groq.com")
    st.stop()

# Teraz dopiero twórz klienta
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)


def get_llm_client():
    """Zwraca (client, model_name) zależnie od wybranego modelu w UI."""
    key = st.session_state.get("selected_model", "🚀 llama-3.1-8b (fast)")
    c = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
    models = {
        "🚀 llama-3.1-8b (fast)": "llama-3.1-8b-instant",
        "⚖️ qwen3-32b (balanced)": "qwen/qwen3-32b",
        "🎯 llama-3.3-70b (best)": "llama-3.3-70b-versatile",
        "🖼️ llama-3.2-90b (vision)": "llama-3.2-90b-vision-preview",
    }
    return c, models.get(key, "llama-3.1-8b-instant")


def rerank_by_keyword_overlap(question, docs):
    """Prosty reranking: liczy overlap słów pytania z każdym chunkiem i wybiera najlepsze."""
    if not docs:
        return []
    import re
    top_k = st.session_state.get("top_k_context", 3)
    q_tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    scored = [(sum(1 for t in q_tokens if t in doc.page_content.lower()), doc) for doc in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for (s, d) in scored][:top_k]


def process_multiple_pdfs(file_streams):
    """Wczytanie wielu PDF, pocięcie na chunki i zbudowanie wektorowej bazy (Chroma)."""
    all_chunks, pdf_names = [], []
    for pdf_file in file_streams:
        pdf_names.append(pdf_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get("chunk_size", 1000),
            chunk_overlap=st.session_state.get("chunk_overlap", 200)
        ).split_documents(PyPDFLoader(tmp_path).load())
        for c in chunks:
            c.metadata["source_pdf"] = pdf_file.name
        all_chunks.extend(chunks)
        os.unlink(tmp_path)
    st.info(f"Processed {len(file_streams)} files, {len(all_chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=tempfile.mkdtemp())
    return vectorstore, all_chunks, pdf_names


def process_multiple_pdfs_multimodal(file_streams):
    """Multimodalne przetwarzanie skanowanych PDF (OCR + obrazy)."""
    all_chunks, pdf_names = [], []
    for pdf_file in file_streams:
        pdf_names.append(pdf_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        ocr_results = extract_text_with_ocr(tmp_path)
        ocr_dict = {page: text for (page, text) in ocr_results}
        images_dir = os.path.join("pdf_images", os.path.splitext(pdf_file.name)[0])
        page_to_imgs = extract_images_per_page(tmp_path, images_dir)
        documents = []
        for page_num in sorted(ocr_dict.keys()):
            ocr_text = ocr_dict[page_num]
            if ocr_text.strip():
                doc = type("Document", (), {})()
                doc.page_content = ocr_text
                doc.metadata = {"page": page_num, "source": pdf_file.name}
                documents.append(doc)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get("chunk_size", 1000),
            chunk_overlap=st.session_state.get("chunk_overlap", 200)
        ).split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source_pdf"] = pdf_file.name
            page = chunk.metadata.get("page", None)
            imgs = page_to_imgs.get(page, []) if page is not None else []
            if "page_images" not in st.session_state:
                st.session_state["page_images"] = {}
            st.session_state["page_images"][(pdf_file.name, page)] = imgs
            chunk.metadata["images_on_page_count"] = len(imgs)
            chunk.metadata["image_preview"] = imgs[0] if imgs else None
        all_chunks.extend(chunks)
        os.unlink(tmp_path)
    st.info(f"MULTIMODAL (OCR): Processed {len(file_streams)} files, {len(all_chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=tempfile.mkdtemp())
    return vectorstore, all_chunks, pdf_names


def bm25_scores_for_docs(question, docs):
    """Liczy BM25 score dla listy doc.page_content względem pytania."""
    if not docs:
        return []
    import re
    corpus = [re.findall(r"\w+", doc.page_content.lower()) for doc in docs]
    bm25 = BM25Okapi(corpus)
    q_tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    return list(bm25.get_scores(q_tokens))


def hybrid_bm25_vector_rerank(question, docs, return_scores=False):
    """Łączy kolejność wektorową (Chroma) z BM25 i zwraca posortowane docy + score'y."""
    if not docs:
        return ([], [], [], []) if return_scores else []
    bm25_scores = bm25_scores_for_docs(question, docs)
    n = len(docs)
    vector_scores = [n - i for i in range(n)]
    combined = sorted(
        [(0.5 * bm + 0.5 * vec, bm, vec, doc) for doc, bm, vec in zip(docs, bm25_scores, vector_scores)],
        key=lambda x: x[0], reverse=True
    )
    if return_scores:
        return [c[3] for c in combined], [c[0] for c in combined], [c[1] for c in combined], [c[2] for c in combined]
    return [c[3] for c in combined]


def ask_llm_RAG(question, vectorstore, selected_pdfs, detail_level="Standard"):
    """Zapytanie do LLM z kontekstem z wybranych PDF (RAG) + historia czatu."""
    if st.session_state.get("llm_calls_count", 0) >= LLM_CALLS_LIMIT:
        return f"You've reached the limit of {LLM_CALLS_LIMIT} LLM queries. Refresh the page to start a new session.", [], {"top_hybrid": 0.0, "threshold": 3.0, "is_low_quality": True}
    chat_history = st.session_state.get("chat_history", [])
    history_block = "".join([f"\nUser: {t['question']}\n{t['answer']}\n" for t in chat_history[-3:]])
    top_k = st.session_state.get("top_k_context", 3)
    all_docs = vectorstore.similarity_search(question, k=top_k + 3)
    filtered_docs = [doc for doc in all_docs if doc.metadata.get("source_pdf") in selected_pdfs] if selected_pdfs else all_docs
    hybrid_docs, hybrid_scores, bm25_scores, vector_scores = hybrid_bm25_vector_rerank(question, filtered_docs, return_scores=True)
    relevant_docs = rerank_by_keyword_overlap(question, hybrid_docs)
    top_hybrid = hybrid_scores[0] if hybrid_scores else 0.0
    QUALITY_THRESHOLD = 3.0
    retrieval_quality = {"top_hybrid": float(top_hybrid), "threshold": QUALITY_THRESHOLD, "is_low_quality": bool(top_hybrid < QUALITY_THRESHOLD or len(relevant_docs) == 0)}
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    extra_rule = "\nFor yes/no questions, the answer should be max 3 sentences." if question.strip().lower().startswith("czy") else ""
    baserule = "Respond in the same language as the user's question. Write correctly, no spelling or grammar errors. Don't make up facts outside documents."
    style_rules = {"ELI5": baserule + " Explain simply, short sentences.", "Expert": baserule + " Use correct terminology."}
    style_rule = style_rules.get(detail_level.split()[0], baserule + " Explain clearly and concisely.")
    prompt = f"""You are a helpful assistant analyzing PDF documents.\n{history_block}\nUser question: {question}\n\nPDF context:\n{context}\n\n{style_rule}{extra_rule}\n\nResponse rules:\n1. Answer the main question in 1-2 sentences.\n2. Max 2-3 bullet points with document reference if relevant.\n3. Don't paste long OCR quotes – paraphrase.\n4. If answer not in documents, say so clearly.\n5. Use correct, natural language.\n\nAnswer in the same language as the question:"""
    client, model_name = get_llm_client()
    max_tokens = 700 if "Learning quiz" in question else 1024
    try:
        completion = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        answer = completion.choices[0].message.content
        st.session_state["llm_calls_count"] += 1
    except Exception as e:
        answer = f"API error: {str(e)}"
        retrieval_quality = {"top_hybrid": 0.0, "threshold": QUALITY_THRESHOLD, "is_low_quality": True}
    return answer, relevant_docs, retrieval_quality


def extract_numerical_data(answer):
    """Wyciąga pary (nazwa - wartość) z odpowiedzi modelu."""
    import re
    data = []
    for line in answer.split("\n"):
        if "-" in line and any(c.isdigit() for c in line):
            try:
                parts = line.split("-")
                if len(parts) == 2:
                    numbers = re.findall(r"-?\d+\.?\d*", parts[1])
                    if numbers:
                        data.append((parts[0].strip(), float(numbers[0])))
            except Exception:
                continue
    return data


def generate_chat_history_txt(chat_history):
    output = io.StringIO()
    for i, chat in enumerate(chat_history):
        output.write(f"\n--- Question {i+1} ---\nUser: {chat['question']}\nAssistant: {chat['answer']}\n\nSources:\n")
        for j, doc in enumerate(chat["docs"]):
            output.write(f"  {doc.metadata.get('source_pdf', 'Unknown')} – Fragment {j+1}: {doc.page_content[:300]}...\n")
    return output.getvalue().encode("utf-8")


def generate_chat_history_csv(chat_history):
    rows = [{"Question": c["question"], "Answer": c["answer"], "Sources": " | ".join([f"{d.metadata.get('source_pdf','?')}: {d.page_content[:120]}..." for d in c["docs"]])} for c in chat_history]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def generate_pdf_insights(chunks, pdf_names):
    """Generuje statystyki i insights z zaindeksowanych dokumentów."""
    insights = {"total_pdfs": len(pdf_names), "total_chunks": len(chunks), "pdfs_info": {}}
    for name in pdf_names:
        pc = [c for c in chunks if c.metadata.get("source_pdf") == name]
        insights["pdfs_info"][name] = {"chunk_count": len(pc), "estimated_pages": max(1, len(pc) // 3), "content_sample": pc[0].page_content[:200] if pc else ""}
    return insights


def display_insights_dashboard(insights):
    """Wyświetla dashboard z insights w Streamlit."""
    st.subheader("📊 Summary of indexed documents")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of files", insights["total_pdfs"])
    col2.metric("Number of chunks", insights["total_chunks"])
    col3.metric("Avg chunks/file", insights["total_chunks"] // insights["total_pdfs"] if insights["total_pdfs"] > 0 else 0)
    st.markdown("---")
    st.subheader("Details for each document")
    for name, info in insights["pdfs_info"].items():
        with st.expander(f"📄 {name}"):
            c1, c2 = st.columns(2)
            c1.write(f"**Chunks:** {info['chunk_count']}")
            c2.write(f"**Estimated pages:** {info['estimated_pages']}")
            st.text(info["content_sample"])


def extract_main_topics(chunks):
    """Używa LLM do wyodrębnienia głównych tematów z dokumentów."""
    if not chunks:
        return "No data."
    sample = "\n\n".join([c.page_content for c in random.sample(chunks, min(5, len(chunks)))])
    client, model_name = get_llm_client()
    try:
        completion = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": f"List 5 main topics from these document fragments, comma-separated:\n\n{sample}\n\nMain topics:"}], max_tokens=500)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def generate_document_summary(chunks):
    """Generuje krótkie podsumowanie całego korpusu dokumentów."""
    if not chunks:
        return "No data."
    step = max(1, len(chunks) // 3)
    sample = "\n\n".join([c.page_content[:300] for c in chunks[::step][:5]])
    client, model_name = get_llm_client()
    try:
        completion = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": f"Create a brief 2-3 sentence summary of these documents:\n\n{sample}\n\nSummary:"}], max_tokens=150)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================
# ZAKŁADKI
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Upload PDF", "💬 Chat & Summaries", "📊 Charts", "🔍 Debug RAG", "🌐 Other Sources"])

with tab1:
    st.header("📂 Upload and index PDF documents")
    uploaded_files = st.file_uploader("Drag & drop PDF files or click to select (multiple allowed!)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Loaded {len(uploaded_files)} files: {', '.join([f.name for f in uploaded_files])}")
        if st.button("🚀 Index documents", type="primary", key="index_docs"):
            with st.spinner("Processing documents..."):
                try:
                    vectorstore, chunks, pdf_names = process_multiple_pdfs(uploaded_files)
                    st.session_state.update({"vectorstore": vectorstore, "chunks": chunks, "pdf_names": pdf_names})
                    st.success("✅ Documents ready for analysis!")
                    insights = generate_pdf_insights(chunks, pdf_names)
                    st.session_state["pdf_insights"] = insights
                    display_insights_dashboard(insights)
                    st.markdown("---")
                    st.subheader("🔍 Main topics")
                    with st.spinner("Extracting topics..."):
                        st.info(f"**Topics:** {extract_main_topics(chunks)}")
                    st.markdown("---")
                    st.subheader("📝 Quick summary")
                    with st.spinner("Generating summary..."):
                        st.write(generate_document_summary(chunks))
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        if st.button("🖼️ Index documents (multimodal)", type="secondary", key="index_docs_mm"):
            with st.spinner("Processing documents multimodally..."):
                try:
                    vectorstore, chunks, pdf_names = process_multiple_pdfs_multimodal(uploaded_files)
                    st.session_state.update({"vectorstore": vectorstore, "chunks": chunks, "pdf_names": pdf_names})
                    st.success("✅ Multimodal documents ready!")
                    insights = generate_pdf_insights(chunks, pdf_names)
                    st.session_state["pdf_insights"] = insights
                    display_insights_dashboard(insights)
                    st.markdown("---")
                    st.subheader("🔍 Main topics")
                    with st.spinner("Extracting topics..."):
                        st.info(f"**Topics:** {extract_main_topics(chunks)}")
                    st.markdown("---")
                    st.subheader("📝 Quick summary")
                    with st.spinner("Generating summary..."):
                        st.write(generate_document_summary(chunks))
                except Exception as e:
                    st.error(f"Error during multimodal processing: {str(e)}")
    if "pdf_insights" in st.session_state:
        st.markdown("---")
        display_insights_dashboard(st.session_state["pdf_insights"])

with tab2:
    for key, default in [("chat_history", []), ("last_question", ""), ("current_question", ""), ("last_mode", "Ask (new question)")]:
        if key not in st.session_state:
            st.session_state[key] = default
    if "vectorstore" not in st.session_state:
        st.warning("First load and index documents in **Upload PDF** tab.")
    else:
        selected_pdfs = st.multiselect("Select sources for analysis", options=st.session_state.get("pdf_names", []), default=st.session_state.get("pdf_names", []))
        mode = st.radio("Query mode", ["Ask (new question)", "Refine (clarify previous)", "Tasks (list)", "Learning (quiz)"], index=0)
        detail_level = st.radio("Explanation level", ["ELI5 (like for a 5-year-old)", "Standard", "Expert"], index=1)
        st.session_state["detail_level"] = detail_level
        question = st.text_input("Your question", key="current_question")

        def handle_ask_refine_click(mode_label):
            st.session_state["last_question"] = st.session_state.get("current_question", "")
            st.session_state["last_mode"] = mode_label
            st.session_state["current_question"] = ""

        col1, col2 = st.columns([1, 4])
        with col1:
            st.button("Ask", type="primary", disabled=not st.session_state.get("current_question", "").strip(), key="ask_button", on_click=handle_ask_refine_click, args=("Ask (new question)",))
        with col2:
            st.button(f"Refine / {mode}", type="secondary", disabled=not st.session_state.get("current_question", "").strip(), key="refine_button", on_click=handle_ask_refine_click, args=(mode,))

        if st.session_state.get("last_question"):
            last_mode = st.session_state.get("last_mode", "Ask (new question)")
            user_question = st.session_state["last_question"]
            if last_mode.startswith("Refine"):
                prev = st.session_state["chat_history"][-1]["answer"] if st.session_state["chat_history"] else ""
                composed_question = f"Clarify previous answer:\n{prev}\n\nNew question:\n{user_question}"
            elif last_mode.startswith("Tasks"):
                composed_question = f"Based on documents, prepare a list of specific tasks:\n{user_question}"
            elif last_mode.startswith("Learning"):
                composed_question = f"Based on document fragments, prepare a learning quiz with 5-10 test questions (multiple choice, with correct answer).\n\nScope: {user_question}"
            else:
                composed_question = user_question
            with st.spinner("Searching for answer..."):
                answer, docs, retrieval_quality = ask_llm_RAG(composed_question, st.session_state["vectorstore"], selected_pdfs, detail_level=st.session_state.get("detail_level", "Standard"))
                st.session_state["chat_history"].append({"question": user_question, "answer": answer, "docs": docs, "mode": last_mode, "retrieval_quality": retrieval_quality})
                st.session_state["last_question"] = ""

        if st.session_state["chat_history"]:
            st.download_button("📥 Export chat (.txt)", data=generate_chat_history_txt(st.session_state["chat_history"]), file_name="chat_history.txt", mime="text/plain")
            st.download_button("📥 Export chat (.csv)", data=generate_chat_history_csv(st.session_state["chat_history"]), file_name="chat_history.csv", mime="text/csv")

        for i, chat in enumerate(reversed(st.session_state["chat_history"])):
            idx = len(st.session_state["chat_history"]) - i
            with st.chat_message("user"):
                st.markdown(f"**{chat.get('mode','Ask')}**\n\n{chat['question']}")
            with st.chat_message("assistant"):
                st.markdown(chat["answer"])
                rq = chat.get("retrieval_quality")
                if rq and rq.get("is_low_quality"):
                    st.warning(f"⚠️ Context may be poorly matched (hybrid score={rq.get('top_hybrid',0.0):.2f}, threshold={rq.get('threshold',0.0):.2f}). Verify sources.")
                if rq:
                    st.caption(f"Retrieval: top hybrid={rq.get('top_hybrid',0.0):.2f}, threshold={rq.get('threshold',0.0):.2f}")
                col_ok, col_bad = st.columns(2)
                with col_ok:
                    if st.button("✅ Accurate", key=f"ok_{idx}"):
                        st.session_state["llm_correct_count"] += 1
                with col_bad:
                    if st.button("❌ Inaccurate", key=f"bad_{idx}"):
                        st.session_state["llm_incorrect_count"] += 1
                with st.expander(f"📚 Sources for question {idx}"):
                    for j, doc in enumerate(chat["docs"]):
                        pdf_name = doc.metadata.get("source_pdf", "Unknown PDF")
                        page_num = doc.metadata.get("page", "?")
                        st.markdown(f"""<div style="background:#fff9e6;padding:10px;border-left:4px solid #ffc107;margin:10px 0;"><strong>{pdf_name}</strong><br><span style="color:#666;">Page: <strong>{page_num}</strong></span></div>""", unsafe_allow_html=True)
                        st.text_area(f"Fragment {j+1}", (doc.page_content[:500] + " ..." if len(doc.page_content) > 500 else doc.page_content), height=100, key=f"source_{i}_{j}", disabled=True)
                        img_path = doc.metadata.get("image_preview")
                        if img_path:
                            st.image(img_path, use_column_width=True)

with tab3:
    if "vectorstore" not in st.session_state:
        st.warning("First load and index PDF documents in **Upload PDF** tab.")
    else:
        st.header("📊 Automatic charts from data")
        chart_type = st.selectbox("Select chart type", ["Bar chart", "Pie chart", "Line chart"])
        if st.button("📈 Generate chart", type="primary"):
            with st.spinner("Analyzing data and creating chart..."):
                answer, _, _ = ask_llm_RAG("Find all numerical data in documents. Format: parameter name - value", st.session_state["vectorstore"], st.session_state.get("pdf_names", []))
                st.text(answer)
                data = extract_numerical_data(answer)
                if data:
                    labels, values = zip(*data)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if chart_type == "Bar chart":
                        ax.bar(labels, values, color='skyblue', edgecolor='navy', alpha=0.7)
                        ax.set_ylabel("Values")
                        plt.xticks(rotation=45, ha='right')
                    elif chart_type == "Pie chart":
                        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                    elif chart_type == "Line chart":
                        ax.plot(labels, values, marker='o', linewidth=2)
                        ax.set_ylabel("Values")
                        plt.xticks(rotation=45, ha='right')
                    ax.set_title("Data from PDF documents", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.dataframe(pd.DataFrame(data, columns=["Parameter", "Value"]), use_container_width=True)
                else:
                    st.warning("Couldn't extract numerical data. Try documents with tables or statistics.")

with tab4:
    st.header("🔍 Debug RAG (top context)")
    if "vectorstore" not in st.session_state:
        st.warning("First load and index documents in **Upload PDF** tab.")
    else:
        pdf_options = st.session_state.get("pdf_names", [])
        selected_pdfs_debug = st.multiselect("Select documents for debugging", options=pdf_options, default=pdf_options)
        debug_question = st.text_input("Question for debugging retrieval")
        debug_n = st.number_input("How many top chunks to show?", min_value=1, max_value=30, value=10, step=1)
        show_post_rerank = st.checkbox("Also show chunks after keyword-reranking", value=True)
        if st.button("🔍 Show top chunks"):
            if not debug_question.strip():
                st.warning("Enter a question to see results.")
            else:
                all_docs = st.session_state["vectorstore"].similarity_search(debug_question, k=max(debug_n, st.session_state.get("top_k_context", 3) + 3))
                filtered_docs = [doc for doc in all_docs if doc.metadata.get("source_pdf") in selected_pdfs_debug] if selected_pdfs_debug else all_docs
                hybrid_docs, hybrid_scores, bm25_scores, vector_scores = hybrid_bm25_vector_rerank(debug_question, filtered_docs, return_scores=True)
                st.subheader(f"Top {debug_n} chunks (BEFORE keyword-reranking)")
                for i, (doc, h, bm, vec) in enumerate(zip(hybrid_docs[:debug_n], hybrid_scores[:debug_n], bm25_scores[:debug_n], vector_scores[:debug_n]), start=1):
                    st.markdown(f"**{i}. {doc.metadata.get('source_pdf','?')}** | Hybrid={h:.3f} | BM25={bm:.3f} | Vector={vec:.3f}")
                    st.text_area(f"BEFORE Fragment {i}", doc.page_content[:500], height=120, key=f"db_{i}")
                    st.markdown("---")
                if show_post_rerank:
                    st.subheader("Chunks sent to LLM (AFTER keyword-reranking)")
                    final_docs = rerank_by_keyword_overlap(debug_question, hybrid_docs)
                    indices_map = {id(doc): idx for idx, doc in enumerate(hybrid_docs)}
                    for j, doc in enumerate(final_docs, start=1):
                        idx = indices_map.get(id(doc), -1)
                        h = hybrid_scores[idx] if idx >= 0 else 0.0
                        st.markdown(f"**{j}. {doc.metadata.get('source_pdf','?')}** | Hybrid={h:.3f} | Position in hybrid list: {idx+1}")
                        st.text_area(f"AFTER Fragment {j}", doc.page_content[:500], height=120, key=f"da_{j}")
                        st.markdown("---")
        st.markdown("---")
        st.subheader("📊 Session summary")
        correct = st.session_state.get("llm_correct_count", 0)
        incorrect = st.session_state.get("llm_incorrect_count", 0)
        total = correct + incorrect
        st.dataframe(pd.DataFrame([{"chunk_size": st.session_state.get("chunk_size", 1000), "overlap": st.session_state.get("chunk_overlap", 200), "top_k": st.session_state.get("top_k_context", 3), "accurate": correct, "inaccurate": incorrect, "accuracy%": round(correct/total*100, 1) if total > 0 else 0.0}]), use_container_width=True)
        if st.button("🔄 Reset accuracy counter"):
            st.session_state["llm_correct_count"] = 0
            st.session_state["llm_incorrect_count"] = 0

with tab5:
    st.header("🌐 Other knowledge sources (URL + text)")
    url_input = st.text_input("Provide URL to index", key="extra_url_input")
    raw_text = st.text_area("Or paste raw text to index", height=200, key="extra_text_input")
    if st.button("➕ Add sources to database", type="primary"):
        if "vectorstore" not in st.session_state:
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            st.session_state["vectorstore"] = Chroma(embedding_function=embeddings, persist_directory=tempfile.mkdtemp())
            st.session_state["chunks"] = []
            st.session_state["pdf_names"] = []
        new_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.get("chunk_size", 1000), chunk_overlap=st.session_state.get("chunk_overlap", 200))
        if url_input.strip():
            url_docs = WebBaseLoader([url_input.strip()]).load()
            for d in url_docs:
                d.metadata["source_pdf"] = url_input.strip()
            url_chunks = splitter.split_documents(url_docs)
            new_docs.extend(url_chunks)
            st.session_state["chunks"].extend(url_chunks)
        if raw_text.strip():
            from langchain.schema import Document
            text_chunks = splitter.split_documents([Document(page_content=raw_text.strip(), metadata={"source_pdf": "User raw text"})])
            new_docs.extend(text_chunks)
            st.session_state["chunks"].extend(text_chunks)
        if new_docs:
            st.session_state["vectorstore"].add_documents(new_docs)
            for name in set(d.metadata.get("source_pdf", "Other") for d in new_docs):
                if name not in st.session_state.get("pdf_names", []):
                    st.session_state["pdf_names"].append(name)
            st.success(f"✅ Added {len(new_docs)} new chunks to database.")
        else:
            st.info("No URL or text provided – no new sources.")

st.markdown("---")
st.markdown("""<div style="text-align:center;color:#666;padding:20px;">
    <p><strong>PDF RAG Chatbot AI</strong> — Portfolio project | Streamlit + LangChain + Hugging Face + RAG + Vector DB</p>
    <p><a href="https://github.com/Lukkaszsz/pdf-rag-chatbot" target="_blank">GitHub</a></p>
</div>""", unsafe_allow_html=True)