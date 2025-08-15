# app.py
"""
Chat with PDFs — Local-only LLaMA + TF-IDF retriever (no OpenAI / hosted APIs)

Dependencies:
pip install streamlit langchain langchain-community langchain-text-splitters pypdf python-dotenv pymupdf pillow scikit-learn joblib

Run:
streamlit run app.py
"""
import os
import io
import json
import time
import hashlib
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# PDF rendering & word bboxes
import fitz  # PyMuPDF

# Local LLM (langchain wrapper around local LlamaCpp)
from langchain_community.llms import LlamaCpp

# Document loading & splitting (local)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Simple local retrieval (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ================ Setup ================
load_dotenv()
st.set_page_config(page_title="Chat with PDFs — Visual + Highlights (Local Only)", layout="wide", page_icon="📚")

DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
DB_DIR = os.path.join(DATA_DIR, "tfidf_db")
INDEX_JSON = os.path.join(DATA_DIR, "index.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ================ Helpers ================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_index() -> Dict[str, str]:
    if os.path.exists(INDEX_JSON):
        try:
            with open(INDEX_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_index(mapping: Dict[str, str]):
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

def get_session_history(session: str):
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = []
    return st.session_state.store[session]

# Extract page text and word-level bboxes using PyMuPDF
def load_pdf_page_words(pdf_path: str, page_number: int):
    """
    Returns: (page_text: str, words: List[(x0,y0,x1,y1, word)])
    page_number is 0-indexed
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number)
    except Exception:
        return "", []
    words = page.get_text("words")  # list of tuples (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    # Normalize to (x0,y0,x1,y1, word)
    simple_words = [(w[0], w[1], w[2], w[3], w[4]) for w in words]
    text = page.get_text("text")
    doc.close()
    return text, simple_words

def render_page_with_highlights(pdf_path: str, page_no: int, highlight_bboxes: List[Tuple[float,float,float,float]], zoom: float = 2.0):
    """
    Render the given page to an image and draw semi-transparent highlights.
    highlight_bboxes: list of (x0,y0,x1,y1) in PDF coordinates (same coords as fitz page.get_text("words"))
    zoom: scale multiplier for render resolution (higher = crisper)
    Returns PIL.Image
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_no)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    draw = ImageDraw.Draw(img, "RGBA")
    # compute scale from PDF coords to pixel coords
    page_rect = page.rect  # has width,height in PDF units
    scale_x = pix.width / page_rect.width
    scale_y = pix.height / page_rect.height

    for (x0, y0, x1, y1) in highlight_bboxes:
        # convert
        rx0 = x0 * scale_x
        ry0 = y0 * scale_y
        rx1 = x1 * scale_x
        ry1 = y1 * scale_y
        # draw translucent rectangle (yellow-ish)
        draw.rectangle([rx0, ry0, rx1, ry1], fill=(255, 230, 80, 140))
    doc.close()
    return img

def find_snippet_bboxes(page_words: List[Tuple[float,float,float,float,str]], snippet: str):
    """
    Attempt to find the snippet in the page by matching sequence of words.
    page_words: list of (x0,y0,x1,y1,word)
    snippet: text snippet (string)
    Returns a list of bounding boxes (x0,y0,x1,y1) covering matched word sequences.
    Best-effort: will try to match first N words of the snippet.
    """
    if not snippet or not page_words:
        return []

    # Normalize words and snippet
    words_list = [w[4].strip() for w in page_words]
    joined = " ".join(words_list).lower()
    s = " ".join(snippet.strip().split())  # collapse whitespace
    s_lower = s.lower()

    # Simple approach: find the snippet text in joined words string
    idx = joined.find(s_lower)
    if idx != -1:
        # need to map char index back to words indices. Build char positions for each word
        positions = []
        pos = 0
        for i, w in enumerate(words_list):
            start = pos
            end = start + len(w)
            positions.append((start, end))
            pos = end + 1  # +1 for space
        # find which word contains char idx
        start_word = None
        for i, (a,b) in enumerate(positions):
            if a <= idx <= b:
                start_word = i
                break
        if start_word is None:
            return []
        # find end index (idx + len(s_lower)-1)
        end_char = idx + len(s_lower) - 1
        end_word = None
        for i, (a,b) in enumerate(positions):
            if a <= end_char <= b:
                end_word = i
                break
        if end_word is None:
            return []
        # collect bbox that spans start_word..end_word
        x0 = min(page_words[i][0] for i in range(start_word, end_word+1))
        y0 = min(page_words[i][1] for i in range(start_word, end_word+1))
        x1 = max(page_words[i][2] for i in range(start_word, end_word+1))
        y1 = max(page_words[i][3] for i in range(start_word, end_word+1))
        return [(x0,y0,x1,y1)]
    else:
        # fallback: look for first few words of snippet as sequence
        snippet_words = s_lower.split()
        for L in range(min(12, len(snippet_words)), 1, -1):
            target = " ".join(snippet_words[:L])
            for i in range(0, len(words_list)-L+1):
                window = " ".join(words_list[i:i+L]).lower()
                if window == target:
                    x0 = min(page_words[j][0] for j in range(i, i+L))
                    y0 = min(page_words[j][1] for j in range(i, i+L))
                    x1 = max(page_words[j][2] for j in range(i, i+L))
                    y1 = max(page_words[j][3] for j in range(i, i+L))
                    return [(x0,y0,x1,y1)]
        return []

# ================ Sidebar ================
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox("LLM Provider", ["Local LLaMA"])
    streaming = st.checkbox("Stream answers (local only)", value=False)
    persist = st.checkbox("Persist TF-IDF index to disk", value=True)
    top_k = st.slider("Top-K chunks", 2, 12, 5, 1)
    chunk_size = st.slider("Chunk size", 800, 6000, 3000, 200)
    chunk_overlap = st.slider("Chunk overlap", 100, 1000, 300, 50)
    role = st.selectbox("Assistant Role", ["Default", "Tutor", "Summarizer", "Exam Question Maker"])
    st.caption("Role affects the tone & structure of answers.")

# ================ Initialize LLM (local) & TF-IDF ================
# Local LLaMA (user must have a local GGML model file)
llm = LlamaCpp(model_path="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048, temperature=0.2)

# TF-IDF store helpers
TFIDF_VECT_FILE = os.path.join(DB_DIR, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_FILE = os.path.join(DB_DIR, "tfidf_matrix.joblib")
DOCS_META_FILE = os.path.join(DB_DIR, "docs_meta.json")

def build_tfidf_index(docs: List[Dict], persist_index: bool = True):
    """
    docs: list of dict {"id": str, "text": str, "source": filename, "page": int}
    """
    texts = [d["text"] for d in docs]
    if not texts:
        return None, None
    vec = TfidfVectorizer(max_features=20000)
    mat = vec.fit_transform(texts)
    if persist_index:
        joblib.dump(vec, TFIDF_VECT_FILE)
        joblib.dump(mat, TFIDF_MATRIX_FILE)
        with open(DOCS_META_FILE, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
    return vec, mat

def load_tfidf_index():
    if os.path.exists(TFIDF_VECT_FILE) and os.path.exists(TFIDF_MATRIX_FILE) and os.path.exists(DOCS_META_FILE):
        vec = joblib.load(TFIDF_VECT_FILE)
        mat = joblib.load(TFIDF_MATRIX_FILE)
        with open(DOCS_META_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return vec, mat, docs
    return None, None, []

def tfidf_similarity_search(vec: TfidfVectorizer, mat, docs_meta: List[Dict], query: str, k: int = 5):
    if vec is None or mat is None or not docs_meta:
        return [], 0.0
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat)[0]  # shape (n_docs,)
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for idx in top_idx:
        d = docs_meta[idx]
        # Attach a confidence score between 0 and 1
        conf = float(sims[idx])
        results.append((d, conf))
    avg_conf = float(sims[top_idx].mean()) if len(top_idx) else 0.0
    return results, avg_conf

# ================ Session state ================
if "file_index" not in st.session_state:
    st.session_state.file_index = {}
if "documents" not in st.session_state:
    st.session_state.documents = []  # list of dicts {"id","text","source","page"}
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "docs_meta" not in st.session_state:
    st.session_state.docs_meta = []
if "chunk_hashes" not in st.session_state:
    st.session_state.chunk_hashes = set()
if "active_view" not in st.session_state:
    st.session_state.active_view = ("", 0)  # (filename, page)
if "last_query_results" not in st.session_state:
    st.session_state.last_query_results = []

# Try load persisted TF-IDF index
if st.session_state.vectorizer is None:
    vec, mat, docs = load_tfidf_index()
    if vec is not None:
        st.session_state.vectorizer = vec
        st.session_state.tfidf_matrix = mat
        st.session_state.docs_meta = docs

# ================ Layout ================
left, right = st.columns([1.25, 1])

with left:
    st.title("📚 Chat with PDFs — Visual + Highlights (Local Only)")
    session_id = st.text_input("Session ID", value="default_session")

    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Drag & drop or browse", type="pdf", accept_multiple_files=True)

    existing_index = load_index()
    new_docs = []
    file_index = {}

    if uploaded_files:
        with st.spinner("Reading PDFs…"):
            for up in uploaded_files:
                content = up.getvalue()
                file_hash = sha256_bytes(content)
                pdf_path = os.path.join(PDF_DIR, up.name)
                with open(pdf_path, "wb") as f:
                    f.write(content)

                changed = (existing_index.get(up.name) != file_hash)

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for d in docs:
                    d.metadata = d.metadata or {}
                    src_name = up.name
                    page_no = int(d.metadata.get("page", 0))
                    d.metadata["source"] = src_name
                    d.metadata["page"] = page_no
                file_index[up.name] = docs

                if changed:
                    new_docs.extend(docs)
                    existing_index[up.name] = file_hash

        st.session_state.file_index = file_index

        # Convert langchain Documents to our simple dict docs and add to documents list
        added_count = 0
        for fname, docs in file_index.items():
            for d in docs:
                text = d.page_content.strip()
                meta = {"id": sha256_bytes((fname + str(d.metadata.get("page",0)) + text[:128]).encode("utf-8")),
                        "text": text,
                        "source": fname,
                        "page": int(d.metadata.get("page", 0))}
                # dedupe by id
                if meta["id"] not in {dd["id"] for dd in st.session_state.documents}:
                    st.session_state.documents.append(meta)
                    added_count += 1

        if new_docs:
            with st.spinner("Chunking & building TF-IDF index (only new/changed PDFs)…"):
                # We will chunk using RecursiveCharacterTextSplitter to maintain compatibility
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", " "],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                # prepare langchain Documents for splitting
                all_new_docs = new_docs
                splits = splitter.split_documents(all_new_docs)
                # convert splits to simple dicts
                splits_simple = []
                for s in splits:
                    text = s.page_content.strip()
                    if not text:
                        continue
                    mid = sha256_bytes(text.encode("utf-8"))
                    if mid in st.session_state.chunk_hashes:
                        continue
                    st.session_state.chunk_hashes.add(mid)
                    meta = {"id": mid, "text": text, "source": s.metadata.get("source", "Unknown.pdf"), "page": int(s.metadata.get("page", 0))}
                    splits_simple.append(meta)
                # append to documents
                if splits_simple:
                    st.session_state.docs_meta.extend(splits_simple)
                    # rebuild TF-IDF index from docs_meta
                    vec, mat = build_tfidf_index(st.session_state.docs_meta, persist_index=persist)
                    st.session_state.vectorizer = vec
                    st.session_state.tfidf_matrix = mat
                    if persist:
                        save_index(existing_index)

    if not st.session_state.docs_meta:
        st.info("Upload one or more PDFs to start (or run Summarize/Study modes after uploading).")
        st.stop()

    st.subheader("Select PDFs to query")
    all_names = sorted(list(st.session_state.file_index.keys()))
    selected_files = st.multiselect("Your question/search will use only these PDFs", options=all_names, default=all_names)
    if not selected_files:
        st.warning("Select at least one PDF.")
        st.stop()

    # Simple hybrid search that filters by source file
    def hybrid_search(query: str, top_k_local: int = 5):
        # perform TF-IDF search over the stored docs (docs_meta)
        results, avg_conf = tfidf_similarity_search(st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.docs_meta, query, k=top_k_local)
        # filter results to only selected_files
        filtered = []
        for item, conf in results:
            if item.get("source") in selected_files:
                filtered.append((item, conf))
        # If no hits in selected files, fall back to original results then filter by source
        if not filtered:
            filtered = results
        # return list of item dicts (no langchain Document objects) and avg_conf
        docs = []
        for d, conf in filtered:
            docs.append({"text": d["text"], "source": d["source"], "page": d["page"], "vector_confidence": conf})
        return docs, avg_conf

    # Prompts
    role_prompts = {
        "Default": "Answer concisely in up to 4 sentences.",
        "Tutor": "Explain like a patient tutor: step-by-step, clear and structured.",
        "Summarizer": "Summarize in tidy bullet points with short phrases.",
        "Exam Question Maker": "Think like an examiner; focus on key facts and common pitfalls.",
    }

    qa_system_prompt = (
        "You answer questions strictly based on the provided context. "
        "If the answer is not in the documents, say you don't know.\n\n"
        "Cite sources inline like: (Source: <PDF>, p.<page>).\n\n"
    )

    st.subheader("💬 Ask a question")
    q = st.text_input("Your question")

    def call_llm(prompt: str) -> str:
        """
        Call the local LLaMA model via LlamaCpp wrapper.
        The LlamaCpp instance supports being called like a function or via .generate depending on version.
        We'll try callable first, else fallback to .generate prints.
        """
        try:
            out = llm(prompt)
            # many wrappers return a dict-like or object; try to get string
            if isinstance(out, str):
                return out
            # try attributes
            text = getattr(out, "content", None) or str(out)
            return text
        except Exception:
            try:
                res = llm.generate([prompt])
                # res may be a LangChain GenerationResult
                # try to extract text:
                generations = getattr(res, "generations", None)
                if generations:
                    return " ".join([g[0].text for g in generations])
                return str(res)
            except Exception as e:
                return f"[LLM error] {e}"

    if st.button("Ask") and q:
        session_history = get_session_history(session_id)
        with st.spinner("Searching & answering…"):
            docs, avg_conf = hybrid_search(q, top_k_local=top_k)

            # build context string from retrieved docs
            context_parts = []
            highlights_info = []
            for i, d in enumerate(docs, 1):
                snippet = d["text"][:800]
                context_parts.append(f"---\nSource: {d['source']} | Page: {d['page']}\nText:\n{d['text']}\n")
                highlights_info.append((d["source"], d["page"], d["text"]))
            context = "\n\n".join(context_parts) if context_parts else ""

            # Build prompt
            role_instr = role_prompts.get(role, role_prompts["Default"])
            full_prompt = (
                f"SYSTEM NOTE: {qa_system_prompt}\nRole instructions: {role_instr}\n\n"
                "CONTEXT START\n"
                f"{context}\n"
                "CONTEXT END\n\n"
                "QUESTION:\n"
                f"{q}\n\n"
                "INSTRUCTIONS: Answer only using the provided CONTEXT. If the information is not contained in the context, reply with 'I don't know based on the provided documents.' Cite your sources inline like (Source: filename.pdf, p.5) when you reference facts."
            )

            # Call local LLM
            answer_text = call_llm(full_prompt)

            # store to session history
            session_history.append({"role": "user", "content": q})
            session_history.append({"role": "assistant", "content": answer_text})

            st.markdown("### 🧠 Answer")
            st.write(answer_text)

            # Show citations and prepare highlights
            if docs:
                st.markdown("#### 📎 Sources used")
                for i, d in enumerate(docs, 1):
                    src = d.get("source", "Unknown.pdf")
                    page = int(d.get("page", 0))
                    conf = d.get("vector_confidence", None)
                    c1, c2 = st.columns([0.8, 0.2])
                    with c1:
                        if conf is not None:
                            st.markdown(f"**{i}. {src} — Page {page}** · confidence ~ {conf:.2f}")
                        else:
                            st.markdown(f"**{i}. {src} — Page {page}**")
                    with c2:
                        if st.button("Open", key=f"open_cite_{src}_{page}_{i}"):
                            st.session_state.active_view = (src, page)
                st.caption(f"Overall retrieval confidence: {avg_conf:.2f}")

                # Store highlights info in session for the right panel to render
                st.session_state._last_highlights = highlights_info

    st.divider()

    # Summaries + Study Modes (basic implementations using local LLM)
    st.subheader("📝 Summaries")
    colA, colB = st.columns(2)

    def local_summarize(docs_texts: List[str], prompt_intro: str):
        context = "\n\n".join(docs_texts[:40])
        prompt = f"{prompt_intro}\n\nCONTEXT:\n{context}\n\nProvide a concise study summary."
        return call_llm(prompt)

    if colA.button("Summarize SELECTED PDFs"):
        with st.spinner("Summarizing selected PDFs…"):
            sel_docs = []
            for fname in selected_files:
                # collect pages as text
                for d in st.session_state.docs_meta:
                    if d["source"] == fname:
                        sel_docs.append(d["text"])
            out = local_summarize(sel_docs, "Summarize the following PDF content into clear bullet points with headings and sub-bullets. Focus on key results, definitions, equations, and takeaways.")
            st.markdown("### 🔍 Summary (Selected)")
            st.write(out)

    if colB.button("Summarize ALL PDFs"):
        with st.spinner("Summarizing all PDFs…"):
            all_texts = [d["text"] for d in st.session_state.docs_meta]
            out = local_summarize(all_texts, "Summarize the following PDF content into clear bullet points with headings and sub-bullets. Focus on key results, definitions, equations, and takeaways.")
            st.markdown("### 📘 Summary (All)")
            st.write(out)

    st.divider()

    st.subheader("🎯 Study / Research Modes")
    study_mode = st.selectbox("Choose a study mode",
                              ["None", "Exam Prep (MCQs + Short Answers)", "Flashcard Generator", "Concept Map Creator"])
    if study_mode != "None" and st.button("Run Study Mode"):
        with st.spinner(f"Running {study_mode}…"):
            mode_texts = []
            for fname in selected_files:
                for d in st.session_state.docs_meta:
                    if d["source"] == fname:
                        mode_texts.append(d["text"])
            context = "\n\n".join(mode_texts[:50])
            if study_mode == "Exam Prep (MCQs + Short Answers)":
                prompt = (
                    "You are an educational content creator. From the provided context, create:\n"
                    "1) 5 Multiple Choice Questions (with 4 options A–D, mark the correct one)\n"
                    "2) 5 Short Answer Questions (with concise model answers)\n\n"
                    f"CONTEXT:\n{context}"
                )
            elif study_mode == "Flashcard Generator":
                prompt = (
                    "You are a flashcard creator. From the provided context, extract important concepts and their explanations as Q&A pairs. Format:\nQ: ...\nA: ...\n\n"
                    f"CONTEXT:\n{context}"
                )
            else:  # Concept Map Creator
                prompt = (
                    "You are a research assistant. From the provided context, extract key topics, subtopics, and relationships. Output as a clean bullet list.\n\n"
                    f"CONTEXT:\n{context}"
                )
            out = call_llm(prompt)
            st.markdown(f"### {study_mode} Output")
            st.write(out)

    st.divider()

    st.subheader("⬇️ Export Chat")
    if st.button("Prepare Chat History"):
        hist = get_session_history(session_id)
        lines = []
        for m in hist:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role in ("assistant", "ai"):
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"{role.capitalize()}: {content}")
        export_text = "\n\n".join(lines) if lines else "No chat yet."
        st.download_button(label="Download Chat (.txt)", data=export_text,
                           file_name=f"chat_history_{session_id}.txt", mime="text/plain")

# ================ Right Panel: PDF Viewer with Highlights ================
with right:
    st.header("📄 Document Viewer (visual)")

    if st.session_state.file_index:
        names = sorted(list(st.session_state.file_index.keys()))
        default_name = st.session_state.active_view[0] if st.session_state.active_view[0] in names else names[0]
        view_name = st.selectbox("Choose PDF", names, index=names.index(default_name))

        # pages available
        pages = sorted({int(d.metadata.get("page", 0)) for d in st.session_state.file_index[view_name]})
        default_page = st.session_state.active_view[1] if st.session_state.active_view[0] == view_name else pages[0]
        if default_page not in pages:
            default_page = pages[0]
        view_page = st.selectbox("Page", pages, index=pages.index(default_page))

        st.session_state.active_view = (view_name, int(view_page))
        pdf_path = os.path.join(PDF_DIR, view_name)

        # collect highlight bboxes for this page from last answer (if any)
        highlight_bboxes = []
        last_highlights = st.session_state.get("_last_highlights", [])
        for (src, pno, snippet) in last_highlights:
            if src == view_name and int(pno) == int(view_page):
                # load page words
                page_text, page_words = load_pdf_page_words(pdf_path, int(view_page))
                bboxes = find_snippet_bboxes(page_words, snippet)
                for bb in bboxes:
                    highlight_bboxes.append(bb)

        # Render the page with highlights (best-effort)
        try:
            img = render_page_with_highlights(pdf_path, int(view_page), highlight_bboxes, zoom=2.0)
            st.image(img, use_column_width=True)
        except Exception as e:
            st.info("Could not render PDF page visually (falling back to text).")
            # fallback: show text
            docs_on_page = [d for d in st.session_state.file_index[view_name] if int(d.metadata.get("page", 0)) == view_page]
            if docs_on_page:
                st.text_area("Page text preview", docs_on_page[0].page_content.strip(), height=520)
            else:
                st.info("No text extracted for this page.")

        # Also show the raw snippets that were highlighted (for clarity)
        if last_highlights:
            st.markdown("**Highlighted snippets from last answer (best-effort):**")
            for src, pno, snippet in last_highlights:
                if src == view_name and int(pno) == int(view_page):
                    st.write("- " + snippet[:800] + ("..." if len(snippet) > 800 else ""))
    else:
        st.info("Upload PDFs on the left to preview pages here.")
