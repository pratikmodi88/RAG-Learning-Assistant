import os
from pathlib import Path
import streamlit as st
from google import genai
import time
import math
import pdfplumber
import re

# ------------------ CLEAN OUTPUT ------------------

def clean_llm_output(text):
    text = re.sub(r"</?div[^>]*>", "", text)
    text = re.sub(r"<(?!table|tr|td|th|thead|tbody|br|b|i)[^>]+>", "", text)
    return text.strip()

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Nova AI", layout="wide")
st.title("🤖 Nova AI Assistant")

# ------------------ ENV ------------------

def load_env():
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

load_env()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------ SESSION ------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ------------------ FILE UPLOAD ------------------

uploaded_files = st.file_uploader(
    "📂 Upload .txt or .pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# ------------------ HELPERS ------------------

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def create_chunks(text, filename):
    return [{"text": text[i:i+400], "source": filename}
            for i in range(0, len(text), 400)]

def embed_texts(texts):
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in res.embeddings]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) *
                  math.sqrt(sum(x*x for x in b)) + 1e-8)

# ------------------ FILE PROCESSING (FINAL FIX) ------------------

current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
previous_file_names = st.session_state.uploaded_file_names

if current_file_names != previous_file_names:

    progress = st.progress(0)
    status = st.empty()

    # ✅ STEP 1: remove deleted files FIRST
    removed_files = set(previous_file_names) - set(current_file_names)

    if removed_files:
        st.session_state.chunks = [
            c for c in st.session_state.chunks
            if c["source"] not in removed_files
        ]

    # ✅ STEP 2: copy updated state
    all_chunks = st.session_state.chunks.copy()

    total_files = len(uploaded_files)

    for i, f in enumerate(uploaded_files):

        percent = int(((i+1)/total_files)*100)
        progress.progress(percent)
        status.text(f"Processing {f.name} ({percent}%)")
        time.sleep(0.2)

        # remove old version of same file
        all_chunks = [c for c in all_chunks if c["source"] != f.name]

        if f.name.endswith(".txt"):
            text = f.read().decode("utf-8")
        else:
            text = read_pdf(f)

        if not text.strip():
            continue

        all_chunks.extend(create_chunks(text, f.name))

    # -------- EMBEDDINGS --------

    valid = [c for c in all_chunks if c["text"].strip()]
    texts = [c["text"] for c in valid]

    if not texts:
        status.error("❌ No readable text")
        st.stop()

    status.text("Generating embeddings...")
    progress.progress(90)

    embeddings = embed_texts(texts)

    for i, c in enumerate(valid):
        c["embedding"] = embeddings[i]

    st.session_state.chunks = valid
    st.session_state.uploaded_file_names = current_file_names

    progress.progress(100)
    status.text("✅ Ready")
    time.sleep(1)

    progress.empty()
    status.empty()

# ------------------ FILE FILTER ------------------

if st.session_state.chunks:
    files = list(set([c["source"] for c in st.session_state.chunks]))
    selected_file = st.selectbox("📄 Select document", ["All Documents"] + files)
else:
    selected_file = "All Documents"

# ------------------ RETRIEVE ------------------

def retrieve(q, chunks):
    if not chunks:
        return []

    q_emb = embed_texts([q])[0]

    scored = [(cosine(q_emb, c["embedding"]), c) for c in chunks]
    scored.sort(reverse=True, key=lambda x: x[0])

    if scored[0][0] < 0.25:
        return []

    return [c for _, c in scored[:3]]

# ------------------ PROMPT ------------------

def build_prompt(q, chunks):
    ctx = "\n\n".join([c["text"] for c in chunks])
    return f"""
You are Nova.

- No HTML tags
- Use markdown tables if needed

Context:
{ctx}

Question:
{q}

Answer:
"""

# ------------------ STREAM ------------------

def stream(prompt):
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = clean_llm_output(res.text)

    box = st.empty()
    out = ""

    for w in text.split():
        out += w + " "
        box.markdown(f"""
        <div style='background:#1e1e1e;padding:14px;border-radius:12px'>
        🤖 {out}
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.02)

    return text

# ------------------ DISPLAY ------------------

def is_table(text):
    return "|" in text and "---" in text

for m in st.session_state.messages:

    if m["role"] == "user":
        st.markdown(f"""
        <div style='text-align:right;background:#2f855a;color:white;padding:10px;border-radius:10px'>
        {m["content"]}
        </div>
        """, unsafe_allow_html=True)

    else:
        ans = m["content"]["answer"]
        src = m["content"]["sources"]

        cleaned = clean_llm_output(ans)

        if is_table(cleaned):
            st.markdown(cleaned)
        else:
            st.markdown(f"""
            <div style='background:#1e1e1e;padding:14px;border-radius:12px'>
            🤖 {cleaned}
            </div>
            """, unsafe_allow_html=True)

        if src:
            st.markdown("### 📚 Sources")

            seen = set()
            for s in src:
                if s["source"] in seen:
                    continue
                seen.add(s["source"])

                st.markdown(f"""
                <div style='background:#111;padding:8px;border-radius:8px'>
                <b>{s["source"]}</b><br>
                {s["text"][:200]}
                </div>
                """, unsafe_allow_html=True)

# ------------------ INPUT ------------------

q = st.chat_input("Ask your question...")

if q:
    st.session_state.last_query = q

    if selected_file != "All Documents":
        chunks = [c for c in st.session_state.chunks if c["source"] == selected_file]
    else:
        chunks = st.session_state.chunks

    st.session_state.messages.append({"role":"user","content":q})

    with st.spinner("Thinking..."):
        rel = retrieve(q, chunks)

        if not rel:
            ans = "No information available."
        else:
            ans = stream(build_prompt(q, rel))

    st.session_state.messages.append({
        "role":"assistant",
        "content":{"answer":ans,"sources":rel}
    })

    st.rerun()