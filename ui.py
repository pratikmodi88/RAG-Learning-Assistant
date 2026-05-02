import os
from pathlib import Path
import streamlit as st
from google import genai
import time
import math
import pdfplumber

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Nova AI", layout="wide")
st.title("🤖 Nova AI Assistant")

# ------------------ STYLE ------------------

st.markdown("""
<style>
.main > div {
    max-width: 1200px;
    margin: auto;
}
div[data-testid="stChatInput"] {
    border: 1px solid #ff4b4b;
    border-radius: 12px;
    padding: 6px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD ENV ------------------

def load_env_file(file_path=".env"):
    env_path = Path(file_path)
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

load_env_file()

# ------------------ GEMINI ------------------

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("API key not found")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# ------------------ SESSION ------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ------------------ FILE UPLOAD ------------------

uploaded_files = st.file_uploader(
    "📂 Upload .txt or .pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def create_chunks(text, filename, chunk_size=400):
    return [
        {"text": text[i:i+chunk_size], "source": filename}
        for i in range(0, len(text), chunk_size)
    ]

# ------------------ EMBEDDINGS ------------------

def embed_texts(texts):
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in res.embeddings]

# ------------------ LOAD FILES WITH PROGRESS ------------------

progress_bar = st.progress(0)
status_text = st.empty()

if uploaded_files and not st.session_state.files_processed:
    all_chunks = []
    total_files = len(uploaded_files)

    for idx, file in enumerate(uploaded_files):

        percent = int(((idx + 1) / total_files) * 100)
        progress_bar.progress(percent)
        status_text.text(f"Processing {file.name} ({percent}%)")
        time.sleep(0.2)

        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")

        elif file.name.endswith(".pdf"):
            text = read_pdf(file)

        else:
            continue

        if not text.strip():
            continue

        chunks = create_chunks(text, file.name)
        all_chunks.extend(chunks)

    # -------- EMBEDDINGS --------

    status_text.text("Generating embeddings...")
    progress_bar.progress(90)

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    for i, c in enumerate(all_chunks):
        c["embedding"] = embeddings[i]

    st.session_state.chunks = all_chunks
    st.session_state.files_processed = True

    progress_bar.progress(100)
    status_text.text("✅ Files ready!")

    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

# ------------------ FILE FILTER DROPDOWN ------------------

if st.session_state.chunks:
    file_names = list(set([c["source"] for c in st.session_state.chunks]))

    selected_file = st.selectbox(
        "📄 Select document (optional)",
        ["All Documents"] + file_names
    )
else:
    selected_file = "All Documents"

# ------------------ RETRIEVAL ------------------

def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b + 1e-8)

def retrieve(query, chunks):
    if not chunks:
        return []

    query_emb = embed_texts([query])[0]

    scored = []
    for c in chunks:
        sim = cosine_sim(query_emb, c["embedding"])
        scored.append((sim, c))

    if not scored:
        return []

    scored.sort(reverse=True, key=lambda x: x[0])

    if scored[0][0] < 0.25:
        return []

    return [c for _, c in scored[:3]]

# ------------------ PROMPT ------------------

def build_prompt(question, chunks):
    context = "\n\n".join([c["text"] for c in chunks])

    return f"""
You are Nova, an AI assistant.

Use the context below to answer the question.
Be flexible in understanding meaning.

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ------------------ GEMINI ------------------

def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# ------------------ HIGHLIGHT ------------------

def highlight_text(text, query):
    words = query.lower().split()
    highlighted = text

    for w in words:
        if len(w) > 2:
            highlighted = highlighted.replace(
                w,
                f"<span style='background:#444;padding:2px 4px;border-radius:4px'>{w}</span>"
            )
            highlighted = highlighted.replace(
                w.capitalize(),
                f"<span style='background:#444;padding:2px 4px;border-radius:4px'>{w.capitalize()}</span>"
            )
    return highlighted

# ------------------ CHAT DISPLAY ------------------

for msg in st.session_state.messages:

    if msg["role"] == "user":
        col1, col2 = st.columns([1, 2])
        with col2:
            st.markdown(f"""
            <div style='background:#2f855a;color:white;padding:12px;border-radius:12px;margin:8px 0;margin-left:auto;max-width:60%;'>
            {msg['content']} 🧑
            </div>
            """, unsafe_allow_html=True)

    else:
        answer = msg["content"]["answer"]
        sources = msg["content"]["sources"]

        st.markdown(f"""
        <div style='background:#1e1e1e;color:white;padding:16px;border-radius:14px;margin:10px 0;'>
        🤖 {answer}
        </div>
        """, unsafe_allow_html=True)

        if sources:
            st.markdown("### 📚 Sources")

            for s in sources:
                highlighted = highlight_text(
                    s["text"][:200],
                    st.session_state.last_query
                )

                st.markdown(f"""
                <div style='background:#111;padding:10px;border-radius:10px;margin-bottom:8px;font-size:13px;color:#ccc;'>
                📄 <b>{s['source']}</b><br>
                {highlighted}...
                </div>
                """, unsafe_allow_html=True)

# ------------------ INPUT BAR ------------------

col1, col2, col3 = st.columns([8, 1, 1])

with col1:
    user_input = st.chat_input("Ask your question...")

with col2:
    if st.button("🧹"):
        st.session_state.messages = []
        st.rerun()

with col3:
    if st.button("🗑️"):
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.files_processed = False
        st.rerun()

# ------------------ PROCESS ------------------

if user_input:
    st.session_state.last_query = user_input

    if selected_file != "All Documents":
        filtered_chunks = [c for c in st.session_state.chunks if c["source"] == selected_file]
    else:
        filtered_chunks = st.session_state.chunks

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Nova is thinking..."):
        relevant = retrieve(user_input, filtered_chunks)

        if not relevant:
            answer = "No information available."
        else:
            prompt = build_prompt(user_input, relevant)
            answer = ask_gemini(prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": {
            "answer": answer,
            "sources": relevant
        }
    })

    st.rerun()