import os
from pathlib import Path
import streamlit as st
from google import genai
import time

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

# ------------------ FILE UPLOAD ------------------

uploaded_files = st.file_uploader(
    "📂 Upload .txt files",
    type=["txt"],
    accept_multiple_files=True
)

def create_chunks(text, filename, chunk_size=400):
    return [
        {"text": text[i:i+chunk_size], "source": filename}
        for i in range(0, len(text), chunk_size)
    ]

banner = st.empty()

if uploaded_files and not st.session_state.files_processed:
    all_chunks = []

    for file in uploaded_files:
        text = file.read().decode("utf-8")
        all_chunks.extend(create_chunks(text, file.name))

    st.session_state.chunks = all_chunks
    st.session_state.files_processed = True

    banner.success("✅ Files loaded into memory")
    time.sleep(2)
    banner.empty()

# ------------------ RETRIEVAL ------------------

def retrieve(query, chunks):
    scored = []

    for chunk in chunks:
        score = sum(1 for w in query.lower().split() if w in chunk["text"].lower())
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:3]]

# ------------------ PROMPT ------------------

def build_prompt(question, chunks):
    context = "\n\n".join([c["text"] for c in chunks])

    return f"""
You are Nova, an AI assistant.

Answer ONLY from the context below.
If not found, say: "No information available."

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

# ------------------ CHAT DISPLAY ------------------

for msg in st.session_state.messages:

    if msg["role"] == "user":
        col1, col2 = st.columns([1, 2])

        with col2:
            st.markdown(f"""
            <div style='
                background:#2f855a;
                color:white;
                padding:12px;
                border-radius:12px;
                margin:8px 0;
                width:fit-content;
                margin-left:auto;
                max-width:60%;
                text-align:right;
            '>
            {msg['content']} 🧑
            </div>
            """, unsafe_allow_html=True)

    else:
        answer = msg["content"]["answer"]
        sources = msg["content"]["sources"]

        st.markdown(f"""
        <div style='
            background:#1e1e1e;
            color:white;
            padding:16px;
            border-radius:14px;
            margin:10px 0;
            width:100%;
        '>
        🤖 {answer}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📚 Sources")

        for s in sources:
            st.markdown(f"""
            <div style='
                background:#111;
                padding:10px;
                border-radius:10px;
                margin-bottom:8px;
                font-size:13px;
                color:#ccc;
            '>
            📄 <b>{s['source']}</b><br>
            {s['text'][:200]}...
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
    if not st.session_state.chunks:
        st.warning("⚠️ Upload document first")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Nova is thinking..."):
            relevant = retrieve(user_input, st.session_state.chunks)
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