import os
from pathlib import Path
from google import genai

DOCS_FOLDER = Path("docs")

# ------------------ LOAD ENV ------------------

def load_env_file(file_path=".env"):
    env_path = Path(file_path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()

# ------------------ GEMINI CLIENT ------------------

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found")
    return genai.Client(api_key=api_key)

# ------------------ LOAD DOCS ------------------

def load_documents():
    docs = []
    for file in DOCS_FOLDER.glob("*.txt"):
        text = file.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs

def create_chunks(documents, chunk_size=400):
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i:i+chunk_size])
    return chunks

# ------------------ SIMPLE RETRIEVAL ------------------

def retrieve(query, chunks):
    scored = []

    for chunk in chunks:
        score = sum(1 for word in query.lower().split() if word in chunk.lower())
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [c for _, c in scored[:3]]

    return top_chunks if any(score > 0 for score, _ in scored) else chunks[:3]

# ------------------ PROMPT ------------------

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)

    return f"""
You are a helpful assistant.

Answer ONLY from the context below.
If not found, say: "No information available."

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ------------------ GEMINI RESPONSE ------------------

def ask_gemini(client, prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------ MAIN ------------------

def main():
    load_env_file()
    client = get_client()

    docs = load_documents()
    chunks = create_chunks(docs)

    print("Nova CLI Bot Ready (type 'exit' to quit)\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break

        relevant = retrieve(q, chunks)
        prompt = build_prompt(q, relevant)
        answer = ask_gemini(client, prompt)

        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    main()