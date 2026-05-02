import os
from pathlib import Path
from google import genai

# ---------- LOAD .env ----------
def load_env():
    env_path = Path(".env")
    if not env_path.exists():
        print("No .env file found")
        return

    for line in env_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

load_env()

# ---------- GET API KEY ----------
api_key = os.environ.get("GEMINI_API_KEY")

print("API KEY USED:", api_key)

if not api_key:
    print("❌ API key not found")
    exit()

# ---------- INIT CLIENT ----------
client = genai.Client(api_key=api_key)

# ---------- TEST CALL ----------
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",   # IMPORTANT: use this model
        contents="Say hello in one short sentence"
    )

    print("\n✅ SUCCESS:\n")
    print(response.text)

except Exception as e:
    print("\n❌ ERROR:\n")
    print(str(e))
