# LOCAL-MODEL
🚀 Usage

    Run the GUI Application:

bash
Copy

python gguf_loader.py

    API Server Usage (after starting from GUI):

Python
Copy

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

# Chat completion
response = client.chat.completions.create(
    model="gguf-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Embeddings
embeddings = client.embeddings.create(
    model="google/embeddinggemma-300m",
    input=["Hello world", "Goodbye world"]
)

    Web Search API:

bash
Copy

curl -X POST http://localhost:8000/v1/web/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest AI news", "max_results": 5}'

    Memory Search API:

bash
Copy

curl -X POST http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 3}'
