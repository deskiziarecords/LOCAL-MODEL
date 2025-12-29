# LOCAL-MODEL Setup Guide

## Quick Start (5 minutes)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd LOCAL-MODEL
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env with your preferences
```

### 4. Get a Model
Download a GGUF model and place it in `./models/`:
- [Mistral 7B](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- [Llama 2](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [OpenHermes](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF)

### 5. Run
```bash
python main.py
# Or: make run
```

---

## Commands

### Using Makefile (Recommended)
```bash
make run            # Start with GUI
make api            # Start API server only
make list-models    # List available models
make debug          # Run in debug mode
make logs           # Watch log files
make clean          # Clean cache
```

### Using Python Directly
```bash
python main.py                     # Start GUI
python main.py --no-gui           # API only
python main.py --model llama      # Load specific model
python main.py --list-models      # Show available models
python main.py --debug --port 9000 # Custom config
```

---

## Configuration

### Edit .env File
```bash
# Change these commonly:
MODEL_PATH=./models
DEFAULT_MODEL=mistral-7b
GPU_LAYERS=-1      # -1 = all, 0 = CPU only
API_PORT=8000

ENABLE_WEB_SEARCH=true
ENABLE_MEMORY=true
```

### Or Command Line
```bash
python main.py --model llama --port 9000 --gpu-layers 20
```

---

## Troubleshooting

### "No models found"
```bash
make list-models
# Check ./models/ directory has .gguf files
```

### GPU not working
```bash
# Check GPU layers setting
python main.py --gpu-layers 0  # Force CPU
# Or set GPU_LAYERS=0 in .env
```

### Port already in use
```bash
python main.py --port 9000  # Use different port
```

### API won't start
```bash
make clean
pip install -r requirements.txt
python main.py --no-gui --debug  # See error details
```

### Out of memory
```bash
# Reduce GPU layers or use smaller model
python main.py --gpu-layers 10
```

---

## API Usage Examples

### Python
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gguf-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### cURL
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gguf-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Using Chat History

```python
from chat_history import ChatHistory

# Load or create history
chat = ChatHistory()

# Add messages
chat.add('user', 'What is Python?')
chat.add('assistant', 'Python is a programming language.')

# Get context for LLM
context = chat.get_conversation_context(include_last_n=10)

# Print summary
print(chat.get_summary())

# Export to text
chat.export('my_conversation.txt')
```

---

## Project Structure
```
LOCAL-MODEL/
├── main.py              # Entry point
├── config.py            # Configuration
├── model_loader.py      # Model loading
├── chat_history.py      # Chat persistence
├── utils.py             # Helper functions
├── .env                 # Your configuration (create from .env.example)
├── .env.example         # Configuration template
├── .gitignore          # Git ignore rules
├── requirements.txt     # Dependencies
├── Makefile            # Quick commands
├── SETUP.md            # This file
├── logs/               # Log files (auto-created)
├── models/             # GGUF model files
└── chat_history.json   # Chat history (auto-created)
```

---

## Tips

### Speed Up First Load
- Use smaller models (7B better than 13B)
- Set `GPU_LAYERS=-1` if you have VRAM
- Pre-download models

### Better Performance
- Increase GPU_LAYERS for faster inference
- Use models quantized as Q4 or Q5
- Check logs for bottlenecks: `make logs`

### Development
- Run with `--debug` flag to see detailed logs
- Use `make clean` before troubleshooting
- Check `chat_history.json` is being created

### Save Space
- Clean up old log files: `rm logs/*.log`
- Remove chat history: `rm chat_history.json`
- Use quantized models (smaller file size)

---

## FAQ

**Q: Which model should I use?**
A: Start with Mistral 7B - good balance of speed and quality

**Q: Can I use multiple models?**
A: Yes, switch with `--model` flag or dropdown in GUI

**Q: How do I make it faster?**
A: Increase GPU_LAYERS, use smaller models, disable features

**Q: Can I use this over network?**
A: Yes, change API_HOST from 127.0.0.1 to 0.0.0.0 (security risk!)

**Q: How much VRAM needed?**
A: 7B models need 6-8GB, 13B need 10-16GB, depends on quantization

---

## Next Steps

1. ✅ Run `make run` to start
2. ✅ Chat with the model in GUI
3. ✅ Check logs with `make logs`
4. ✅ Export chat with `chat.export()`
5. ✅ Customize .env for your hardware


