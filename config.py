import os
from pathlib import Path

class Config:
    # Model settings
    DEFAULT_MODEL_PATH = "models/"
    MAX_CONTEXT_LENGTH = 4096
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 40
    REPEAT_PENALTY = 1.1
    
    # Embeddings
    EMBEDDING_MODEL = "google/embeddinggemma-300m"
    EMBEDDING_DIM = 768
    
    # Memory
    MEMORY_FILE = "long_term_memory.json"
    MEMORY_SYNTHESIS_INTERVAL = 5  # conversations
    
    # Web search
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")
    
    # Canvas
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 600
    
    # MCP
    MCP_PORT = 8000
    MCP_HOST = "0.0.0.0"
    
    # CPU settings
    N_THREADS = 4
    N_BATCH = 512
    
    @staticmethod
    def ensure_directories():
        Path("models").mkdir(exist_ok=True)
        Path("memory").mkdir(exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)
