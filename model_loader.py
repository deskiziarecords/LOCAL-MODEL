from llama_cpp import Llama
from typing import List, Dict, Optional, Iterator
import json
import time
from config import Config

class GGUFModelLoader:
    def __init__(self, model_path: str, n_ctx: int = Config.MAX_CONTEXT_LENGTH):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None
        self.conversation_history = []
        
    def load_model(self):
        """Load the GGUF model with CPU-only settings"""
        try:
            self.model = Llama(
                model
