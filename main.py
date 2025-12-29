# gguf_loader.py
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import requests
from dataclasses import dataclass

# LLM and Embeddings
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Web Framework
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Canvas and Visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Web Search
from duckduckgo_search import DDGS

# MCP Integration
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class GGUFModelLoader:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.embedding_model = None
        self.memory_file = "long_term_memory.json"
        self.memories = []
        self.app = FastAPI(title="GGUF Model Loader API", version="1.0.0")
        self.setup_api_routes()
        
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        default_config = {
            "model_path": "models/model.gguf",
            "embedding_model": "google/embeddinggemma-300m",
            "n_ctx": 4096,
            "n_batch": 512,
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "server_host": "127.0.0.1",
            "server_port": 8000,
            "api_key": "local-gguf-key"
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config
    
    def load_model(self):
        """Load the GGUF model"""
        try:
            self.model = Llama(
                model_path=self.config["model_path"],
                n_ctx=self.config["n_ctx"],
                n_batch=self.config["n_batch"],
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            logger.info(f"Model loaded successfully: {self.config['model_path']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_embedding_model(self):
        """Load the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.config["embedding_model"])
            logger.info(f"Embedding model loaded: {self.config['embedding_model']}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for given texts"""
        if not self.embedding_model:
            self.load_embedding_model()
        
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False) -> Any:
        """Generate chat completion"""
        if not self.model:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            if stream:
                return self._stream_chat_completion(messages)
            else:
                response = self.model.create_chat_completion(
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    top_p=self.config["top_p"],
                    repeat_penalty=self.config["repeat_penalty"],
                    stream=False
                )
                return response
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _stream_chat_completion(self, messages: List[Dict[str, str]]):
        """Stream chat completion"""
        try:
            for chunk in self.model.create_chat_completion(
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                top_p=self.config["top_p"],
                repeat_penalty=self.config["repeat_penalty"],
                stream=True
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def synthesize_and_store_memory(self, conversation: List[ChatMessage]):
        """Synthesize important facts from conversation and store as long-term memory"""
        try:
            # Extract key information from conversation
            conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation])
            
            # Generate summary using the model
            summary_prompt = f"""Analyze this conversation and extract the most important facts, preferences, and key information that should be remembered for future interactions:

{conversation_text}

Please provide a concise summary of the most important information to remember."""
            
            summary_response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = summary_response['choices'][0]['message']['content']
            
            # Generate embedding for the summary
            embedding = self.generate_embeddings([summary])[0]
            
            # Store memory
            memory = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "embedding": embedding,
                "conversation_length": len(conversation)
            }
            
            self.memories.append(memory)
            self.save_memories()
            
            logger.info("Memory synthesized and stored successfully")
            
        except Exception as e:
            logger.error(f"Memory synthesis error: {e}")
    
    def save_memories(self):
        """Save memories to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def load_memories(self):
        """Load memories from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.memories = json.load(f)
                logger.info(f"Loaded {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def search_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search memories using semantic similarity"""
        try:
            if not self.memories:
                return []
            
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Calculate similarities
            similarities = []
            for memory in self.memories:
                memory_embedding = memory['embedding']
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
                similarities.append((similarity, memory))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [mem for _, mem in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            return []
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform web search using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "")
                    })
                return results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

# API Models
class ChatMessageModel(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageModel]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str

class WebSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class MemorySearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

# GUI Application
class GGUFLoaderGUI:
    def __init__(self, loader: GGUFModelLoader):
        self.loader = loader
        self.root = tk.Tk()
        self.root.title("GGUF Model Loader")
        self.root.geometry("1200x800")
        
        self.setup_ui()
        self.current_conversation = []
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chat area
        chat_frame = ttk.LabelFrame(main_frame, text="Chat", padding="10")
        chat_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=20, width=60)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Input area
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.input_text = tk.Text(input_frame, height=4, width=60)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Load Embedding Model", command=self.load_embedding_model).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Start API Server", command=self.start_api_server).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Search Web", command=self.search_web).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Search Memories", command=self.search_memories).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Chat", command=self.clear_chat).pack(fill=tk.X, pady=2)
        
        # Canvas for visualization
        canvas_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        canvas_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def load_model(self):
        """Load the GGUF model"""
        try:
            self.loader.load_model()
            self.display_message("System", "Model loaded successfully!")
        except Exception as e:
            self.display_message("System", f"Error loading model: {e}")
    
    def load_embedding_model(self):
        """Load the embedding model"""
        try:
            self.loader.load_embedding_model()
            self.display_message("System", "Embedding model loaded successfully!")
        except Exception as e:
            self.display_message("System", f"Error loading embedding model: {e}")
    
    def send_message(self):
        """Send message to model"""
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return
        
        self.display_message("User", user_input)
        self.input_text.delete("1.0", tk.END)
        
        # Add to conversation
        self.current_conversation.append(ChatMessage("user", user_input))
        
        try:
            # Prepare messages for model
            messages = [{"role": msg.role, "content": msg.content} for msg in self.current_conversation]
            
            # Get response
            response = self.loader.chat_completion(messages, stream=False)
            assistant_message = response['choices'][0]['message']['content']
            
            self.display_message("Assistant", assistant_message)
            self.current_conversation.append(ChatMessage("assistant", assistant_message))
            
        except Exception as e:
            self.display_message("System", f"Error: {e}")
    
    def display_message(self, role: str, message: str):
        """Display message in chat area"""
        self.chat_display.insert(tk.END, f"\n[{role}]: {message}\n")
        self.chat_display.see(tk.END)
    
    def start_api_server(self):
        """Start the API server in a separate thread"""
        def run_server():
            try:
                self.loader.setup_api_routes()
                uvicorn.run(
                    self.loader.app,
                    host=self.loader.config["server_host"],
                    port=self.loader.config["server_port"],
                    log_level="info"
                )
            except Exception as e:
                logger.error(f"API server error: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        self.display_message("System", f"API server started at http://{self.loader.config['server_host']}:{self.loader.config['server_port']}")
    
    def search_web(self):
        """Perform web search"""
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            return
        
        try:
            results = self.loader.web_search(query)
            self.display_message("Web Search", f"Results for '{query}':")
            for i, result in enumerate(results, 1):
                self.display_message("Result", f"{i}. {result['title']}\n{result['snippet']}\nURL: {result['url']}")
        except Exception as e:
            self.display_message("System", f"Web search error: {e}")
    
    def search_memories(self):
        """Search long-term memories"""
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            return
        
        try:
            memories = self.loader.search_memories(query)
            self.display_message("Memory Search", f"Relevant memories for '{query}':")
            for i, memory in enumerate(memories, 1):
                self.display_message("Memory", f"{i}. {memory['summary'][:200]}...")
        except Exception as e:
            self.display_message("System", f"Memory search error: {e}")
    
    def clear_chat(self):
        """Clear current chat and synthesize memory"""
        if len(self.current_conversation) > 2:
            # Synthesize memory before clearing
            self.loader.synthesize_and_store_memory(self.current_conversation)
            self.display_message("System", "Chat cleared and memory synthesized!")
        
        self.current_conversation = []
        self.chat_display.delete("1.0", tk.END)
    
    def run(self):
        """Run the GUI application"""
        self.loader.load_memories()
        self.root.mainloop()

# Enhanced API Routes
def setup_api_routes(self):
    """Setup FastAPI routes for OpenAI-compatible API"""
    
    @self.app.get("/v1/models")
    async def list_models():
        """List available models"""
        return {
            "data": [
                {
                    "id": "gguf-model",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "gguf-loader",
                    "permission": []
                }
            ]
        }
    
    @self.app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Create chat completion"""
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        if request.stream:
            return StreamingResponse(
                self.chat_completion(messages, stream=True),
                media_type="text/event-stream"
            )
        else:
            response = self.chat_completion(messages, stream=False)
            return response
    
    @self.app.post("/v1/embeddings")
    async def create_embeddings(request: EmbeddingRequest):
        """Create embeddings"""
        try:
            embeddings = self.generate_embeddings(request.input)
            return {
                "data": [
                    {
                        "object": "embedding",
                        "embedding": emb,
                        "index": i
                    } for i, emb in enumerate(embeddings)
                ],
                "model": request.model,
                "object": "list"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @self.app.post("/v1/web/search")
    async def web_search(request: WebSearchRequest):
        """Perform web search"""
        results = self.web_search(request.query, request.max_results)
        return {"results": results}
    
    @self.app.post("/v1/memory/search")
    async def memory_search(request: MemorySearchRequest):
        """Search long-term memories"""
        memories = self.search_memories(request.query, request.top_k)
        return {"memories": memories}
    
    @self.app.get("/v1/memory/all")
    async def get_all_memories():
        """Get all memories"""
        return {"memories": self.memories}

# Add the setup_api_routes method to the class
GGUFModelLoader.setup_api_routes = setup_api_routes

# MCP Server Integration
class MCPIntegration:
    def __init__(self, loader: GGUFModelLoader):
        self.loader = loader
        self.mcp_server = FastMCP("GGUF Model MCP Server")
        self.setup_mcp_tools()
    
    def setup_mcp_tools(self):
        """Setup MCP tools"""
        
        @self.mcp_server.tool()
        async def search_web(query: str, max_results: int = 5) -> Dict:
            """Search the web for information"""
            return {"results": self.loader.web_search(query, max_results)}
        
        @self.mcp_server.tool()
        async def search_memories(query: str, top_k: int = 3) -> Dict:
            """Search long-term memories"""
            return {"memories": self.loader.search_memories(query, top_k)}
        
        @self.mcp_server.tool()
        async def get_model_info() -> Dict:
            """Get model information"""
            return {
                "model_path": self.loader.config["model_path"],
                "embedding_model": self.loader.config["embedding_model"],
                "context_length": self.loader.config["n_ctx"]
            }
        
        @self.mcp_server.tool()
        async def create_embeddings(texts: List[str]) -> Dict:
            """Create embeddings for texts"""
            return {"embeddings": self.loader.generate_embeddings(texts)}

# Main Application
def main():
    """Main application entry point"""
    # Initialize loader
    loader = GGUFModelLoader()
    
    # Load existing memories
    loader.load_memories()
    
    # Create and run GUI
    gui = GGUFLoaderGUI(loader)
    
    try:
        gui.run()
    except KeyboardInterrupt:
        logger.info("Application shutting down...")
        # Save memories before exit
        loader.save_memories()
        logger.info("Memories saved. Goodbye!")

if __name__ == "__main__":
    main()
