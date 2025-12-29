"""
Configuration management for LOCAL-MODEL
Supports environment variables and .env file
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file if it exists
ENV_FILE = Path(__file__).parent / '.env'
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class Config:
    """Central configuration management"""
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', './models')
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'mistral-7b')
    GPU_LAYERS = int(os.getenv('GPU_LAYERS', -1))  # -1 = all layers on GPU
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    TOP_P = float(os.getenv('TOP_P', 0.9))
    
    # API settings
    API_HOST = os.getenv('API_HOST', '127.0.0.1')
    API_PORT = int(os.getenv('API_PORT', 8000))
    API_KEY = os.getenv('API_KEY', 'your-api-key-here')
    
    # Feature flags
    ENABLE_WEB_SEARCH = os.getenv('ENABLE_WEB_SEARCH', 'false').lower() == 'true'
    ENABLE_MEMORY = os.getenv('ENABLE_MEMORY', 'true').lower() == 'true'
    ENABLE_EMBEDDINGS = os.getenv('ENABLE_EMBEDDINGS', 'false').lower() == 'true'
    
    # GUI settings
    ENABLE_GUI = os.getenv('ENABLE_GUI', 'true').lower() == 'true'
    GUI_THEME = os.getenv('GUI_THEME', 'auto')  # auto, light, dark
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', './logs/app.log')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))
    
    # Chat history
    CHAT_HISTORY_FILE = os.getenv('CHAT_HISTORY_FILE', 'chat_history.json')
    CHAT_HISTORY_ENABLED = os.getenv('CHAT_HISTORY_ENABLED', 'true').lower() == 'true'
    
    # Performance
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 300))
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 2))
    
    # Development
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration on startup"""
        errors = []
        
        # Check model path exists
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"MODEL_PATH does not exist: {cls.MODEL_PATH}")
        
        # Check API port is valid
        if not (1 <= cls.API_PORT <= 65535):
            errors.append(f"API_PORT must be between 1 and 65535, got {cls.API_PORT}")
        
        # Check log directory exists or can be created
        log_dir = Path(cls.LOG_FILE).parent
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create log directory: {e}")
        
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  ❌ {error}")
            return False
        
        return True
    
    @classmethod
    def get_summary(cls) -> str:
        """Get a summary of current configuration"""
        summary = """
╔════════════════════════════════════════╗
║     LOCAL-MODEL Configuration          ║
╚════════════════════════════════════════╝
Model Settings:
  • Model Path: {model_path}
  • Default Model: {default_model}
  • Max Tokens: {max_tokens}
  • Temperature: {temperature}
  • GPU Layers: {gpu_layers}

API Settings:
  • Host: {api_host}
  • Port: {api_port}
  • Features: {'Web Search' if {enable_web_search} else ''}
             {'Memory' if {enable_memory} else ''}
             {'Embeddings' if {enable_embeddings} else ''}

Interface:
  • GUI Enabled: {enable_gui}
  • Log Level: {log_level}
  • Debug Mode: {debug}

Chat:
  • History Enabled: {chat_history_enabled}
  • History File: {chat_history_file}
"""
        return summary.format(
            model_path=cls.MODEL_PATH,
            default_model=cls.DEFAULT_MODEL,
            max_tokens=cls.MAX_TOKENS,
            temperature=cls.TEMPERATURE,
            gpu_layers=cls.GPU_LAYERS,
            api_host=cls.API_HOST,
            api_port=cls.API_PORT,
            enable_web_search=cls.ENABLE_WEB_SEARCH,
            enable_memory=cls.ENABLE_MEMORY,
            enable_embeddings=cls.ENABLE_EMBEDDINGS,
            enable_gui=cls.ENABLE_GUI,
            log_level=cls.LOG_LEVEL,
            debug=cls.DEBUG,
            chat_history_enabled=cls.CHAT_HISTORY_ENABLED,
            chat_history_file=cls.CHAT_HISTORY_FILE,
        )
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {key: getattr(cls, key) for key in dir(cls) 
                if not key.startswith('_') and key.isupper()}


if __name__ == '__main__':
    print(Config.get_summary())
    print(f"\nConfiguration Valid: {Config.validate()}")
