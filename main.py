"""
LOCAL-MODEL main entry point
Supports GUI and API server modes with CLI arguments
"""

import os
import sys
import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from config import Config


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with file and console output"""
    
    # Create logs directory
    log_dir = Path(Config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('local_model')
    level = logging.DEBUG if debug else logging.getLevelName(Config.LOG_LEVEL)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=Config.LOG_MAX_BYTES,
            backupCount=Config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(levelname)-8s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LOCAL-MODEL - Local LLM with GUI and API Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with GUI
  python main.py --no-gui           # API server only
  python main.py --model llama      # Load specific model
  python main.py --port 9000        # Custom API port
  python main.py --debug --no-gui   # Debug mode, API only
        """
    )
    
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run API server only (no GUI)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=Config.API_PORT,
        help=f'API server port (default: {Config.API_PORT})'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=Config.API_HOST,
        help=f'API server host (default: {Config.API_HOST})'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model file to load (name or path)'
    )
    
    parser.add_argument(
        '--gpu-layers',
        type=int,
        default=Config.GPU_LAYERS,
        help=f'GPU layers (-1 for all, default: {Config.GPU_LAYERS})'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom .env config file'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='LOCAL-MODEL 1.0.0'
    )
    
    return parser.parse_args()


def get_available_models(model_dir: str = None) -> list:
    """Get list of available GGUF models"""
    model_dir = Path(model_dir or Config.MODEL_PATH)
    
    if not model_dir.exists():
        return []
    
    models = list(model_dir.glob('*.gguf'))
    models.extend(model_dir.glob('*.bin'))
    models.extend(model_dir.glob('*.safetensors'))
    
    return sorted(models)


def print_models(logger: logging.Logger):
    """Print available models and exit"""
    models = get_available_models()
    
    if not models:
        print(f"\n❌ No models found in {Config.MODEL_PATH}")
        print(f"   Please add GGUF/BIN files to: {Path(Config.MODEL_PATH).absolute()}")
        sys.exit(1)
    
    print(f"\n✓ Found {len(models)} model(s):\n")
    for i, model in enumerate(models, 1):
        size = model.stat().st_size / (1024**3)  # Convert to GB
        print(f"  {i}. {model.name}")
        print(f"     Path: {model}")
        print(f"     Size: {size:.2f} GB\n")
    
    sys.exit(0)


def start_api_server(host: str, port: int, model_path: Optional[str] = None, 
                     logger: logging.Logger = None):
    """Start the API server"""
    if logger is None:
        logger = logging.getLogger('local_model')
    
    logger.info(f"Starting API server on {host}:{port}")
    
    try:
        # Import here to avoid issues if fastapi not installed
        from fastapi import FastAPI
        import uvicorn
        
        logger.info("API server dependencies loaded")
        
        # Create FastAPI app
        app = FastAPI(title="LOCAL-MODEL API")
        
        # TODO: Add your API routes here
        # Example:
        # @app.get("/health")
        # async def health():
        #     return {"status": "ok"}
        
        logger.info(f"Starting server at http://{host}:{port}")
        logger.info("API docs available at http://localhost:8000/docs")
        
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except ImportError as e:
        logger.error(f"FastAPI not installed: {e}")
        logger.error("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"API server error: {e}", exc_info=True)
        sys.exit(1)


def start_gui(model_path: Optional[str] = None, logger: logging.Logger = None):
    """Start the GUI application"""
    if logger is None:
        logger = logging.getLogger('local_model')
    
    logger.info("Starting GUI application")
    
    try:
        # Import here to allow running without GUI dependencies
        # TODO: Import your GUI module
        # from gui.main_window import MainWindow
        
        logger.info("GUI loaded successfully")
        
        # TODO: Start your GUI here
        # app = MainWindow(model_path=model_path)
        # app.run()
        
        print("GUI not yet implemented")
        logger.warning("GUI module not found - running API server instead")
        
    except ImportError as e:
        logger.error(f"GUI dependencies not available: {e}")
        logger.info("Falling back to API server mode")
        start_api_server(Config.API_HOST, Config.API_PORT, model_path, logger)
    except Exception as e:
        logger.error(f"GUI error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Override config with custom .env if provided
    if args.config:
        from dotenv import load_dotenv
        load_dotenv(args.config)
    
    # Setup logging
    logger = setup_logging(debug=args.debug)
    logger.info("="*50)
    logger.info("LOCAL-MODEL Starting")
    logger.info("="*50)
    
    # Validate configuration
    if not Config.validate():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Print config summary
    print(Config.get_summary())
    
    # List models if requested
    if args.list_models:
        print_models(logger)
    
    # Determine model path
    model_path = None
    if args.model:
        model_dir = Path(Config.MODEL_PATH)
        
        # If arg is just a name, look in model directory
        if not Path(args.model).exists():
            potential_path = model_dir / f"{args.model}.gguf"
            if potential_path.exists():
                model_path = str(potential_path)
                logger.info(f"Using model: {model_path}")
            else:
                available = get_available_models()
                logger.error(f"Model not found: {args.model}")
                logger.info(f"Available models: {[m.name for m in available]}")
                sys.exit(1)
        else:
            model_path = args.model
            logger.info(f"Using model: {model_path}")
    
    # Start application
    try:
        if args.no_gui:
            logger.info("Mode: API Server Only")
            start_api_server(args.host, args.port, model_path, logger)
        else:
            logger.info("Mode: GUI + API Server")
            start_gui(model_path, logger)
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        print("\n\nShutdown complete")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
