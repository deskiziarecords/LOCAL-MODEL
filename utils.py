# ========================================
# FILE 1: utils.py
# ========================================

"""
Utility functions for LOCAL-MODEL
"""

import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from datetime import datetime

logger = logging.getLogger('local_model.utils')


class PerfTracker:
    """Simple performance tracking"""
    
    def __init__(self):
        self.times: Dict[str, List[float]] = defaultdict(list)
        self.calls: Dict[str, int] = defaultdict(int)
    
    def track(self, name: str):
        """Context manager for tracking operation time"""
        class Timer:
            def __init__(self, tracker, op_name):
                self.tracker = tracker
                self.name = op_name
                self.start = None
            
            def __enter__(self):
                self.start = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start
                self.tracker.times[self.name].append(elapsed)
                self.tracker.calls[self.name] += 1
        
        return Timer(self, name)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        times = self.times.get(name, [])
        if not times:
            return {'count': 0, 'total': 0, 'avg': 0, 'min': 0, 'max': 0}
        
        return {
            'count': len(times),
            'total': sum(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def print_report(self):
        """Print performance report"""
        if not self.times:
            print("No performance data collected")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        for name in sorted(self.times.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Calls:    {stats['count']}")
            print(f"  Total:    {stats['total']:.3f}s")
            print(f"  Average:  {stats['avg']:.3f}s")
            print(f"  Min:      {stats['min']:.3f}s")
            print(f"  Max:      {stats['max']:.3f}s")
        
        print("\n" + "="*60)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


class FileUtils:
    """File operation utilities"""
    
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        """Ensure directory exists"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size_mb(path: Path) -> float:
        """Get file size in MB"""
        return Path(path).stat().st_size / (1024**2)
    
    @staticmethod
    def get_file_size_gb(path: Path) -> float:
        """Get file size in GB"""
        return Path(path).stat().st_size / (1024**3)
    
    @staticmethod
    def safe_load_json(path: Path, default: Any = None) -> Any:
        """Safely load JSON file with fallback"""
        try:
            if Path(path).exists():
                return json.loads(Path(path).read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load {path}: {e}")
        return default
    
    @staticmethod
    def safe_save_json(path: Path, data: Any) -> bool:
        """Safely save JSON file"""
        try:
            Path(path).write_text(json.dumps(data, indent=2))
            return True
        except Exception as e:
            logger.error(f"Could not save {path}: {e}")
            return False


class StringUtils:
    """String utility functions"""
    
    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = '...') -> str:
        """Truncate text to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def count_tokens_approx(text: str) -> int:
        """Rough token count estimate (1 token ≈ 4 chars)"""
        return len(text) // 4
    
    @staticmethod
    def format_bytes(bytes_val: int) -> str:
        """Format bytes as human readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}PB"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as human readable duration"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class Validators:
    """Input validation utilities"""
    
    @staticmethod
    def validate_port(port: int) -> bool:
        """Check if port is valid"""
        return 1 <= port <= 65535
    
    @staticmethod
    def validate_path(path: str) -> bool:
        """Check if path exists"""
        return Path(path).exists()
    
    @staticmethod
    def validate_file_extension(path: str, extensions: List[str]) -> bool:
        """Check if file has valid extension"""
        suffix = Path(path).suffix.lower()
        return suffix in [f".{ext.lower()}" for ext in extensions]


# Global performance tracker instance
perf_tracker = PerfTracker()


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    # Performance tracking
    with perf_tracker.track('test_operation'):
        time.sleep(0.1)
    
    with perf_tracker.track('test_operation'):
        time.sleep(0.2)
    
    perf_tracker.print_report()
    
    # String utilities
    print("\nString Utils Examples:")
    print(f"Truncate: {StringUtils.truncate('This is a very long text', 20)}")
    print(f"Token count: {StringUtils.count_tokens_approx('Hello world')}")
    print(f"Format bytes: {StringUtils.format_bytes(1536)}")
    print(f"Format duration: {StringUtils.format_duration(125)}")


# ========================================
# FILE 2: Makefile
# ========================================

# LOCAL-MODEL Makefile
# Quick commands for local development

.PHONY: run gui api test clean logs help list-models debug
.DEFAULT_GOAL := help

PYTHON := python3
PIP := pip3

help:
	@echo "LOCAL-MODEL - Available Commands:"
	@echo ""
	@echo "  make run              - Start with GUI (default)"
	@echo "  make gui              - Start GUI application"
	@echo "  make api              - Start API server only"
	@echo "  make debug            - Run in debug mode"
	@echo "  make list-models      - List available models"
	@echo "  make test             - Run tests"
	@echo "  make clean            - Clean cache files"
	@echo "  make logs             - Watch log files (tail -f)"
	@echo "  make install          - Install dependencies"
	@echo "  make freeze           - Update requirements.txt"
	@echo ""

run: gui

gui:
	$(PYTHON) main.py

api:
	$(PYTHON) main.py --no-gui

debug:
	$(PYTHON) main.py --debug

list-models:
	$(PYTHON) main.py --list-models

test:
	$(PYTHON) -m pytest tests/ -v

install:
	$(PIP) install -r requirements.txt

freeze:
	$(PIP) freeze > requirements.txt

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf .pytest_cache build dist

logs:
	tail -f logs/*.log

config:
	cat .env

show-config:
	$(PYTHON) config.py
