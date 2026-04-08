# server/app.py - Entry point alias for openenv validate compatibility
# The actual app lives in app/main.py
from app.main import app

__all__ = ["app"]
