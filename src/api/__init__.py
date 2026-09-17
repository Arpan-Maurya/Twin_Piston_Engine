"""
API Package — REST & Real-Time Server (Module 20).

Exposes app factory and FastAPI application instance.
"""

from src.api.app import app, create_app

__all__ = ["app", "create_app"]
