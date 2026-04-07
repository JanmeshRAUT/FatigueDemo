"""Compatibility module to keep older run scripts working.

This module intentionally re-exports the production FastAPI application
defined in app.py.
"""

from app import app
