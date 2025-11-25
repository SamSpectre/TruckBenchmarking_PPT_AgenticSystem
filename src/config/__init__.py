"""
Configuration package.

Usage:
    from config import settings
    print(settings.openai_api_key)
"""

from .settings import settings, Settings

__all__ = ["settings", "Settings"]