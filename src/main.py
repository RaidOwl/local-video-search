#!/usr/bin/env python3
"""Main entry point for the Local Video Search & Summarization API."""

import uvicorn
from .config import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()