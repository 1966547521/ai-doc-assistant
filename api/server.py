"""FastAPI process entry point for local and container startup."""

import os

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    """Start the API using the configured host and port."""
    load_dotenv()
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
