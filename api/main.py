"""FastAPI application entry point for AI Document Assistant control backend."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.routes import auth, documents, analysis, admin

load_dotenv()

def _cors_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv("API_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if "*" in origins:
        raise ValueError("API_CORS_ORIGINS must contain explicit origins")
    return origins


def create_app() -> FastAPI:
    application = FastAPI(
        title="AI Document Assistant API",
        description="REST API for AI-powered document analysis, summarization, Q&A, and more",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    origins = _cors_origins()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @application.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> PlainTextResponse:
        return PlainTextResponse(
            str(exc.detail), status_code=exc.status_code, headers=exc.headers
        )

    application.include_router(auth.router)
    application.include_router(documents.router)
    application.include_router(analysis.router)
    application.include_router(admin.router)

    @application.get("/", tags=["root"])
    def root():
        return {
            "service": "AI Document Assistant API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
