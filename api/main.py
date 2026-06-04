"""FastAPI application entry point for AI Document Assistant control backend."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import auth, documents, analysis, admin

load_dotenv()

app = FastAPI(
    title="AI Document Assistant API",
    description="REST API for AI-powered document analysis, summarization, Q&A, and more",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(analysis.router)
app.include_router(admin.router)


@app.get("/", tags=["root"])
def root():
    return {
        "service": "AI Document Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
