# ================================================================
# MEDSENTRY FASTAPI
# ================================================================

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------
# IMPORT RAG PIPELINE
# ------------------------------------------------

from medsentry.rag.pipeline import medsentry_pipeline


# ------------------------------------------------
# FASTAPI APP
# ------------------------------------------------

app = FastAPI(
    title="MedSentry API",
    description="Grounded and safety-aware medical RAG API",
    version="1.0.0"
)


# ------------------------------------------------
# REQUEST MODEL
# ------------------------------------------------

class QueryRequest(BaseModel):

    query: str
    top_k: int = 3


# ------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "MedSentry API",
        "version": "1.0.0"
    }


# ------------------------------------------------
# ROOT
# ------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "MedSentry API is running",
        "docs": "/docs",
        "health": "/health"
    }


# ------------------------------------------------
# MEDICAL QUERY
# ------------------------------------------------

@app.post("/query")
def query_medical(request: QueryRequest):

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if request.top_k < 1 or request.top_k > 10:

        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10."
        )


    result = medsentry_pipeline(
        query=query,
        top_k=request.top_k
    )


    return result
