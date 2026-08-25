# ================================================================
# MEDSENTRY RAG + LLM PIPELINE
# ================================================================

import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# GenAI components
from medsentry.prompt import build_prompt
from medsentry.model import generate_answer
from medsentry.parser import parse_model_output


# ================================================================
# PATHS
# ================================================================

# pipeline.py is located at:
# medsentry/rag/pipeline.py
#
# Therefore:
# parent      = medsentry/rag
# parent.parent = medsentry

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"


# ================================================================
# LOAD KNOWLEDGE BASE
# ================================================================

def load_knowledge_base():
    """
    Load the MedSentry knowledge base from JSON.
    """

    if not KNOWLEDGE_BASE_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}"
        )

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # ------------------------------------------------------------
    # Support different JSON structures
    # ------------------------------------------------------------

    if isinstance(data, dict):

        if "documents" in data:

            data = data["documents"]

        elif "knowledge_base" in data:

            data = data["knowledge_base"]

        else:

            data = [data]

    # ------------------------------------------------------------
    # Make sure result is a list
    # ------------------------------------------------------------

    if not isinstance(data, list):

        data = [data]

    return data


# ================================================================
# EMBEDDING MODEL
# ================================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ================================================================
# NORMALIZE DOCUMENT
# ================================================================

def normalize_document(item):
    """
    Convert different knowledge-base formats
    into one consistent structure.
    """

    if isinstance(item, dict):

        metadata = item.get(
            "metadata",
            {}
        )

        if not isinstance(metadata, dict):

            metadata = {}

        title = (
            item.get("title")
            or metadata.get("title")
            or "Unknown"
        )

        source = (
            item.get("source")
            or metadata.get("source")
            or "Unknown"
        )

        text = (
            item.get("text")
            or item.get("content")
            or metadata.get("text")
            or metadata.get("content")
            or ""
        )

        return {
            "title": str(title),
            "source": str(source),
            "text": str(text)
        }

    return {
        "title": "Unknown",
        "source": "Unknown",
        "text": str(item)
    }


# ================================================================
# LOAD + EMBED DOCUMENTS
# ================================================================

knowledge_base = [
    normalize_document(item)
    for item in load_knowledge_base()
]


documents = [
    item["text"]
    for item in knowledge_base
]


# ================================================================
# CREATE DOCUMENT EMBEDDINGS
# ================================================================

if documents:

    document_embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

else:

    document_embeddings = np.empty(
        (0, 384)
    )


# ================================================================
# PROMPT INJECTION DETECTION
# ================================================================

def detect_prompt_injection(query):
    """
    Detect common prompt-injection and jailbreak attempts.
    """

    patterns = [

        r"ignore\s+(all\s+)?previous\s+instructions",

        r"ignore\s+(the\s+)?previous\s+instructions",

        r"forget\s+(all\s+)?previous\s+instructions",

        r"disregard\s+(all\s+)?previous\s+instructions",

        r"you\s+are\s+now\s+a\s+doctor",

        r"act\s+as\s+a\s+doctor",

        r"pretend\s+you\s+are\s+a\s+doctor",

        r"give\s+me\s+a\s+prescription",

        r"prescribe\s+(medication|medicine|drugs)",

        r"override\s+(your\s+)?instructions",

        r"reveal\s+(your\s+)?system\s+prompt",

        r"show\s+(me\s+)?your\s+system\s+prompt",

        r"bypass\s+(the\s+)?safety",

        r"jailbreak"
    ]

    query_lower = query.lower()

    return any(
        re.search(
            pattern,
            query_lower
        )
        for pattern in patterns
    )


# ================================================================
# CLINICIAN REVIEW
# ================================================================

def requires_clinician_review(query):
    """
    Identify questions that require professional
    healthcare review.
    """

    patterns = [

        r"\bdiagnos(e|is|ing)\b",

        r"\bprescription\b",

        r"\bprescribe\b",

        r"\bmedication\s+dose\b",

        r"\bdosage\b",

        r"\bshould\s+i\s+take\b",

        r"\bshould\s+i\s+stop\b",

        r"\bshould\s+i\s+start\b",

        r"\bchange\s+my\s+medication\b",

        r"\bwhat\s+medicine\s+should\s+i\b",

        r"\btreatment\s+for\s+me\b"
    ]

    query_lower = query.lower()

    return any(
        re.search(
            pattern,
            query_lower
        )
        for pattern in patterns
    )


# ================================================================
# RAG RETRIEVAL
# ================================================================

def retrieve_evidence(
    query,
    top_k=3,
    min_score=0.30
):
    """
    Retrieve the most semantically relevant
    knowledge-base documents.
    """

    if not documents:

        return []

    # ------------------------------------------------------------
    # Encode user query
    # ------------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # ------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------

    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    # ------------------------------------------------------------
    # Rank documents
    # ------------------------------------------------------------

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    # ------------------------------------------------------------
    # Select top-K evidence
    # ------------------------------------------------------------

    for index in ranked_indices[:top_k]:

        score = float(
            scores[index]
        )

        if score < min_score:

            continue

        document = knowledge_base[index]

        results.append({
            "title": document["title"],
            "source": document["source"],
            "text": document["text"],
            "score": score
        })

    return results


# ================================================================
# LLM GROUNDED ANSWER
# ================================================================

def build_answer(
    query,
    evidence
):
    """
    Build a grounded prompt, call Gemini,
    and parse the model response.
    """

    # ------------------------------------------------------------
    # No evidence
    # ------------------------------------------------------------

    if not evidence:

        return {
            "answer": (
                "I could not find sufficient evidence in the "
                "MedSentry knowledge base to answer this question."
            ),

            "safety": (
                "Please consult a qualified healthcare "
                "professional for medical advice."
            ),

            "raw_response": ""
        }

    # ------------------------------------------------------------
    # BUILD PROMPT
    # ------------------------------------------------------------

    prompt = build_prompt(
        query=query,
        evidence=evidence
    )

    # ------------------------------------------------------------
    # CALL GEMINI
    # ------------------------------------------------------------

    raw_response = generate_answer(
        prompt
    )

    # ------------------------------------------------------------
    # PARSE MODEL RESPONSE
    # ------------------------------------------------------------

    parsed = parse_model_output(
        raw_response
    )

    return parsed


# ================================================================
# MAIN MEDSENTRY PIPELINE
# ================================================================

def medsentry_pipeline(
    query,
    top_k=3
):
    """
    Complete MedSentry pipeline.

    Flow:

    User Query
        ↓
    Input Validation
        ↓
    Prompt Injection Detection
        ↓
    RAG Retrieval
        ↓
    Evidence Grounding
        ↓
    Prompt Template
        ↓
    Gemini LLM
        ↓
    Output Parser
        ↓
    Safety / Clinician Review
        ↓
    Structured Result
    """

    start_time = time.perf_counter()

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    query = str(
        query
    ).strip()

    if not query:

        return {
            "status": "BLOCKED",
            "query": query,
            "injection_detected": False,
            "safety_pass": False,
            "grounded": False,
            "requires_clinician_review": True,

            "answer": (
                "Please provide a valid medical question."
            ),

            "evidence_used": [],

            "safety_violations": [
                "empty_query"
            ],

            "safety_note": "",

            "latency_ms": 0.0
        }


    # ============================================================
    # PROMPT INJECTION CHECK
    # ============================================================

    if detect_prompt_injection(query):

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        return {
            "status": "BLOCKED",
            "query": query,
            "injection_detected": True,
            "safety_pass": False,
            "grounded": False,
            "requires_clinician_review": True,

            "answer": (
                "I cannot follow instructions that attempt "
                "to override MedSentry's safety rules."
            ),

            "evidence_used": [],

            "safety_violations": [
                "prompt_injection"
            ],

            "safety_note": (
                "The request was blocked because it "
                "contained a prompt-injection attempt."
            ),

            "latency_ms": round(
                latency,
                2
            )
        }


    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    evidence = retrieve_evidence(
        query=query,
        top_k=top_k
    )


    # ============================================================
    # ABSTENTION
    # ============================================================

    if not evidence:

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        return {
            "status": "ABSTAIN",
            "query": query,
            "injection_detected": False,
            "safety_pass": True,
            "grounded": False,
            "requires_clinician_review": True,

            "answer": (
                "I could not find sufficient evidence in "
                "the MedSentry knowledge base to answer "
                "this question."
            ),

            "evidence_used": [],

            "safety_violations": [],

            "safety_note": (
                "The system abstained because sufficient "
                "supporting evidence was not retrieved."
            ),

            "latency_ms": round(
                latency,
                2
            )
        }


    # ============================================================
    # LLM ANSWER GENERATION
    # ============================================================

    try:

        parsed_answer = build_answer(
            query=query,
            evidence=evidence
        )

    except Exception as e:

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        return {
            "status": "ERROR",
            "query": query,
            "injection_detected": False,
            "safety_pass": False,
            "grounded": True,
            "requires_clinician_review": True,

            "answer": (
                "MedSentry could not generate the answer "
                "because the language-model service failed."
            ),

            "evidence_used": evidence,

            "safety_violations": [
                "llm_generation_error"
            ],

            "safety_note": (
                "The model service returned an error. "
                "Please try again."
            ),

            "error": str(e),

            "latency_ms": round(
                latency,
                2
            )
        }


    # ============================================================
    # PARSED ANSWER
    # ============================================================

    answer = parsed_answer.get(
        "answer",
        "No answer generated."
    )

    model_safety = parsed_answer.get(
        "safety",
        ""
    )

    raw_response = parsed_answer.get(
        "raw_response",
        ""
    )


    # ============================================================
    # CLINICIAN REVIEW
    # ============================================================

    clinician_review = (
        requires_clinician_review(
            query
        )
    )


    # ============================================================
    # FINAL LATENCY
    # ============================================================

    latency = (
        time.perf_counter()
        - start_time
    ) * 1000


    # ============================================================
    # FINAL RESULT
    # ============================================================

    return {
        "status": "PASS",

        "query": query,

        "injection_detected": False,

        "safety_pass": True,

        "grounded": True,

        "requires_clinician_review": (
            clinician_review
        ),

        "answer": answer,

        "evidence_used": evidence,

        "safety_violations": [],

        "safety_note": model_safety,

        "raw_model_response": raw_response,

        "latency_ms": round(
            latency,
            2
        )
    }
