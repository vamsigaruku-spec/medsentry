# ================================================================
# MEDSENTRY RAG + GROQ LLM PIPELINE
# ================================================================

import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from medsentry.prompt import build_prompt
from medsentry.model import generate_answer
from medsentry.parser import parse_model_output


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"


# ================================================================
# LOAD KNOWLEDGE BASE
# ================================================================

def load_knowledge_base():

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

    if isinstance(data, dict):

        if "documents" in data:

            data = data["documents"]

        elif "knowledge_base" in data:

            data = data["knowledge_base"]

        else:

            data = [data]

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
# LOAD KNOWLEDGE BASE
# ================================================================

knowledge_base = [
    normalize_document(item)
    for item in load_knowledge_base()
]


documents = [
    item["text"]
    for item in knowledge_base
    if item["text"].strip()
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

    if not documents:

        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

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
# LLM ANSWER GENERATION
# ================================================================

def build_answer(
    query,
    evidence
):

    if not evidence:

        return {
            "answer": (
                "I could not find sufficient evidence in "
                "the MedSentry knowledge base to answer "
                "this question."
            ),
            "safety": (
                "Please consult a qualified healthcare "
                "professional for medical advice."
            ),
            "raw_response": ""
        }

    # ------------------------------------------------------------
    # Build structured prompt
    # ------------------------------------------------------------

    prompt = build_prompt(
        query=query,
        evidence=evidence
    )

    # ------------------------------------------------------------
    # Call Groq
    # ------------------------------------------------------------

    raw_response = generate_answer(
        prompt
    )

    # ------------------------------------------------------------
    # Parse + validate response
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

    start_time = time.perf_counter()

    query = str(
        query
    ).strip()


    # ============================================================
    # INPUT VALIDATION
    # ============================================================

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
    # PROMPT INJECTION
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
    # LLM + PARSER
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
            "grounded": False,
            "requires_clinician_review": True,
            "answer": (
                "MedSentry could not generate a valid "
                "structured response."
            ),
            "evidence_used": evidence,
            "safety_violations": [
                "llm_or_parser_error"
            ],
            "safety_note": (
                "The model response could not be "
                "validated successfully."
            ),
            "error": str(e),
            "latency_ms": round(
                latency,
                2
            )
        }


    # ============================================================
    # EXTRACT PARSED FIELDS
    # ============================================================

    answer = parsed_answer.get(
        "answer",
        ""
    ).strip()

    model_safety = parsed_answer.get(
        "safety",
        ""
    ).strip()

    raw_response = parsed_answer.get(
        "raw_response",
        ""
    )


    # ============================================================
    # EMPTY MODEL ANSWER
    # ============================================================

    if not answer:

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        return {
            "status": "ERROR",
            "query": query,
            "injection_detected": False,
            "safety_pass": False,
            "grounded": False,
            "requires_clinician_review": True,
            "answer": (
                "The model returned an empty answer."
            ),
            "evidence_used": evidence,
            "safety_violations": [
                "empty_model_answer"
            ],
            "safety_note": model_safety,
            "raw_model_response": raw_response,
            "latency_ms": round(
                latency,
                2
            )
        }


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
        "requires_clinician_review": clinician_review,
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
