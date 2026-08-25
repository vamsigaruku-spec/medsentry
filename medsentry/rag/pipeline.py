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

KNOWLEDGE_BASE_PATH = (
    DATA_DIR / "knowledge_base.json"
)


# ================================================================
# LOAD KNOWLEDGE BASE
# ================================================================

def load_knowledge_base():

    if not KNOWLEDGE_BASE_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge base not found: "
            f"{KNOWLEDGE_BASE_PATH}"
        )

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

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
            or "Unknown source"
        )

        source = (
            item.get("source")
            or metadata.get("source")
            or "Unknown source"
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
        "title": "Unknown source",
        "source": "Unknown source",
        "text": str(item)
    }


# ================================================================
# KNOWLEDGE BASE
# ================================================================

knowledge_base = [
    normalize_document(item)
    for item in load_knowledge_base()
]


# ================================================================
# SEARCHABLE DOCUMENTS
# ================================================================
#
# IMPORTANT:
# Keep the document and embedding together.
# This prevents index mismatch when empty documents exist.
# ================================================================

searchable_documents = [

    item

    for item in knowledge_base

    if item.get("text", "").strip()
]


documents = [

    item["text"]

    for item in searchable_documents
]


# ================================================================
# EMBEDDING MODEL
# ================================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ================================================================
# DOCUMENT EMBEDDINGS
# ================================================================

if documents:

    document_embeddings = (
        embedding_model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
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

    query_lower = str(
        query
    ).lower()

    return any(
        re.search(
            pattern,
            query_lower
        )
        for pattern in patterns
    )


# ================================================================
# CLINICIAN REVIEW DETECTION
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

    query_lower = str(
        query
    ).lower()

    return any(
        re.search(
            pattern,
            query_lower
        )
        for pattern in patterns
    )


# ================================================================
# RETRIEVE EVIDENCE
# ================================================================

def retrieve_evidence(
    query,
    top_k=3,
    min_score=0.30
):

    if not documents:

        return []

    try:

        top_k = int(top_k)

    except (
        TypeError,
        ValueError
    ):

        top_k = 3

    top_k = max(
        1,
        min(top_k, len(documents))
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
    )

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

        document = searchable_documents[
            int(index)
        ]

        results.append({

            "title": document[
                "title"
            ],

            "source": document[
                "source"
            ],

            "text": document[
                "text"
            ],

            "score": round(
                score,
                4
            )
        })

    return results


# ================================================================
# BUILD LLM ANSWER
# ================================================================

def build_answer(
    query,
    evidence
):

    if not evidence:

        return {

            "answer": (
                "I could not find sufficient "
                "evidence in the MedSentry "
                "knowledge base to answer "
                "this question."
            ),

            "safety": (
                "Please consult a qualified "
                "healthcare professional for "
                "medical advice."
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
    # CALL GROQ
    # ------------------------------------------------------------

    raw_response = generate_answer(
        prompt
    )

    # ------------------------------------------------------------
    # PARSE RESPONSE
    # ------------------------------------------------------------

    parsed = parse_model_output(
        raw_response
    )

    if not parsed.get("answer"):

        raise ValueError(
            "Parser returned an empty answer."
        )

    return parsed


# ================================================================
# MAIN MEDSENTRY PIPELINE
# ================================================================

def medsentry_pipeline(
    query,
    top_k=3
):

    start_time = (
        time.perf_counter()
    )

    query = str(
        query or ""
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
                "Please provide a valid "
                "medical question."
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
                "I cannot follow instructions "
                "that attempt to override "
                "MedSentry's safety rules."
            ),

            "evidence_used": [],

            "safety_violations": [
                "prompt_injection"
            ],

            "safety_note": (
                "The request was blocked because "
                "it contained a prompt-injection "
                "attempt."
            ),

            "latency_ms": round(
                latency,
                2
            )
        }

    # ============================================================
    # RETRIEVE EVIDENCE
    # ============================================================

    try:

        evidence = retrieve_evidence(
            query=query,
            top_k=top_k
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
                "MedSentry encountered an "
                "error while retrieving evidence."
            ),

            "evidence_used": [],

            "safety_violations": [
                "retrieval_error"
            ],

            "safety_note": (
                "The knowledge retrieval stage "
                "could not be completed."
            ),

            "error": (
                f"{type(e).__name__}: {e}"
            ),

            "latency_ms": round(
                latency,
                2
            )
        }

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
                "I could not find sufficient "
                "evidence in the MedSentry "
                "knowledge base to answer "
                "this question."
            ),

            "evidence_used": [],

            "safety_violations": [],

            "safety_note": (
                "The system abstained because "
                "sufficient supporting evidence "
                "was not retrieved."
            ),

            "latency_ms": round(
                latency,
                2
            )
        }

    # ============================================================
    # GROQ + PARSER
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
                "MedSentry could not generate "
                "a valid response."
            ),

            "evidence_used": evidence,

            "safety_violations": [
                "llm_or_parser_error"
            ],

            "safety_note": (
                "The model response could not "
                "be processed successfully."
            ),

            # IMPORTANT FOR DEBUGGING
            "error": (
                f"{type(e).__name__}: {e}"
            ),

            "raw_model_response": "",

            "latency_ms": round(
                latency,
                2
            )
        }

    # ============================================================
    # EXTRACT RESPONSE
    # ============================================================

    answer = str(
        parsed_answer.get(
            "answer",
            ""
        )
    ).strip()

    model_safety = str(
        parsed_answer.get(
            "safety",
            ""
        )
    ).strip()

    raw_response = str(
        parsed_answer.get(
            "raw_response",
            ""
        )
    )

    # ============================================================
    # EMPTY RESPONSE
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
                "The model returned an "
                "empty answer."
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
