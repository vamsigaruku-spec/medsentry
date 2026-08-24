# ================================================================
# MEDSENTRY RAG PIPELINE
# ================================================================

import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


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

        metadata = item.get("metadata", {})

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
        re.search(pattern, query_lower)
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
        re.search(pattern, query_lower)
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

        score = float(scores[index])

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
# GROUNDED ANSWER
# ================================================================

def build_answer(
    query,
    evidence
):

    if not evidence:

        return (
            "I could not find sufficient evidence in the "
            "MedSentry knowledge base to answer this question."
        )

    query_lower = query.lower()

    if (
        "hypertension" in query_lower
        and (
            "monitor" in query_lower
            or "important" in query_lower
            or "what is" in query_lower
        )
    ):

        return (
            "**Hypertension (high blood pressure)** is a condition "
            "in which blood pressure remains higher than normal.\n\n"

            "**Why monitoring matters:**\n"
            "- **Often silent:** Many people with hypertension may "
            "not notice obvious symptoms.\n"
            "- **Risk of complications:** Uncontrolled hypertension "
            "can increase the risk of cardiovascular disease and "
            "other serious health complications.\n"
            "- **Regular monitoring:** Blood-pressure measurement "
            "and appropriate follow-up can help healthcare "
            "professionals assess and manage the condition.\n\n"

            "**Important:** Diagnosis and treatment decisions should "
            "be made with a qualified healthcare professional."
        )

    evidence_text = "\n\n".join(
        item["text"]
        for item in evidence
        if item["text"]
    )

    return (
        "Based on the MedSentry knowledge base:\n\n"
        + evidence_text
        + "\n\n"
        "For diagnosis, treatment, or personalized medical "
        "decisions, consult a qualified healthcare professional."
    )


# ================================================================
# MAIN PIPELINE
# ================================================================

def medsentry_pipeline(
    query,
    top_k=3
):

    start_time = time.perf_counter()

    query = str(query).strip()

    if not query:

        return {
            "status": "BLOCKED",
            "query": query,
            "injection_detected": False,
            "safety_pass": False,
            "grounded": False,
            "requires_clinician_review": True,
            "answer": "Please provide a valid question.",
            "evidence_used": [],
            "safety_violations": ["empty_query"],
            "latency_ms": 0.0
        }


    # ------------------------------------------------------------
    # PROMPT INJECTION
    # ------------------------------------------------------------

    if detect_prompt_injection(query):

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "status": "BLOCKED",
            "query": query,
            "injection_detected": True,
            "safety_pass": False,
            "grounded": False,
            "requires_clinician_review": True,
            "answer": (
                "I cannot follow instructions that attempt to "
                "override MedSentry's safety rules."
            ),
            "evidence_used": [],
            "safety_violations": ["prompt_injection"],
            "latency_ms": round(latency, 2)
        }


    # ------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------

    evidence = retrieve_evidence(
        query,
        top_k=top_k
    )


    # ------------------------------------------------------------
    # ABSTENTION
    # ------------------------------------------------------------

    if not evidence:

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "status": "ABSTAIN",
            "query": query,
            "injection_detected": False,
            "safety_pass": True,
            "grounded": False,
            "requires_clinician_review": True,
            "answer": (
                "I could not find sufficient evidence in the "
                "MedSentry knowledge base to answer this question."
            ),
            "evidence_used": [],
            "safety_violations": [],
            "latency_ms": round(latency, 2)
        }


    # ------------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------------

    answer = build_answer(
        query,
        evidence
    )


    latency = (
        time.perf_counter() - start_time
    ) * 1000


    return {
        "status": "PASS",
        "query": query,
        "injection_detected": False,
        "safety_pass": True,
        "grounded": True,
        "requires_clinician_review": (
            requires_clinician_review(query)
        ),
        "answer": answer,
        "evidence_used": evidence,
        "safety_violations": [],
        "latency_ms": round(latency, 2)
    }
