import sys
from pathlib import Path

import streamlit as st


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# IMPORT PIPELINE
# ================================================================

from medsentry.rag.pipeline import medsentry_pipeline


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="MedSentry",
    page_icon="🩺",
    layout="wide"
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.answer-box {
    padding: 22px;
    border-radius: 15px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-top: 15px;
}

.status-box {
    padding: 15px;
    border-radius: 12px;
    background: #f1f5f9;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ================================================================
# HEADER
# ================================================================

st.markdown("""
<div class="hero">

<h1>🩺 MedSentry</h1>

<p>
Grounded medical question answering with
retrieval, safety checks and evidence.
</p>

</div>
""", unsafe_allow_html=True)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Evidence documents",
        min_value=1,
        max_value=5,
        value=3
    )

    st.divider()

    st.subheader("Safety")

    st.success("✓ Grounding enabled")
    st.success("✓ Safety checks enabled")
    st.success("✓ Prompt-injection detection")


# ================================================================
# QUERY INPUT
# ================================================================

st.subheader("Ask MedSentry")

query = st.text_area(
    "Medical question",
    placeholder="Example: What is hypertension and why is monitoring important?",
    height=120
)


# ================================================================
# ASK BUTTON
# ================================================================

if st.button(
    "🔍 Ask MedSentry",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving evidence and checking safety..."):

            try:

                result = medsentry_pipeline(
                    query=query.strip(),
                    top_k=top_k
                )

                st.session_state["result"] = result

            except Exception as e:

                st.error(f"Pipeline error: {e}")


# ================================================================
# DISPLAY RESULT
# ================================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    # ------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------

    st.subheader("Pipeline Status")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if result.get("status") == "PASS":
            st.success("PASS")
        else:
            st.error(result.get("status", "UNKNOWN"))

    with c2:
        if result.get("grounded"):
            st.success("Grounded")
        else:
            st.warning("Not Grounded")

    with c3:
        if result.get("safety_pass"):
            st.success("Safety PASS")
        else:
            st.error("Safety FAIL")

    with c4:
        if result.get("injection_detected"):
            st.error("Injection")
        else:
            st.success("No Injection")


    # ------------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------------

    st.subheader("Answer")

    st.markdown(
        f"""
        <div class="answer-box">
        {result.get("answer", "No answer generated.")}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------
    # EVIDENCE
    # ------------------------------------------------------------

    st.subheader("📚 Evidence Used")

    evidence = result.get("evidence_used", [])

    if evidence:

        for i, item in enumerate(evidence, 1):

            with st.expander(
                f"{i}. {item.get('title', 'Unknown source')}"
            ):

                st.write(
                    f"**Source:** {item.get('source', 'Unknown')}"
                )

                st.write(
                    f"**Similarity Score:** "
                    f"{item.get('score', 0):.4f}"
                )

                st.write(
                    item.get(
                        "text",
                        "No evidence text available."
                    )
                )

    else:

        st.info("No evidence retrieved.")


    # ------------------------------------------------------------
    # SAFETY INFORMATION
    # ------------------------------------------------------------

    st.subheader("🛡️ Safety Information")

    violations = result.get(
        "safety_violations",
        []
    )

    if violations:

        for violation in violations:
            st.error(str(violation))

    else:

        st.success("No safety violations detected.")


    # ------------------------------------------------------------
    # CLINICIAN REVIEW
    # ------------------------------------------------------------

    if result.get("requires_clinician_review"):

        st.warning(
            "⚠️ This response requires review by a "
            "qualified healthcare professional."
        )


    # ------------------------------------------------------------
    # LATENCY
    # ------------------------------------------------------------

    latency = result.get("latency_ms")

    if latency is not None:

        st.caption(
            f"Pipeline latency: {latency:.2f} ms"
        )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "MedSentry is an evidence-grounded prototype. "
    "It does not replace professional medical advice."
)
