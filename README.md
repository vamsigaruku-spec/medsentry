# MedSentry

MedSentry is a safety-aware healthcare Retrieval-Augmented Generation (RAG)
prototype. It retrieves evidence from a small healthcare knowledge base,
checks for prompt-injection attempts, abstains when evidence is insufficient,
and displays the answer with evidence and safety status.

## Project structure

```text
medsentry/
├── data/
│   └── knowledge_base.json
├── rag/
│   └── pipeline.py
├── api/
│   └── main.py
└── ui/
    └── app.py
app.py
requirements.txt
README.md
.gitignore
```

## Run Streamlit locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run FastAPI locally

```bash
uvicorn medsentry.api.main:app --reload
```

API docs will be available at `/docs`.

## Streamlit Cloud

Use `app.py` as the main file.

## Notes

- The current pipeline uses the `all-MiniLM-L6-v2` sentence-transformer model.
- The knowledge base is intentionally small and is included in the repository.
- This is a prototype and does not replace professional medical advice.
