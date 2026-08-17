# Agent Specification - Buổi 07: RAG Pipeline System

## Workspace Boundaries
- Read-only: `rag_foundation/buoi_05/output/chunks/`, `rag_foundation/buoi_06/`
- Read/Write: `rag_foundation/buoi_07/`
- Strict rule: Do not modify Buổi 05 or Buổi 06 code or outputs.

## Python Interpreter
- Use `.venv` from Buổi 05 (`rag_foundation/buoi_05/.venv`).
- Do not create new venv.

## Data Source
- Load chunk JSON files from `rag_foundation/buoi_05/output/chunks/`.
- Do not perform OCR, PDF parsing, or re-chunking.

## Data & Validation Contract
Required fields for each chunk:
- `chunk_id`: non-empty string, unique across dataset
- `strategy`: string ('fixed-size', 'semantic', 'hierarchical')
- `source`: non-empty string
- `page_start`: integer >= 1
- `page_end`: integer >= 1, page_start <= page_end
- `text`: string (skip if text.strip() is empty)
