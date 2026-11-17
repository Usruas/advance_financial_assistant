# Ledgerline · Annual Report Intelligence

Local-first annual report analysis with FastAPI, Ollama, ChromaDB, and SQLite.

## Features

- Upload annual report PDFs and extract their text with PyMuPDF4LLM.
- Create semantic chunks using Ollama embeddings and index them in persistent ChromaDB.
- Ask questions over reports with retrieval-augmented Ollama chat and source filenames.
- Extract revenue, income, cash flow, assets, liabilities, risks, and growth drivers.
- Store extracted KPIs in local SQLite; no Azure OpenAI, Azure AI Search, or PostgreSQL credentials are required.
- Review reports, KPIs, and local model status in the dashboard.

## Requirements

- Python 3.12 recommended.
- Ollama running at the configured `OLLAMA_BASE_URL`.
- The configured chat and embedding models pulled into Ollama. Defaults:

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Run

```powershell
uv venv
uv pip install -r requirement.txt
Copy-Item .env.example .env
uv run uvicorn app:app --reload
```

Open http://localhost:8000. PDF uploads are saved and queued, then processed in the background. The dashboard polls the job and shows stages through semantic chunking, Chroma indexing, and KPI extraction. Jobs run one at a time in the API process to avoid competing local Ollama and Chroma workloads. Set `MAX_UPLOAD_MB` to change the upload limit. Chroma data, uploaded in-progress PDFs, and the KPI/job SQLite database live under `data/` by default.

For useful company/year metadata, name PDFs like `2024_Apple.pdf` or `Company_2024.pdf`.

## Deploy online

The repository includes a Dockerfile and Render Blueprint (`render.yaml`). To deploy:

1. Push this repository to GitHub, then create a new Blueprint on Render from that repository.
2. Set `OLLAMA_BASE_URL` in Render to the URL of an Ollama server reachable from Render. Do not use `http://localhost:11434`; that points back to the web service container. The server must have the configured chat and embedding models available.
3. Deploy the service. The Render disk keeps uploaded reports, the Chroma index, and the SQLite database across deploys.

The Blueprint uses Render's Starter web service and a 10 GB persistent disk, which incur charges. The app's API health endpoint does not verify Ollama connectivity; the first chat or report-processing request will. Treat uploaded financial reports as sensitive and use a private GitHub repository for source code. Local `.env`, databases, vector indexes, uploaded PDFs, and generated markdown are excluded from Git.

## API

- `GET /api/health` — app/model configuration status (does not actively ping Ollama).
- `GET /api/reports` — distinct source files from Chroma metadata.
- `GET /api/metrics` — extracted KPI rows.
- `POST /api/reports` — multipart PDF upload; returns a `job_id` after the file is saved.
- `GET /api/jobs/{job_id}` — processing stage and status (`queued`, `processing`, `completed`, or `failed`).
- `POST /api/chat` — JSON `{ "question": "...", "company": "Apple", "year": 2024 }` (filters optional).
- `POST /api/metrics/extract?company=Apple&year=2024` — extract and save KPIs.

## Notes

- Existing Chroma collections built with a different embedding model should be re-created before use; vector dimensions must match. Back up `data/chroma_db` before replacing it.
- `/api/health` confirms API configuration only. The first embedding or chat operation verifies that Ollama and the selected model are actually available.
- Semantic chunking embeds sentences to find boundaries; Chroma separately embeds the resulting chunks for search. The installed Ollama embedding integration sends each set as a batch, so removing either pass would harm the feature or retrieval quality. Retrieval context is capped at 8 chunks for KPI extraction (chat uses 6).
- BackgroundTasks keeps upload processing out of the request/response wait, but it does not make the model calculations themselves faster. For jobs that must survive server restarts, use a persistent worker queue such as RQ/Celery instead of the in-process worker.
