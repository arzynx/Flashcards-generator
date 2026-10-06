# Recallly — AI Flashcard Generator

Recallly turns pasted study material into a two-page visual revision deck. The app uses Gemini to summarise source notes first, then presents the ideas as memorable visual study sheets with relationships, prompts, and downloadable CSV data.

## Features

- Gemini-powered study blueprint and visual revision-sheet generation
- Offline demo fallback for reliable presentations
- Q&A card mode and CSV export
- Hand-drawn visual notes with diagrams, memory hooks, and application questions

## Run locally

```powershell
cd "C:\Users\Arzaan\Documents\Codex\2026-10-06\ai-flashcard-generator"
python -m streamlit run app.py
```

Open `http://localhost:8502`.

## Gemini setup

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your-key-here"
```

This file is intentionally ignored by Git. For Streamlit Community Cloud, add the same value in the deployment **Advanced settings → Secrets** field; never commit it to GitHub.
