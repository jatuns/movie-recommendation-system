# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (CPU-only PyTorch, avoids 2GB CUDA download)
pip install -r requirements.txt

# Run the app
uvicorn api:app --reload --port 8000
# → http://localhost:8000  (serves frontend/index.html + API)

# Copy and fill in environment variables
cp .env.example .env
```

## Environment Variables

Six API keys are required (see `.env.example`):
- `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` / `SPOTIFY_REDIRECT_URI` — from developer.spotify.com
- `GENIUS_ACCESS_TOKEN` — from genius.com/api-clients
- `TMDB_API_KEY` — from themoviedb.org
- `GROQ_API_KEY` — from console.groq.com (free tier, Llama 3.1 for explanations)

## Architecture

FastAPI backend + static HTML/CSS/JS frontend. The pipeline runs in a background thread per session.

```
Spotify OAuth → Top Tracks/Artists
    ↓
Genius API → Lyrics (5 parallel threads, top 20 tracks)
    ↓
HuggingFace DistilRoBERTa → 7-emotion vector (joy/sadness/anger/fear/surprise/disgust/neutral)
    ↓
Genre→audio-feature map + emotion vector → 12-dim feature vector
    ↓
K-Means (k=6) → Personality cluster (0–5)
    ↓
TMDB 5000 movies → Sentence Transformer embeddings → cosine similarity to mood description → Top 10
    ↓
Groq / llama-3.1-8b-instant → Per-movie personalized explanations
```

### `api.py` — FastAPI Backend

Single file, ~515 lines. Key design:
- **Session state**: in-memory `_sessions` dict keyed by UUID; resets on restart
- **Pipeline**: runs in `ThreadPoolExecutor` (2 workers) via `_run_pipeline(session_id)`; progress tracked in `_analysis_status` dict with step names (`lyrics`, `emotions`, `personality`, `movies`, `explanations`)
- **ML caches**: `_movies` and `_embeddings` are module-level globals, loaded once and reused across all sessions
- **Frontend**: `frontend/` directory served as static files at `/`

API flow: `POST /api/auth/spotify` → user visits OAuth URL → `GET /api/callback?code=...` returns `session_id` → all subsequent calls use `?session_id=<id>`

### `frontend/` — HTML/CSS/JS Pages

Five standalone HTML pages (no build step): `index.html`, `dashboard.html`, `profile.html`, `movie.html`, `history.html`. Each page polls or calls the API directly using the `session_id` stored in `localStorage`.

### Core Modules (`src/`)

| File | Role |
|---|---|
| `spotify_collector.py` | Spotify OAuth + top tracks/artists; `GENRE_FEATURE_MAP` estimates audio features since Spotify deprecated the audio-features API in 2024 |
| `nlp_analyzer.py` | Genius lyrics fetch (2s rate-limit delay) + HuggingFace 7-emotion analysis → combined feature vector |
| `personality_clustering.py` | Loads `models/kmeans_model.pkl` + `models/scaler.pkl`; assigns one of 6 personality profiles; computes PCA coordinates |
| `movie_recommender.py` | TMDB fetch + cache (`data/movies_cache.json`); builds/caches Sentence Transformer embeddings (`data/movie_embeddings.npy`) using `all-MiniLM-L6-v2`; cosine similarity ranking |
| `claude_explainer.py` | Groq API calls to explain each recommendation in terms of the user's personality + emotions + top artists |
| `history_store.py` | Reads/writes `data/history/{user_id}.json` for cross-session persistence |

### 6 Personality Clusters

| ID | Profile |
|---|---|
| 0 | 🖤 Dark & Introspective |
| 1 | ⚡ Energetic & Bold |
| 2 | ☀️ Feel-good & Social |
| 3 | 🌙 Moody & Atmospheric |
| 4 | 🎭 Sophisticated & Complex |
| 5 | 🌿 Calm & Reflective |

### Caching & First-Run Cost

First run builds embeddings for 5000 TMDB movies (several minutes). Subsequent runs load from `data/movie_embeddings.npy`. The `models/` and `data/` directories are git-ignored — `models/kmeans_model.pkl` and `models/scaler.pkl` must exist for personality clustering to work.

### Deployment

Deployed to HuggingFace Spaces via Docker (`Dockerfile`). The container runs `uvicorn api:app --host 0.0.0.0 --port 7860`. Live demo: [jatuns-musicmatch.hf.space](https://jatuns-musicmatch.hf.space)
