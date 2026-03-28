# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app (primary UI)
streamlit run app.py
# → http://localhost:8501

# Run FastAPI backend (alternative)
uvicorn api:app --reload --port 8000
# → http://localhost:8000

# Copy and fill in environment variables
cp .env.example .env
```

## Environment Variables

Six API keys are required (see `.env.example`):
- `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` / `SPOTIFY_REDIRECT_URI` — from developer.spotify.com
- `GENIUS_ACCESS_TOKEN` — from genius.com/api-clients
- `TMDB_API_KEY` — from themoviedb.org
- `GROQ_API_KEY` — from console.groq.com (free tier, used for LLM explanations)

## Architecture

This app converts a user's Spotify listening history into movie recommendations via a 5-phase pipeline:

```
Spotify OAuth → Top Tracks/Artists
    ↓
Genius API → Lyrics → HuggingFace emotion analysis (j-hartmann/emotion-english-distilroberta-base)
    ↓
Audio features (estimated from genre map) + emotion vector → 12-dim feature vector
    ↓
K-Means clustering (k=6) → Personality type (0–5)
    ↓
TMDB 5000 movies → Sentence Transformer embeddings → cosine similarity to mood description
    ↓
Groq/Llama 3 → Per-movie personalized explanations
    ↓
Streamlit UI with history persistence
```

### Two UIs, One Pipeline

- **`app.py`** (Streamlit, 1235 lines) — primary UI; multi-page routing via `st.session_state.page`; calls `src/` modules directly
- **`api.py`** (FastAPI, 515 lines) — REST API for the standalone HTML frontend in `frontend/`; uses an in-memory session store and `ThreadPoolExecutor` for background pipeline execution; ML models cached globally on startup

### Core Modules (`src/`)

| File | Role |
|---|---|
| `spotify_collector.py` | Spotify OAuth + top tracks/artists; genre→audio-feature map (Spotify deprecated the audio features API) |
| `nlp_analyzer.py` | Genius lyrics fetch (rate-limited 2s delay) + HuggingFace 7-emotion analysis → feature vector |
| `personality_clustering.py` | Loads `models/kmeans_model.pkl` + `models/scaler.pkl`; assigns one of 6 personality profiles; computes PCA coordinates for visualization |
| `movie_recommender.py` | TMDB fetch + cache (`data/movies_cache.json`); builds/caches Sentence Transformer embeddings (`data/movie_embeddings.npy`); cosine similarity ranking |
| `claude_explainer.py` | Groq API calls (Llama 3) to explain each recommendation in terms of the user's personality + emotions + artists |
| `history_store.py` | Reads/writes `data/history/{user_id}.json` for session persistence |

### 6 Personality Clusters

| ID | Profile |
|---|---|
| 0 | 🖤 Dark & Introspective |
| 1 | ⚡ Energetic & Bold |
| 2 | ☀️ Feel-good & Social |
| 3 | 🌙 Moody & Atmospheric |
| 4 | 🎭 Sophisticated & Complex |
| 5 | 🌿 Calm & Reflective |

### Caching

First run is slow: building movie embeddings for 5000 TMDB movies takes several minutes. Subsequent runs load from `data/movie_embeddings.npy`. The `models/` directory (pre-trained K-Means + scaler) and `data/` directory are git-ignored.
