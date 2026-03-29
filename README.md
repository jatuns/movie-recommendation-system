---
title: musicmatch
emoji: 🎬
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# musicmatch

A full-stack data science app that analyzes your Spotify listening history, builds a music personality profile, and recommends movies that match your taste.

## How It Works

```
Spotify OAuth → Top Tracks + Top Artists
        ↓
Genius API → Lyrics (parallel, 5 threads)
        ↓
HuggingFace → 7-emotion analysis (joy, sadness, anger, fear, surprise, disgust, neutral)
        ↓
K-Means (k=6) → Personality cluster assignment + PCA coordinates
        ↓
TMDB 5000 movies → Sentence Transformer embeddings → cosine similarity → Top 10
        ↓
Groq / Llama 3.1 → Personalized explanation for each recommendation
        ↓
FastAPI backend + Plain HTML/CSS/JS frontend
```

## Personality Profiles

| Profile | Description |
|---------|-------------|
| 🖤 Dark & Introspective | Melancholy, low valence, high acousticness |
| ⚡ Energetic & Bold | High energy, fast tempo, powerful rhythms |
| ☀️ Feel-good & Social | High valence, danceable, joyful |
| 🌙 Moody & Atmospheric | Cinematic, instrumental, mysterious |
| 🎭 Sophisticated & Complex | Multi-layered emotions, intellectual depth |
| 🌿 Calm & Reflective | Low energy, peaceful, introspective |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

| API | Where to get it | Free? |
|-----|-----------------|-------|
| Spotify | [developer.spotify.com](https://developer.spotify.com/dashboard) | ✅ |
| Genius | [genius.com/api-clients](https://genius.com/api-clients) | ✅ |
| TMDB | [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) | ✅ |
| Groq | [console.groq.com](https://console.groq.com) | ✅ Free tier |

### 3. Add Spotify Redirect URI

In your Spotify Developer Dashboard, add this to your app's Redirect URIs:

```
http://127.0.0.1:8000/api/callback
```

### 4. Run

```bash
# FastAPI backend + HTML frontend (recommended)
uvicorn api:app --reload --port 8000
# → http://localhost:8000

# or Streamlit
streamlit run app.py
# → http://localhost:8501
```

## Project Structure

```
musicmatch/
├── api.py                        # FastAPI backend — 7 endpoints, background pipeline
├── app.py                        # Streamlit UI (alternative)
├── frontend/
│   ├── index.html                # Landing page
│   ├── dashboard.html            # Movie recommendations + analysis progress
│   ├── profile.html              # Full user profile with visualizations
│   ├── movie.html                # Movie detail page
│   └── history.html              # Past recommendation sessions
├── src/
│   ├── spotify_collector.py      # OAuth, top tracks/artists, genre estimation
│   ├── nlp_analyzer.py           # Genius lyrics fetch + HuggingFace emotion analysis
│   ├── personality_clustering.py # K-Means assignment, PCA visualization
│   ├── movie_recommender.py      # TMDB fetch, Sentence Transformer embeddings
│   ├── claude_explainer.py       # Groq/Llama 3 recommendation explanations
│   └── history_store.py          # JSON-based session history
├── models/                       # Trained K-Means + scaler (gitignored)
├── data/                         # Movie cache + embeddings (gitignored)
├── requirements.txt
└── .env.example
```

## Frontend Pages

| Page | Content |
|------|---------|
| **Dashboard** | 10 movie recommendations with posters, AI explanations, and match scores. Personality card and emotion profile on the side. |
| **Profile** | Top Artists with Spotify photos, Genre Cloud, Audio DNA radar chart, Personality Cluster Map (PCA scatter plot), Personality Affinity bars, Emotion Breakdown, Top Tracks with album covers. |
| **Movie** | Full movie detail, AI-generated explanation tied to your music personality, similar movies. |
| **History** | All past recommendation sessions with expandable movie lists. |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn |
| Auth | Spotipy (Spotify OAuth 2.0) |
| Lyrics | LyricsGenius (parallel fetching) |
| Emotion analysis | HuggingFace `j-hartmann/emotion-english-distilroberta-base` |
| Movie embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Clustering | scikit-learn K-Means, PCA, StandardScaler |
| LLM | Groq API — Llama 3.1-8b-instant (free tier) |
| Movie data | TMDB API (5000+ movies) |
| Frontend | Tailwind CSS (CDN), Lucide SVG icons, vanilla JS |

## Performance Notes

- **First run is slow** — building embeddings for 5000 movies takes a few minutes. Subsequent runs load from `data/movie_embeddings.npy`.
- **Lyrics fetching** runs in parallel via `ThreadPoolExecutor` (5 workers), cutting analysis time from ~90s to ~20s.
- **Emotion model** is cached at module level — loaded once, reused across all requests.
- **Pipeline cancellation** — if the user navigates away mid-analysis, the frontend sends a `sendBeacon` request to cancel the background pipeline gracefully.
