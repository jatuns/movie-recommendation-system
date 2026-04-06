# musicmatch

**Your music, your movies.**

musicmatch analyzes your Spotify listening history, builds a music personality profile, and recommends films that genuinely match your taste — powered by emotion AI and semantic search.

🔗 **Live demo:** [jatuns-musicmatch.hf.space](https://jatuns-musicmatch.hf.space)

> **Denemek ister misin?** Spotify API erişimi kısıtlı olduğundan uygulamayı test etmek için benimle iletişime geçmen gerekiyor.
> 📬 GitHub: [@jatuns](https://github.com/jatuns) · LinkedIn: [Barış Tuna Tuğrul](https://www.linkedin.com/in/baristunatugrul/)

---

## How It Works

```
Spotify OAuth → Top Tracks + Top Artists
        ↓
Genius API → Song lyrics (5 parallel threads)
        ↓
HuggingFace DistilRoBERTa → 7-emotion analysis
(joy · sadness · anger · fear · surprise · disgust · neutral)
        ↓
K-Means (k=6) → Personality cluster + PCA coordinates
        ↓
TMDB 5000 movies → Sentence Transformer embeddings → cosine similarity → Top 10
        ↓
Groq / Llama 3.1 → Personalized explanation for each recommendation
        ↓
FastAPI backend + HTML/CSS/JS frontend
```

Analysis takes about 30–60 seconds. No account needed beyond Spotify.

---

## Personality Profiles

| Profile | Description |
|---------|-------------|
| 🖤 Dark & Introspective | Melancholy, low valence, high acousticness |
| ⚡ Energetic & Bold | High energy, fast tempo, powerful rhythms |
| ☀️ Feel-good & Social | High valence, danceable, joyful |
| 🌙 Moody & Atmospheric | Cinematic, instrumental, mysterious |
| 🎭 Sophisticated & Complex | Multi-layered emotions, intellectual depth |
| 🌿 Calm & Reflective | Low energy, peaceful, introspective |

---

## Frontend Pages

| Page | What you see |
|------|-------------|
| **Landing** | Connect with Spotify — takes 30 seconds |
| **Dashboard** | 10 movie recommendations with posters, genre/IMDB/match badges, and AI-written explanations |
| **Profile** | Spotify profile photo, Top Artists, Genre Cloud, Audio DNA radar chart, Personality Cluster Map (PCA), Affinity bars, Emotion Breakdown, Top Tracks with album covers |
| **Movie Detail** | Full poster, synopsis, AI explanation tied to your music personality, and more recommendations |
| **History** | All past sessions with expandable movie lists |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn |
| Auth | Spotipy (Spotify OAuth 2.0) |
| Lyrics | LyricsGenius — parallel fetching via `ThreadPoolExecutor` |
| Emotion analysis | HuggingFace `j-hartmann/emotion-english-distilroberta-base` |
| Movie embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Clustering | scikit-learn K-Means, PCA, StandardScaler |
| LLM | Groq API — `llama-3.1-8b-instant` (free tier) |
| Movie data | TMDB API (5000+ movies) |
| Frontend | Tailwind CSS (CDN), vanilla JS, SVG icons |
| Deployment | HuggingFace Spaces (Docker, free tier) |

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/jatuns/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Fill in your keys:

| API | Where to get it | Free? |
|-----|-----------------|-------|
| Spotify | [developer.spotify.com](https://developer.spotify.com/dashboard) | ✅ |
| Genius | [genius.com/api-clients](https://genius.com/api-clients) | ✅ |
| TMDB | [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) | ✅ |
| Groq | [console.groq.com](https://console.groq.com) | ✅ Free tier |

### 3. Add Spotify Redirect URI

In your [Spotify Developer Dashboard](https://developer.spotify.com/dashboard), add to Redirect URIs:

```
http://127.0.0.1:8000/api/callback
```

### 4. Run

```bash
uvicorn api:app --reload --port 8000
# → http://localhost:8000
```

---

## Project Structure

```
musicmatch/
├── api.py                        # FastAPI backend — 7 endpoints, background pipeline
├── frontend/
│   ├── index.html                # Landing page
│   ├── dashboard.html            # Recommendations + emotion profile
│   ├── profile.html              # Full user profile with visualizations
│   ├── movie.html                # Movie detail page
│   └── history.html              # Past recommendation sessions
├── src/
│   ├── spotify_collector.py      # OAuth, top tracks/artists, genre inference
│   ├── nlp_analyzer.py           # Genius lyrics + HuggingFace emotion analysis
│   ├── personality_clustering.py # K-Means cluster assignment + PCA
│   ├── movie_recommender.py      # TMDB fetch + Sentence Transformer embeddings
│   ├── claude_explainer.py       # Groq/Llama 3 personalized explanations
│   └── history_store.py          # JSON session persistence
├── models/                       # Pre-trained K-Means + scaler
├── data/                         # Movies cache (embeddings rebuilt on first run)
├── Dockerfile                    # HuggingFace Spaces deployment
├── requirements.txt
└── .env.example
```

---

## Performance Notes

- **First run:** movie embeddings for 5000 titles are built on startup (~2–3 min). Cached to `data/movie_embeddings.npy` afterwards.
- **Lyrics fetching:** parallelized across 5 threads — cuts analysis from ~90s to ~20s.
- **Emotion model:** loaded once at startup, reused across all requests.
- **Pipeline cancellation:** navigating away mid-analysis sends a `sendBeacon` cancel request to stop background processing.

---

## License

MIT
