"""
FastAPI backend for Music → Movie Match.

Endpoints:
  POST /api/auth/spotify      → returns Spotify OAuth URL
  GET  /api/callback          → exchanges code, fetches Spotify data, returns session_id
  GET  /api/profile           → runs NLP + ML pipeline, returns personality profile
  GET  /api/recommendations   → returns top-10 movies + AI explanations
  GET  /api/clusters          → returns PCA cluster positions
  GET  /api/history/{user_id} → returns saved recommendation sessions
  GET  /api/health            → health check

Authentication flow:
  1. Frontend calls POST /api/auth/spotify  → gets auth_url
  2. User visits auth_url, Spotify redirects to /api/callback?code=...
  3. Callback returns session_id — frontend stores this
  4. Subsequent endpoints require ?session_id=<id> query param

Run with:
  uvicorn api:app --reload --port 8000
"""

import asyncio
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spotify_collector import (
    get_auth_url, exchange_code_for_token,
    get_spotify_client, collect_user_data,
)
from nlp_analyzer import (
    fetch_lyrics_for_tracks, compute_emotion_profile,
    build_user_feature_vector,
)
from personality_clustering import assign_personality, get_pca_coordinates
from movie_recommender import fetch_movies, load_or_build_embeddings, recommend_movies
from claude_explainer import explain_all_recommendations
from history_store import save_session, load_history

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Music → Movie Match API",
    description="Music personality analysis → personalized movie recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────────────────────
# Keyed by session_id (UUID). Resets on server restart.
# For production, replace with Redis or a DB.
_sessions: dict[str, dict] = {}

# ── Global ML caches (loaded once, reused across requests) ──────────────────
_movies: list | None = None
_embeddings: Any = None   # numpy array, loaded lazily

# ── Background analysis pipeline ─────────────────────────────────────────────
_analysis_status: dict = {}
_cancelled_sessions: set = set()
_executor = ThreadPoolExecutor(max_workers=2)

_step_messages = {
    "lyrics":       "Fetching lyrics from Genius API...",
    "emotions":     "Analyzing emotions with HuggingFace...",
    "personality":  "Determining your music personality...",
    "movies":       "Loading 5,000+ movies...",
    "explanations": "Writing AI explanations with Groq...",
    "complete":     "Analysis complete!",
    "error":        "Analysis failed.",
}


def _update_status(sid: str, step: str, pct: int, done: bool = False, error: str | None = None):
    _analysis_status[sid] = {
        "step":    step,
        "pct":     pct,
        "done":    done,
        "message": _step_messages.get(step, step),
        "error":   error,
    }


def _is_cancelled(session_id: str) -> bool:
    return session_id in _cancelled_sessions


def _run_pipeline(session_id: str):
    global _movies, _embeddings
    session = _sessions.get(session_id)
    if not session:
        return
    try:
        _update_status(session_id, "lyrics", 10)
        user_data = session["user_data"]
        tracks_with_lyrics = fetch_lyrics_for_tracks(user_data["tracks_df"], top_n=20)

        if _is_cancelled(session_id): return

        _update_status(session_id, "emotions", 40)
        emotion_profile = compute_emotion_profile(tracks_with_lyrics)
        feature_vector = build_user_feature_vector(user_data["tracks_df"], emotion_profile)
        session["emotion_profile"] = emotion_profile
        session["feature_vector"] = feature_vector

        if _is_cancelled(session_id): return

        _update_status(session_id, "personality", 62)
        personality = assign_personality(feature_vector)
        session["personality"] = personality

        if _is_cancelled(session_id): return

        _update_status(session_id, "movies", 72)
        if _movies is None:
            _movies = fetch_movies(total=5000)
        if _embeddings is None:
            _embeddings = load_or_build_embeddings(_movies)
        raw_recs = recommend_movies(personality["mood_description"], _movies, _embeddings, top_n=10)

        if _is_cancelled(session_id): return

        _update_status(session_id, "explanations", 88)
        top_artists = user_data["artists_df"]["artist_name"].tolist()[:5]
        top_genres  = user_data.get("all_genres", [])[:10]
        recommendations = explain_all_recommendations(personality, emotion_profile, raw_recs, top_artists, top_genres)
        session["recommendations"] = recommendations

        try:
            user_info = user_data["user_info"]
            save_session(
                user_info["user_id"] or "anonymous",
                user_info["display_name"],
                personality,
                recommendations,
                emotion_profile,
                user_data.get("all_genres", []),
            )
        except Exception:
            pass

        _update_status(session_id, "complete", 100, done=True)
    except Exception as e:
        _update_status(session_id, "error", 0, done=True, error=str(e))
    finally:
        _cancelled_sessions.discard(session_id)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /api/callback first.")
    return session


def _safe_float(val) -> float:
    """Converts numpy float32/64 to Python float for JSON serialization."""
    try:
        return float(val)
    except Exception:
        return 0.0


def _serialize_personality(p: dict) -> dict:
    return {
        "name":             p.get("name", ""),
        "emoji":            p.get("emoji", ""),
        "description":      p.get("description", ""),
        "confidence":       _safe_float(p.get("confidence", 0)),
        "mood_description": p.get("mood_description", ""),
        "cluster_id":       int(p.get("cluster_id", 0)),
        "all_similarities": {
            k: _safe_float(v)
            for k, v in p.get("all_similarities", {}).items()
        },
    }


def _serialize_movie(m: dict) -> dict:
    return {
        "movie_id":         m.get("movie_id"),
        "title":            m.get("title", ""),
        "overview":         m.get("overview", ""),
        "release_year":     m.get("release_year", ""),
        "genres":           m.get("genres", []),
        "vote_average":     _safe_float(m.get("vote_average", 0)),
        "similarity_score": _safe_float(m.get("similarity_score", 0)),
        "poster_url":       m.get("poster_url", ""),
        "explanation":      m.get("claude_explanation", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    """Health check."""
    return {"status": "ok", "version": "1.0.0"}


# ── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/api/auth/spotify")
def start_auth():
    """
    Returns the Spotify OAuth URL.
    Frontend should redirect the user to auth_url.
    """
    auth_url = get_auth_url()
    return {"auth_url": auth_url}


@app.get("/api/callback")
def callback(
    code: str = Query(..., description="Authorization code returned by Spotify"),
):
    """
    Exchanges the Spotify authorization code for an access token,
    fetches the user's Spotify data, creates a session, and redirects
    the browser to /dashboard?session_id=<id>.
    """
    try:
        token_info = exchange_code_for_token(code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {e}")

    try:
        sp = get_spotify_client(token_info)
        user_data = collect_user_data(sp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spotify data fetch failed: {e}")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "token_info":      token_info,
        "user_data":       user_data,
        "emotion_profile": None,
        "feature_vector":  None,
        "personality":     None,
        "recommendations": None,
    }

    # Store extra user context in the session so the dashboard can read it
    user_info  = user_data["user_info"]
    track_cols = ["track_name", "artist_name", "album_image"] if "album_image" in user_data["tracks_df"].columns else ["track_name", "artist_name"]
    tracks     = user_data["tracks_df"][track_cols].head(20).to_dict("records")
    artist_cols = ["artist_name", "genres", "image_url"] if "image_url" in user_data["artists_df"].columns else ["artist_name", "genres"]
    artists    = user_data["artists_df"][artist_cols].head(20).to_dict("records")
    _sessions[session_id]["_meta"] = {
        "user":        {"display_name": user_info["display_name"], "user_id": user_info["user_id"]},
        "top_tracks":  tracks,
        "top_artists": artists,
        "all_genres":  user_data.get("all_genres", []),
    }

    return RedirectResponse(url=f"/dashboard?session_id={session_id}")


# ── Profile (NLP + ML pipeline) ──────────────────────────────────────────────

@app.get("/api/profile")
def get_profile(session_id: str = Query(...)):
    """
    Runs the full NLP + ML pipeline:
      1. Fetches lyrics for top 20 tracks (Genius API)
      2. Analyzes emotions with HuggingFace DistilRoBERTa
      3. Builds feature vector (audio features + emotions)
      4. Assigns personality cluster via K-Means

    Results are cached in the session — calling again is instant.
    """
    session = _get_session(session_id)

    if session["personality"] is None:
        user_data = session["user_data"]
        try:
            tracks_with_lyrics = fetch_lyrics_for_tracks(user_data["tracks_df"], top_n=20)
            emotion_profile = compute_emotion_profile(tracks_with_lyrics)
            feature_vector = build_user_feature_vector(user_data["tracks_df"], emotion_profile)
            personality = assign_personality(feature_vector)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

        session["emotion_profile"] = emotion_profile
        session["feature_vector"]  = feature_vector
        session["personality"]     = personality

    p  = session["personality"]
    ep = session["emotion_profile"]
    td = session["user_data"]["tracks_df"]

    audio_cols = ["energy", "valence", "danceability", "acousticness",
                  "tempo", "instrumentalness", "speechiness"]
    audio_features = {
        col: _safe_float(td[col].mean())
        for col in audio_cols
        if col in td.columns
    }

    return {
        "personality":    _serialize_personality(p),
        "emotion_profile": {k: _safe_float(v) for k, v in ep.items()},
        "audio_features": audio_features,
    }


# ── Recommendations ───────────────────────────────────────────────────────────

@app.get("/api/recommendations")
def get_recommendations(session_id: str = Query(...)):
    """
    Recommends top-10 movies based on personality profile.
      1. Loads/caches 5,000 TMDB movies + Sentence Transformer embeddings
      2. Finds top-10 via cosine similarity
      3. Generates personalized explanations with Groq (Llama 3)

    Requires /api/profile to have been called first.
    Results are cached in the session.
    """
    global _movies, _embeddings

    session = _get_session(session_id)

    if session["personality"] is None:
        raise HTTPException(
            status_code=400,
            detail="Personality not computed yet. Call GET /api/profile first.",
        )

    if session["recommendations"] is None:
        try:
            if _movies is None:
                _movies = fetch_movies(total=5000)
            if _embeddings is None:
                _embeddings = load_or_build_embeddings(_movies)

            personality     = session["personality"]
            emotion_profile = session["emotion_profile"]
            user_data       = session["user_data"]

            raw_recs = recommend_movies(
                personality["mood_description"], _movies, _embeddings, top_n=10
            )
            top_artists = user_data["artists_df"]["artist_name"].tolist()[:5]
            recommendations = explain_all_recommendations(
                personality, emotion_profile, raw_recs, top_artists
            )
            session["recommendations"] = recommendations

            # Persist to history
            try:
                user_info = user_data["user_info"]
                save_session(
                    user_info["user_id"] or "anonymous",
                    user_info["display_name"],
                    personality,
                    recommendations,
                    emotion_profile,
                    user_data.get("all_genres", []),
                )
            except Exception:
                pass

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

    return {
        "personality": _serialize_personality(session["personality"]),
        "recommendations": [_serialize_movie(m) for m in session["recommendations"]],
    }


# ── Clusters (PCA) ────────────────────────────────────────────────────────────

@app.get("/api/clusters")
def get_clusters(session_id: str = Query(...)):
    """
    Returns 2D PCA cluster positions for all 6 personality types
    and the current user's projected position.

    Requires /api/profile to have been called first.
    """
    session = _get_session(session_id)

    if session["feature_vector"] is None:
        raise HTTPException(
            status_code=400,
            detail="Feature vector not computed yet. Call GET /api/profile first.",
        )

    pca_data = get_pca_coordinates(session["feature_vector"])

    centroids = [
        {
            "name":  c["name"],
            "emoji": c["emoji"],
            "color": c["color"],
            "x":     _safe_float(c["x"]),
            "y":     _safe_float(c["y"]),
        }
        for c in pca_data["centroid_coords"]
    ]

    return {
        "user_position": {
            "x": _safe_float(pca_data["user_x"]),
            "y": _safe_float(pca_data["user_y"]),
        },
        "centroids": centroids,
        "user_personality": session["personality"]["name"],
    }


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history/{user_id}")
def get_history(user_id: str):
    """
    Returns all saved recommendation sessions for a user.
    Sessions are stored locally in data/history/{user_id}.json.
    """
    history = load_history(user_id)
    return {"user_id": user_id, "sessions": history, "count": len(history)}


# ── Background analysis pipeline endpoints ────────────────────────────────────

@app.post("/api/analyze")
async def start_analysis(session_id: str = Query(...)):
    """Kicks off the background analysis pipeline for a session."""
    session = _get_session(session_id)
    if session.get("recommendations") is not None:
        _update_status(session_id, "complete", 100, done=True)
        return {"status": "already_done"}
    _update_status(session_id, "lyrics", 0)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, session_id)
    return {"status": "started"}


@app.post("/api/analyze/cancel")
def cancel_analysis(session_id: str = Query(...)):
    """Signals the background pipeline to stop after its current step."""
    _cancelled_sessions.add(session_id)
    return {"status": "cancellation_requested"}


@app.get("/api/analyze/status")
def get_analysis_status(session_id: str = Query(...)):
    """Returns the current progress of the background analysis pipeline."""
    _get_session(session_id)
    return _analysis_status.get(
        session_id,
        {"step": "not_started", "pct": 0, "done": False, "message": "Not started", "error": None},
    )


# ── Session info ──────────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}")
def get_session_status(session_id: str):
    """Returns which pipeline steps have been completed for a session."""
    session = _get_session(session_id)
    meta = session.get("_meta", {})

    # Sanitize top_artists: ensure genres is always a plain list of strings
    raw_artists = meta.get("top_artists", [])
    top_artists = []
    for a in raw_artists:
        genres_val = a.get("genres", [])
        if not isinstance(genres_val, list):
            genres_val = list(genres_val) if genres_val else []
        top_artists.append({**a, "genres": [str(g) for g in genres_val]})

    # Build all_genres from meta; fall back to extracting from artists
    all_genres = [str(g) for g in meta.get("all_genres", []) if g]
    if not all_genres:
        for a in top_artists:
            all_genres.extend(a.get("genres", []))

    return {
        "session_id":          session_id,
        "has_spotify_data":    session["user_data"] is not None,
        "has_profile":         session["personality"] is not None,
        "has_recommendations": session["recommendations"] is not None,
        "user":                meta.get("user", {}),
        "top_tracks":          meta.get("top_tracks", []),
        "top_artists":         top_artists,
        "all_genres":          all_genres,
    }


# ── Frontend page routes ──────────────────────────────────────────────────────

import os as _os
from fastapi.responses import FileResponse as _FileResponse

_frontend = _os.path.join(_os.path.dirname(__file__), "frontend")


@app.get("/dashboard")
def serve_dashboard():
    return _FileResponse(_os.path.join(_frontend, "dashboard.html"))


@app.get("/movie")
def serve_movie():
    return _FileResponse(_os.path.join(_frontend, "movie.html"))


@app.get("/profile")
def serve_profile():
    return _FileResponse(_os.path.join(_frontend, "profile.html"))


@app.get("/history")
def serve_history():
    return _FileResponse(_os.path.join(_frontend, "history.html"))


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# ── Static frontend (must be last — catches all unmatched routes) ─────────────
if _os.path.exists(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
