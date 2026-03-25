"""
Phase 4: Movie Recommendation Engine
Fetches movie data from TMDB API, builds embeddings with Sentence Transformers,
and recommends the top 10 movies via cosine similarity with the personality profile.
"""

from __future__ import annotations

import os
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
CACHE_PATH = "data/movies_cache.json"
EMBEDDINGS_PATH = "data/movie_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and effective sentence embedding model


def _tmdb_get(endpoint: str, params: dict = None) -> dict:
    """Makes a TMDB API request."""
    api_key = os.getenv("TMDB_API_KEY")
    base_params = {"api_key": api_key, "language": "en-US"}
    if params:
        base_params.update(params)
    r = requests.get(f"{TMDB_BASE}{endpoint}", params=base_params, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_movies(total: int = 5000) -> list[dict]:
    """
    Fetches a movie list from the TMDB popular movies endpoint.
    20 movies per page, fetching total / 20 pages.
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print("Loading movies from cache...")
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(f"Fetching {total} movies from TMDB...")
    movies = []
    pages = total // 20

    for page in range(1, pages + 1):
        if page % 50 == 0:
            print(f"  Page {page}/{pages}")
        try:
            data = _tmdb_get("/discover/movie", {
                "page": page,
                "sort_by": "popularity.desc",
                "vote_count.gte": 100,
                "with_original_language": "en",
            })
            for m in data.get("results", []):
                if not m.get("overview"):
                    continue
                movies.append({
                    "movie_id": m["id"],
                    "title": m["title"],
                    "overview": m["overview"],
                    "genres": [],  # genre names filled in below
                    "genre_ids": m.get("genre_ids", []),
                    "release_year": (m.get("release_date") or "")[:4],
                    "poster_path": m.get("poster_path"),
                    "vote_average": m.get("vote_average", 0),
                    "popularity": m.get("popularity", 0),
                })
        except Exception as e:
            print(f"  Page {page} error: {e}")
            continue

    # Add genre names
    genre_map = _get_genre_map()
    for m in movies:
        m["genres"] = [genre_map.get(gid, "") for gid in m["genre_ids"] if gid in genre_map]

    with open(CACHE_PATH, "w") as f:
        json.dump(movies, f, ensure_ascii=False)

    print(f"Saved {len(movies)} movies.")
    return movies


def _get_genre_map() -> dict:
    """TMDB genre id → name mapping."""
    data = _tmdb_get("/genre/movie/list")
    return {g["id"]: g["name"] for g in data.get("genres", [])}


def _build_movie_text(movie: dict) -> str:
    """Builds a rich text string for a movie to embed."""
    genres = " ".join(movie.get("genres", []))
    return f"{movie['title']}. {genres}. {movie['overview']}"


def load_or_build_embeddings(movies: list[dict]) -> np.ndarray:
    """Loads movie embeddings from cache or builds them fresh."""
    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading embeddings from cache...")
        return np.load(EMBEDDINGS_PATH)

    print("Loading Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [_build_movie_text(m) for m in movies]
    print(f"Building embeddings for {len(texts)} movies...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    os.makedirs("data", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print("Embeddings saved.")
    return embeddings


def recommend_movies(
    mood_description: str,
    movies: list[dict],
    embeddings: np.ndarray,
    top_n: int = 10,
) -> list[dict]:
    """
    Embeds the personality profile's mood description and recommends
    top_n movies via cosine similarity.
    """
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([mood_description])

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        movie = movies[idx].copy()
        movie["similarity_score"] = float(similarities[idx])
        movie["poster_url"] = (
            f"{TMDB_IMAGE_BASE}{movie['poster_path']}"
            if movie.get("poster_path")
            else None
        )
        recommendations.append(movie)

    return recommendations


def get_movie_poster_url(poster_path: str | None) -> str | None:
    if poster_path:
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    return None
