"""
Phase 1: Spotify OAuth & Data Collection
Fetches the user's top tracks and top artists.
Note: the audio-features endpoint was restricted by Spotify in 2024.
Audio features are estimated from artist genres instead.
"""

from __future__ import annotations

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SCOPE = (
    "user-top-read "
    "user-read-recently-played "
    "user-read-private"
)

# Genre → audio feature estimation (replaces audio-features API)
GENRE_FEATURE_MAP = {
    "pop":              {"energy": 0.7, "valence": 0.7, "danceability": 0.7, "acousticness": 0.2, "tempo": 120},
    "rock":             {"energy": 0.8, "valence": 0.5, "danceability": 0.5, "acousticness": 0.1, "tempo": 130},
    "hip hop":          {"energy": 0.7, "valence": 0.6, "danceability": 0.8, "acousticness": 0.1, "tempo": 95},
    "rap":              {"energy": 0.7, "valence": 0.5, "danceability": 0.8, "acousticness": 0.1, "tempo": 95},
    "r&b":              {"energy": 0.6, "valence": 0.6, "danceability": 0.75, "acousticness": 0.3, "tempo": 95},
    "soul":             {"energy": 0.5, "valence": 0.6, "danceability": 0.65, "acousticness": 0.4, "tempo": 90},
    "jazz":             {"energy": 0.4, "valence": 0.6, "danceability": 0.5, "acousticness": 0.7, "tempo": 120},
    "classical":        {"energy": 0.3, "valence": 0.5, "danceability": 0.2, "acousticness": 0.9, "tempo": 100},
    "electronic":       {"energy": 0.8, "valence": 0.6, "danceability": 0.8, "acousticness": 0.05, "tempo": 128},
    "dance":            {"energy": 0.85, "valence": 0.7, "danceability": 0.85, "acousticness": 0.05, "tempo": 128},
    "edm":              {"energy": 0.9, "valence": 0.65, "danceability": 0.85, "acousticness": 0.02, "tempo": 130},
    "indie":            {"energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.4, "tempo": 110},
    "alternative":      {"energy": 0.6, "valence": 0.4, "danceability": 0.45, "acousticness": 0.3, "tempo": 120},
    "metal":            {"energy": 0.95, "valence": 0.3, "danceability": 0.4, "acousticness": 0.05, "tempo": 150},
    "punk":             {"energy": 0.9, "valence": 0.4, "danceability": 0.5, "acousticness": 0.05, "tempo": 160},
    "folk":             {"energy": 0.3, "valence": 0.6, "danceability": 0.35, "acousticness": 0.85, "tempo": 90},
    "country":          {"energy": 0.6, "valence": 0.7, "danceability": 0.6, "acousticness": 0.5, "tempo": 110},
    "blues":            {"energy": 0.4, "valence": 0.4, "danceability": 0.45, "acousticness": 0.6, "tempo": 85},
    "reggae":           {"energy": 0.5, "valence": 0.8, "danceability": 0.7, "acousticness": 0.3, "tempo": 80},
    "latin":            {"energy": 0.75, "valence": 0.8, "danceability": 0.85, "acousticness": 0.2, "tempo": 100},
    "k-pop":            {"energy": 0.8, "valence": 0.75, "danceability": 0.8, "acousticness": 0.1, "tempo": 120},
    "ambient":          {"energy": 0.2, "valence": 0.4, "danceability": 0.2, "acousticness": 0.7, "tempo": 80},
    "trap":             {"energy": 0.7, "valence": 0.4, "danceability": 0.75, "acousticness": 0.05, "tempo": 140},
    "acoustic":         {"energy": 0.3, "valence": 0.6, "danceability": 0.35, "acousticness": 0.9, "tempo": 90},
    "singer-songwriter":{"energy": 0.35, "valence": 0.5, "danceability": 0.35, "acousticness": 0.75, "tempo": 95},
}

DEFAULT_FEATURES = {"energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.3, "tempo": 110}


def _estimate_features_from_genres(genres: list[str]) -> dict:
    """Estimates audio features from an artist's genre list."""
    matched = []
    for genre in genres:
        genre_lower = genre.lower()
        for key, features in GENRE_FEATURE_MAP.items():
            if key in genre_lower:
                matched.append(features)
                break

    if not matched:
        return DEFAULT_FEATURES.copy()

    return {
        col: float(np.mean([f[col] for f in matched]))
        for col in DEFAULT_FEATURES
    }


def get_spotify_client(token_info: dict = None) -> spotipy.Spotify:
    if token_info:
        return spotipy.Spotify(auth=token_info["access_token"])

    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501/callback"),
        scope=SCOPE,
        cache_path=".spotify_cache",
        show_dialog=True,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def get_auth_url() -> str:
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501/callback"),
        scope=SCOPE,
        show_dialog=True,
    )
    return auth_manager.get_authorize_url()


def exchange_code_for_token(code: str) -> dict:
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501/callback"),
        scope=SCOPE,
    )
    return auth_manager.get_access_token(code, as_dict=True)


def get_top_tracks(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    results = sp.current_user_top_tracks(limit=limit, time_range="medium_term")
    tracks = []
    for item in results.get("items", []):
        album   = item.get("album", {})
        images  = album.get("images", [])
        album_image = images[-1]["url"] if images else None  # smallest image for thumbnails
        tracks.append({
            "track_id":    item.get("id"),
            "track_name":  item.get("name"),
            "artist_name": item["artists"][0]["name"] if item.get("artists") else "Unknown",
            "artist_id":   item["artists"][0]["id"] if item.get("artists") else None,
            "popularity":  item.get("popularity", 0),
            "album_name":  album.get("name", ""),
            "album_image": album_image,
        })
    return tracks


def get_top_artists(sp: spotipy.Spotify, limit: int = 20) -> list[dict]:
    results = sp.current_user_top_artists(limit=limit, time_range="medium_term")
    artists = []
    for item in results.get("items", []):
        images = item.get("images", [])
        # Prefer smallest image (last) for thumbnails, fallback to largest
        image_url = images[-1]["url"] if images else None
        artists.append({
            "artist_id": item.get("id"),
            "artist_name": item.get("name"),
            "genres": item.get("genres", []),
            "genres_str": ", ".join(item.get("genres", [])[:3]),
            "popularity": item.get("popularity", 0),
            "image_url": image_url,
        })
    return artists


def collect_user_data(sp: spotipy.Spotify) -> dict:
    """
    Fetches top tracks + top artists.
    Audio features are estimated from genres since the API is restricted.
    """
    user_info = sp.current_user()

    print("Fetching top tracks...")
    tracks = get_top_tracks(sp, limit=50)
    tracks_df = pd.DataFrame(tracks)

    print("Fetching top artists...")
    artists = get_top_artists(sp, limit=20)
    artists_df = pd.DataFrame(artists)

    # If top-artists returned empty genres (common Spotify API limitation),
    # enrich using sp.artists() batch call with unique artist IDs from tracks
    all_genres = []
    for row in artists:
        all_genres.extend(row.get("genres", []))

    if not all_genres and "artist_id" in tracks_df.columns:
        print("Top-artist genres empty — enriching from track artist IDs...")
        unique_ids = [aid for aid in tracks_df["artist_id"].dropna().unique() if aid][:50]
        # Spotify batch limit is 50 per call
        for i in range(0, len(unique_ids), 50):
            batch = sp.artists(unique_ids[i:i+50]).get("artists", [])
            for a in batch:
                if not a:
                    continue
                genres = a.get("genres", [])
                all_genres.extend(genres)
                # Patch genres back into artists list if artist matches
                for artist_row in artists:
                    if artist_row.get("artist_id") == a.get("id") and not artist_row.get("genres"):
                        artist_row["genres"] = genres
                        artist_row["genres_str"] = ", ".join(genres[:3])

        # Rebuild artists_df with enriched genres
        artists_df = pd.DataFrame(artists)

    # Collect all genres and estimate average features
    print("Estimating audio features from genres...")

    estimated_features = _estimate_features_from_genres(all_genres)

    # Apply estimated features to all tracks
    for col, val in estimated_features.items():
        tracks_df[col] = val
    tracks_df["instrumentalness"] = 0.1
    tracks_df["speechiness"] = 0.1
    tracks_df["loudness"] = -8.0

    return {
        "tracks_df": tracks_df,
        "artists_df": artists_df,
        "user_info": {
            "display_name": user_info.get("display_name", "User"),
            "user_id": user_info.get("id"),
            "country": user_info.get("country"),
        },
        "all_genres": all_genres,
    }
