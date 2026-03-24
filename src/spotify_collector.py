"""
Phase 1: Spotify OAuth & Data Collection
Kullanıcının top tracks, top artists ve audio features verilerini çeker.
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SCOPE = (
    "user-top-read "
    "user-read-recently-played "
    "user-read-private"
)


def get_spotify_client(token_info: dict = None) -> spotipy.Spotify:
    """Spotify istemcisi döndürür. Streamlit session token varsa onu kullanır."""
    if token_info:
        return spotipy.Spotify(auth=token_info["access_token"])

    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501/callback"),
        scope=SCOPE,
        cache_path=".spotify_cache",
        show_dialog=True,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def get_auth_url() -> str:
    """Spotify OAuth login URL'ini döndürür."""
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501/callback"),
        scope=SCOPE,
        show_dialog=True,
    )
    return auth_manager.get_authorize_url()


def exchange_code_for_token(code: str) -> dict:
    """Authorization code'u token'a çevirir."""
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501/callback"),
        scope=SCOPE,
    )
    return auth_manager.get_access_token(code, as_dict=True)


def get_top_tracks(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    """Kullanıcının son 4 haftadaki top track'lerini çeker."""
    results = sp.current_user_top_tracks(limit=limit, time_range="medium_term")
    tracks = []
    for item in results["items"]:
        tracks.append({
            "track_id": item["id"],
            "track_name": item["name"],
            "artist_name": item["artists"][0]["name"],
            "artist_id": item["artists"][0]["id"],
            "popularity": item["popularity"],
            "album_name": item["album"]["name"],
        })
    return tracks


def get_top_artists(sp: spotipy.Spotify, limit: int = 20) -> list[dict]:
    """Kullanıcının top artist'lerini çeker."""
    results = sp.current_user_top_artists(limit=limit, time_range="medium_term")
    artists = []
    for item in results["items"]:
        artists.append({
            "artist_id": item["id"],
            "artist_name": item["name"],
            "genres": ", ".join(item["genres"][:3]),
            "popularity": item["popularity"],
        })
    return artists


def get_audio_features(sp: spotipy.Spotify, track_ids: list[str]) -> list[dict]:
    """
    Track ID listesi için Spotify audio features çeker.
    Özellikler: energy, valence, tempo, danceability, acousticness,
                instrumentalness, loudness, speechiness
    """
    features = []
    # Spotify API max 100 track per request
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        results = sp.audio_features(batch)
        for f in results:
            if f is None:
                continue
            features.append({
                "track_id": f["id"],
                "energy": f["energy"],
                "valence": f["valence"],
                "tempo": f["tempo"],
                "danceability": f["danceability"],
                "acousticness": f["acousticness"],
                "instrumentalness": f["instrumentalness"],
                "loudness": f["loudness"],
                "speechiness": f["speechiness"],
            })
    return features


def collect_user_data(sp: spotipy.Spotify) -> dict:
    """
    Tüm kullanıcı verisini toplar ve birleştirilmiş DataFrame döndürür.
    Returns:
        {
            "tracks_df": DataFrame (tracks + audio features birleştirilmiş),
            "artists_df": DataFrame,
            "user_info": dict,
        }
    """
    user_info = sp.current_user()

    print("Top tracks çekiliyor...")
    tracks = get_top_tracks(sp, limit=50)
    tracks_df = pd.DataFrame(tracks)

    print("Audio features çekiliyor...")
    track_ids = tracks_df["track_id"].tolist()
    features = get_audio_features(sp, track_ids)
    features_df = pd.DataFrame(features)

    merged_df = tracks_df.merge(features_df, on="track_id", how="left")

    print("Top artists çekiliyor...")
    artists = get_top_artists(sp, limit=20)
    artists_df = pd.DataFrame(artists)

    return {
        "tracks_df": merged_df,
        "artists_df": artists_df,
        "user_info": {
            "display_name": user_info.get("display_name", "Kullanıcı"),
            "user_id": user_info.get("id"),
            "country": user_info.get("country"),
        },
    }
