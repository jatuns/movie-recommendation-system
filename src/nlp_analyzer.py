"""
Phase 2: NLP Feature Engineering
Performs emotion analysis from lyrics and combines with audio features.
Model: j-hartmann/emotion-english-distilroberta-base
Emotions: joy, sadness, anger, fear, surprise, disgust, neutral
"""

from __future__ import annotations

import os
import time
import lyricsgenius
import pandas as pd
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def get_genius_client() -> lyricsgenius.Genius:
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    genius = lyricsgenius.Genius(token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"])
    genius.verbose = False
    genius.remove_section_headers = True
    return genius


def fetch_lyrics(genius: lyricsgenius.Genius, track_name: str, artist_name: str) -> str | None:
    """Fetches lyrics for a single track from the Genius API."""
    try:
        song = genius.search_song(track_name, artist_name)
        if song and song.lyrics:
            # First 1000 chars are sufficient (model token limit)
            return song.lyrics[:1000]
    except Exception:
        pass
    return None


def fetch_lyrics_for_tracks(tracks_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Fetches lyrics for the top N tracks.
    Short delay between requests to avoid rate limiting.
    """
    genius = get_genius_client()
    df = tracks_df.head(top_n).copy()
    lyrics_list = []

    for _, row in df.iterrows():
        print(f"  Lyrics: {row['track_name']} - {row['artist_name']}")
        lyrics = fetch_lyrics(genius, row["track_name"], row["artist_name"])
        lyrics_list.append(lyrics)
        time.sleep(0.5)  # Rate limit guard

    df["lyrics"] = lyrics_list
    return df


def load_emotion_model():
    """Loads the HuggingFace emotion analysis model."""
    print("Loading emotion model (may take a moment on first run)...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,  # Return all emotion scores
        device=-1,   # CPU
    )


def analyze_emotions(classifier, text: str) -> dict:
    """Returns 7 emotion scores for a text (sum to 1.0)."""
    if not text or len(text.strip()) < 20:
        # Default to neutral for empty lyrics
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    try:
        results = classifier(text[:512])[0]  # Model max 512 tokens
        scores = {r["label"].lower(): r["score"] for r in results}
        # Fill missing emotions with 0
        return {e: scores.get(e, 0.0) for e in EMOTIONS}
    except Exception:
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}


def compute_emotion_profile(tracks_with_lyrics: pd.DataFrame) -> dict:
    """
    Averages emotion scores across all tracks.
    Returns: {"joy": 0.3, "sadness": 0.4, ...} as the user's emotion profile
    """
    classifier = load_emotion_model()
    emotion_records = []

    print("Analyzing lyrics...")
    for _, row in tracks_with_lyrics.iterrows():
        scores = analyze_emotions(classifier, row.get("lyrics"))
        emotion_records.append(scores)

    emotion_df = pd.DataFrame(emotion_records)
    return emotion_df.mean().to_dict()


def build_user_feature_vector(tracks_df: pd.DataFrame, emotion_profile: dict) -> np.ndarray:
    """
    Combines averaged audio features with emotion scores into a single feature vector.

    Feature order:
    [energy, valence, tempo_norm, danceability, acousticness,
     instrumentalness, speechiness, joy, sadness, anger, fear, surprise, disgust]
    """
    audio_cols = ["energy", "valence", "danceability", "acousticness",
                  "instrumentalness", "speechiness"]

    audio_means = tracks_df[audio_cols].mean().to_dict()

    # Normalize tempo to 0-1 (typical range: 60-200 BPM)
    tempo_norm = (tracks_df["tempo"].mean() - 60) / 140
    tempo_norm = float(np.clip(tempo_norm, 0, 1))

    vector = [
        audio_means["energy"],
        audio_means["valence"],
        tempo_norm,
        audio_means["danceability"],
        audio_means["acousticness"],
        audio_means["instrumentalness"],
        audio_means["speechiness"],
        emotion_profile.get("joy", 0),
        emotion_profile.get("sadness", 0),
        emotion_profile.get("anger", 0),
        emotion_profile.get("fear", 0),
        emotion_profile.get("surprise", 0),
        emotion_profile.get("disgust", 0),
    ]

    return np.array(vector, dtype=float)


FEATURE_NAMES = [
    "energy", "valence", "tempo", "danceability", "acousticness",
    "instrumentalness", "speechiness",
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
]
