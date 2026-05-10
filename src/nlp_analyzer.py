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
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline, logging as transformers_logging

transformers_logging.set_verbosity_error()
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


def fetch_lyrics_for_tracks(tracks_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Fetches lyrics for the top N tracks in parallel (up to 5 workers).
    """
    genius = get_genius_client()
    df = tracks_df.head(top_n).copy()
    rows = list(df.itertuples(index=True))
    lyrics_map: dict[int, str | None] = {}

    def _fetch(row):
        print(f"  Lyrics: {row.track_name} - {row.artist_name}")
        lyrics = fetch_lyrics(genius, row.track_name, row.artist_name)
        time.sleep(0.2)  # minimal rate-limit guard per worker
        return row.Index, lyrics

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_fetch, row) for row in rows]
        for future in as_completed(futures):
            idx, lyrics = future.result()
            lyrics_map[idx] = lyrics

    df["lyrics"] = [lyrics_map[i] for i in df.index]
    return df


_emotion_classifier = None


def load_emotion_model():
    """Loads the HuggingFace emotion analysis model (cached after first load)."""
    global _emotion_classifier
    if _emotion_classifier is None:
        print("Loading emotion model (may take a moment on first run)...")
        _emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,  # Return all emotion scores
            device=-1,   # CPU
        )
    return _emotion_classifier


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
    Averages emotion scores across tracks that have analyzable lyrics.
    Tracks without lyrics are skipped (not folded in as neutral).
    If lyrics analysis is uninformative (mostly neutral — common with
    non-English lyrics or instrumentals), blend with an audio-feature
    heuristic so the user doesn't see "100% Neutral".
    """
    classifier = load_emotion_model()
    print("Analyzing lyrics...")

    texts = []
    for _, row in tracks_with_lyrics.iterrows():
        lyrics = row.get("lyrics")
        if lyrics and len(str(lyrics).strip()) >= 20:
            texts.append(str(lyrics)[:512])

    n_with = len(texts)
    n_total = len(tracks_with_lyrics)
    print(f"  {n_with}/{n_total} tracks have analyzable lyrics")

    if not texts:
        print("  No usable lyrics — using audio-feature emotion fallback")
        return _audio_feature_emotion_fallback(tracks_with_lyrics)

    try:
        batch_results = classifier(texts, batch_size=8)
    except Exception as e:
        print(f"  Emotion classifier error: {e} — using audio-feature fallback")
        return _audio_feature_emotion_fallback(tracks_with_lyrics)

    emotion_records = []
    for results in batch_results:
        scores = {r["label"].lower(): r["score"] for r in results}
        emotion_records.append({e: scores.get(e, 0.0) for e in EMOTIONS})

    profile = pd.DataFrame(emotion_records).mean().to_dict()

    # If neutral dominates (likely non-English lyrics), blend with audio fallback
    if profile.get("neutral", 0) > 0.6:
        print(f"  High neutral ({profile['neutral']:.2f}) — blending with audio fallback")
        audio_emotion = _audio_feature_emotion_fallback(tracks_with_lyrics)
        w_audio = min(0.7, profile["neutral"])
        profile = {
            e: (1 - w_audio) * profile.get(e, 0.0) + w_audio * audio_emotion.get(e, 0.0)
            for e in EMOTIONS
        }

    return profile


def _audio_feature_emotion_fallback(tracks_df: pd.DataFrame) -> dict:
    """
    Heuristic emotion vector from average audio features.
    Loosely follows Russell's circumplex (valence × arousal).
    """
    if tracks_df.empty or "energy" not in tracks_df.columns:
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    energy       = float(tracks_df["energy"].mean())
    valence      = float(tracks_df["valence"].mean())
    acousticness = float(tracks_df["acousticness"].mean()) if "acousticness" in tracks_df else 0.3
    danceability = float(tracks_df["danceability"].mean()) if "danceability" in tracks_df else 0.5

    raw = {
        "joy":      valence * 0.6 + danceability * 0.4,
        "sadness":  (1 - valence) * 0.5 + acousticness * 0.5,
        "anger":    energy * (1 - valence) * 1.2,
        "fear":     (1 - valence) * 0.3 + energy * 0.15,
        "surprise": danceability * 0.3 + energy * 0.2,
        "disgust":  0.05,
        "neutral":  0.15,
    }
    total = sum(raw.values()) or 1.0
    return {e: v / total for e, v in raw.items()}


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
