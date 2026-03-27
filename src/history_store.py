"""
History Store — saves and loads past recommendation sessions per user.
Sessions are stored as JSON in data/history/{user_id}.json (gitignored).
"""

import json
import os
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path("data/history")
MAX_SESSIONS = 20


def _history_path(user_id: str) -> Path:
    return HISTORY_DIR / f"{user_id}.json"


def save_session(
    user_id: str,
    user_name: str,
    personality: dict,
    recommendations: list,
    emotion_profile: dict,
    top_genres: list,
) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _history_path(user_id)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_name": user_name,
        "personality": {
            "name": personality["name"],
            "emoji": personality["emoji"],
            "description": personality["description"],
            "confidence": personality["confidence"],
        },
        "movies": [
            {
                "title": m["title"],
                "release_year": m.get("release_year", ""),
                "genres": m.get("genres", [])[:3],
                "vote_average": m.get("vote_average", 0),
                "similarity_score": m.get("similarity_score", 0),
                "poster_url": m.get("poster_url", ""),
                "overview": m.get("overview", ""),
                "explanation": m.get("claude_explanation", ""),
            }
            for m in recommendations
        ],
        "emotion_profile": {
            k: round(v, 3) for k, v in emotion_profile.items() if v > 0.01
        },
        "top_genres": top_genres[:8],
    }

    history: list = []
    if path.exists():
        try:
            with open(path) as f:
                history = json.load(f)
        except Exception:
            history = []

    # Prepend newest, keep cap
    history.insert(0, entry)
    history = history[:MAX_SESSIONS]

    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history(user_id: str) -> list:
    path = _history_path(user_id)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []
