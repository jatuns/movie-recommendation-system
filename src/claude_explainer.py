"""
Phase 5: Claude API Integration
Her film önerisi için kişiselleştirilmiş açıklama üretir.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def explain_recommendation(
    personality_profile: dict,
    emotion_profile: dict,
    movie: dict,
    top_artists: list[str] = None,
) -> str:
    """
    Kullanıcının kişilik profili + film bilgisi verildiğinde
    Claude'un 2-3 cümlelik kişiselleştirilmiş açıklama üretmesini sağlar.

    Args:
        personality_profile: assign_personality() çıktısı
        emotion_profile: {"joy": 0.3, "sadness": 0.5, ...}
        movie: {"title": ..., "overview": ..., "genres": [...]}
        top_artists: Kullanıcının top artist'leri (opsiyonel, daha kişisel açıklama için)
    """
    client = _get_client()

    # Dominant duyguları bul
    dominant_emotions = sorted(
        emotion_profile.items(), key=lambda x: x[1], reverse=True
    )[:3]
    emotions_str = ", ".join(
        f"{e} ({v:.0%})" for e, v in dominant_emotions if v > 0.05
    )

    artists_str = ""
    if top_artists:
        artists_str = f"\nFavori sanatçılar: {', '.join(top_artists[:5])}"

    genres_str = ", ".join(movie.get("genres", [])[:3]) or "Drama"

    prompt = f"""You are a movie recommendation expert who writes personalized, insightful explanations.

User's music personality profile:
- Profile: {personality_profile['emoji']} {personality_profile['name']}
- Description: {personality_profile['description']}
- Dominant emotions in their music: {emotions_str}{artists_str}

Movie to explain:
- Title: {movie['title']} ({movie.get('release_year', '')})
- Genres: {genres_str}
- Synopsis: {movie['overview'][:300]}

Write a personalized 2-3 sentence explanation of WHY this user will love this movie, based on their music personality.
Be specific about emotional connections between their music taste and the film's themes.
Write in English, in a warm and engaging tone. Do not start with "This movie" or "Based on".
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()


def explain_all_recommendations(
    personality_profile: dict,
    emotion_profile: dict,
    movies: list[dict],
    top_artists: list[str] = None,
) -> list[dict]:
    """
    Öneri listesindeki her film için Claude açıklaması ekler.
    Returns: Her filme "claude_explanation" alanı eklenmiş liste
    """
    results = []
    for i, movie in enumerate(movies):
        print(f"  [{i+1}/{len(movies)}] {movie['title']} için açıklama üretiliyor...")
        try:
            explanation = explain_recommendation(
                personality_profile, emotion_profile, movie, top_artists
            )
        except Exception as e:
            explanation = f"Bu film, {personality_profile['name']} profilinizle harika bir uyum sağlıyor."
            print(f"  Hata: {e}")

        movie_with_explanation = movie.copy()
        movie_with_explanation["claude_explanation"] = explanation
        results.append(movie_with_explanation)

    return results
