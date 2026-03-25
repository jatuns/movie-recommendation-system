"""
Phase 5: LLM Integration
Generates personalized explanations for each movie recommendation.
Uses the Groq API (free tier, Llama 3 model).
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def explain_recommendation(
    personality_profile: dict,
    emotion_profile: dict,
    movie: dict,
    top_artists: list = None,
) -> str:
    client = _get_client()

    dominant_emotions = sorted(
        emotion_profile.items(), key=lambda x: x[1], reverse=True
    )[:3]
    emotions_str = ", ".join(
        f"{e} ({v:.0%})" for e, v in dominant_emotions if v > 0.05
    )

    artists_str = ""
    if top_artists:
        artists_str = f"\nFavorite artists: {', '.join(top_artists[:5])}"

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
Write in English, warm and engaging tone. Do not start with "This movie" or "Based on"."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def explain_all_recommendations(
    personality_profile: dict,
    emotion_profile: dict,
    movies: list,
    top_artists: list = None,
) -> list:
    results = []
    for i, movie in enumerate(movies):
        print(f"  [{i+1}/{len(movies)}] Generating explanation for {movie['title']}...")
        try:
            explanation = explain_recommendation(
                personality_profile, emotion_profile, movie, top_artists
            )
        except Exception as e:
            explanation = f"Your {personality_profile['name']} personality makes this film a compelling watch — it echoes the emotional depth found in your music."
            print(f"  Error: {e}")

        movie_with_explanation = movie.copy()
        movie_with_explanation["claude_explanation"] = explanation
        results.append(movie_with_explanation)

    return results
