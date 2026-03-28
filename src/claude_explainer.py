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


def build_recommendation_prompt(personality, emotions, top_artists,
                                top_genres, movie_title, movie_year,
                                movie_genre, movie_description):
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    emotions_str = ", ".join([f"{e} ({round(v*100)}%)" for e, v in top_emotions])
    artists_str = ", ".join(top_artists[:3])

    return f"""You are a sharp, culturally fluent film critic who connects \
music psychology to cinema.

User's music profile:
- Personality: {personality}
- Top emotions detected in their music: {emotions_str}
- Favorite artists: {artists_str}

Film: {movie_title} ({movie_year}) — {movie_genre}
Synopsis: {movie_description[:300]}

Write exactly 2-3 sentences explaining why this film matches their musical soul.

Rules:
- Mention at least one of their actual artists by name
- Connect a specific film element to their music personality
- Sound like a knowledgeable friend, not a recommendation engine
- Never start with "This film", "Based on", or "As someone who"
- Never use: resonate, journey, tapestry, sonic, captivating, compelling
- Second person only ("your taste", "you'll find")
- Every sentence must earn its place — no filler"""


def explain_recommendation(
    personality_profile: dict,
    emotion_profile: dict,
    movie: dict,
    top_artists: list = None,
    top_genres: list = None,
) -> str:
    client = _get_client()

    personality = f"{personality_profile.get('emoji', '')} {personality_profile.get('name', '')}"
    genres_str  = ", ".join(movie.get("genres", [])[:3]) or "Drama"

    prompt = build_recommendation_prompt(
        personality       = personality,
        emotions          = emotion_profile,
        top_artists       = top_artists or [],
        top_genres        = top_genres or [],
        movie_title       = movie.get("title", ""),
        movie_year        = movie.get("release_year", ""),
        movie_genre       = genres_str,
        movie_description = movie.get("overview", ""),
    )

    response = _get_client().chat.completions.create(
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
    top_genres: list = None,
) -> list:
    results = []
    for i, movie in enumerate(movies):
        print(f"  [{i+1}/{len(movies)}] Generating explanation for {movie['title']}...")
        try:
            explanation = explain_recommendation(
                personality_profile, emotion_profile, movie, top_artists, top_genres
            )
        except Exception as e:
            explanation = f"Your {personality_profile['name']} personality makes this film a compelling watch — it echoes the emotional depth found in your music."
            print(f"  Error: {e}")

        movie_with_explanation = movie.copy()
        movie_with_explanation["claude_explanation"] = explanation
        results.append(movie_with_explanation)

    return results
