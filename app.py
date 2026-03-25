"""
Phase 6: Streamlit App
Music Personality → Movie Recommendation System main application.
"""

import os
import sys
import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spotify_collector import get_auth_url, exchange_code_for_token, get_spotify_client, collect_user_data
from nlp_analyzer import fetch_lyrics_for_tracks, compute_emotion_profile, build_user_feature_vector, EMOTIONS
from personality_clustering import assign_personality, get_pca_coordinates, PERSONALITY_PROFILES
from movie_recommender import fetch_movies, load_or_build_embeddings, recommend_movies
from claude_explainer import explain_all_recommendations

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music → Movie Match",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1DB954, #191414);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .profile-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #333;
        text-align: center;
    }
    .movie-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #2a2a2a;
        height: 100%;
    }
    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.3rem;
    }
    .movie-genres {
        font-size: 0.75rem;
        color: #1DB954;
        margin-bottom: 0.5rem;
    }
    .movie-explanation {
        font-size: 0.85rem;
        color: #ccc;
        font-style: italic;
        line-height: 1.5;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
        background: #2a2a2a;
        color: #ddd;
    }
    .badge-green { background: rgba(29,185,84,0.15); color: #1DB954; border: 1px solid #1DB954; }
    .badge-yellow { background: rgba(255,193,7,0.15); color: #ffc107; border: 1px solid #ffc107; }
    .spotify-btn {
        background-color: #1DB954;
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 700;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s;
        letter-spacing: 0.02em;
    }
    .spotify-btn:hover { background-color: #1ed760; color: white; }
    .step-box {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        border-left: 3px solid #1DB954;
    }
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #222;
    }
    .stProgress .st-bo { background-color: #1DB954; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ─────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "token_info": None,
        "user_data": None,
        "emotion_profile": None,
        "feature_vector": None,
        "personality": None,
        "recommendations": None,
        "analysis_done": False,
        "page": "home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── OAuth Callback ────────────────────────────────────────────────────────────
def handle_oauth_callback():
    """Captures the code param from the URL and exchanges it for a token."""
    try:
        query_params = st.query_params
        code = query_params.get("code")
        if code and not st.session_state.token_info:
            with st.spinner("Connecting to Spotify..."):
                token_info = exchange_code_for_token(code)
                st.session_state.token_info = token_info
            st.query_params.clear()
            st.session_state.page = "analyze"
            st.rerun()
    except Exception as e:
        st.error(f"OAuth error: {e}")


# ─── Page: Home ─────────────────────────────────────────────────────────────────
def page_home():
    st.markdown('<div class="main-header">🎵 Music → Movie Match</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Discover movies that match your music personality — powered by AI.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### How it works")

        steps = [
            ("🎧", "Connect Spotify", "Your top tracks & artists are fetched automatically"),
            ("📝", "Lyrics Analysis", "Genius API pulls lyrics for emotion detection"),
            ("🧠", "Personality Profiling", "K-Means clustering assigns your music personality"),
            ("🎬", "Movie Matching", "Sentence Transformers find your best 10 movies from 5,000+"),
            ("✨", "AI Explanations", "Groq AI writes a personalized reason for each pick"),
        ]
        for icon, title, desc in steps:
            st.markdown(
                f'<div class="step-box"><b>{icon} {title}</b> — <span style="color:#aaa">{desc}</span></div>',
                unsafe_allow_html=True,
            )

        auth_url = get_auth_url()
        st.markdown(f"""
        <div style="text-align: center; margin-top: 2.5rem;">
            <a href="{auth_url}" class="spotify-btn" target="_self">
                🎵 &nbsp;Connect with Spotify
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.caption("🔒 Read-only access to your listening data. Nothing is stored or shared.")

    st.markdown(
        '<div class="footer">Built with Streamlit · Spotify API · HuggingFace · Groq (Llama 3) · TMDB</div>',
        unsafe_allow_html=True,
    )


# ─── Page: Analyze ──────────────────────────────────────────────────────────────
def page_analyze():
    st.markdown('<div class="main-header">🔬 Analyzing Your Music...</div>', unsafe_allow_html=True)

    if st.session_state.analysis_done:
        st.session_state.page = "results"
        st.rerun()
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Spotify data (cached)
        if not st.session_state.get("user_data"):
            status_text.text("🎵 Fetching your top tracks from Spotify...")
            sp = get_spotify_client(st.session_state.token_info)
            user_data = collect_user_data(sp)
            st.session_state.user_data = user_data
        else:
            user_data = st.session_state.user_data
        progress_bar.progress(20)

        # Step 2 & 3: Lyrics + NLP (cached)
        if not st.session_state.get("emotion_profile"):
            status_text.text("📝 Fetching lyrics via Genius API...")
            tracks_df = user_data["tracks_df"]
            tracks_with_lyrics = fetch_lyrics_for_tracks(tracks_df, top_n=20)
            progress_bar.progress(40)

            status_text.text("🧠 Running emotion analysis (HuggingFace)...")
            emotion_profile = compute_emotion_profile(tracks_with_lyrics)
            st.session_state.emotion_profile = emotion_profile
            feature_vector = build_user_feature_vector(tracks_df, emotion_profile)
            st.session_state.feature_vector = feature_vector
        else:
            emotion_profile = st.session_state.emotion_profile
            feature_vector = st.session_state.feature_vector
        progress_bar.progress(60)

        # Step 4: Personality profile (cached)
        if not st.session_state.get("personality"):
            status_text.text("🎭 Determining your music personality...")
            personality = assign_personality(feature_vector)
            st.session_state.personality = personality
        else:
            personality = st.session_state.personality
        progress_bar.progress(70)

        # Step 5: Movie recommendations (cached)
        if not st.session_state.get("recommendations"):
            status_text.text("🎬 Loading and analyzing 5,000+ movies...")
            movies = fetch_movies(total=5000)
            embeddings = load_or_build_embeddings(movies)
            raw_recs = recommend_movies(personality["mood_description"], movies, embeddings, top_n=10)
            progress_bar.progress(85)

            # Step 6: AI explanations
            status_text.text("✨ Writing personalized explanations with Groq AI...")
            top_artists = user_data["artists_df"]["artist_name"].tolist()[:5]
            recommendations = explain_all_recommendations(
                personality, emotion_profile, raw_recs, top_artists
            )
            st.session_state.recommendations = recommendations
        progress_bar.progress(100)

        st.session_state.analysis_done = True
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        st.session_state.page = "results"
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.info("Please make sure all API keys are correctly configured in your .env file.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.session_state.token_info = None
            st.rerun()


# ─── Page: Results ──────────────────────────────────────────────────────────────
def page_results():
    personality = st.session_state.personality
    emotion_profile = st.session_state.emotion_profile
    user_data = st.session_state.user_data
    recommendations = st.session_state.recommendations

    if not personality:
        st.session_state.page = "home"
        st.rerun()
        return

    # ── Header ──
    user_name = user_data["user_info"]["display_name"]
    st.markdown(f"## Hello, {user_name}! 👋")

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["🎭 Your Personality", "🎬 Movie Picks", "📊 Personality Map"])

    # ── Tab 1: Personality Profile ────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class="profile-card">
                <div style="font-size: 4rem;">{personality['emoji']}</div>
                <h2 style="color: white; margin: 0.5rem 0;">{personality['name']}</h2>
                <p style="color: #aaa; font-size: 1rem;">{personality['description']}</p>
                <p style="color: #1DB954; font-size: 0.9rem; margin-top: 1rem;">
                    Match confidence: <strong>{personality['confidence']:.0%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("#### 🎵 Your Top Tracks")
            tracks_display = user_data["tracks_df"][["track_name", "artist_name"]].head(10)
            for _, row in tracks_display.iterrows():
                st.markdown(f"- **{row['track_name']}** — {row['artist_name']}")

        with col2:
            # Radar Chart: Audio Features
            audio_cols = ["energy", "valence", "danceability", "acousticness"]
            audio_values = [
                float(user_data["tracks_df"][c].mean()) for c in audio_cols
            ]

            fig_audio = go.Figure(data=go.Scatterpolar(
                r=audio_values + [audio_values[0]],
                theta=["Energy", "Valence", "Danceability", "Acousticness", "Energy"],
                fill="toself",
                fillcolor="rgba(29, 185, 84, 0.25)",
                line=dict(color="#1DB954", width=2),
            ))
            fig_audio.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Audio Profile",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=300,
            )
            st.plotly_chart(fig_audio, use_container_width=True)

            # Emotion Bar Chart
            emotion_data = {k: v for k, v in emotion_profile.items() if k != "neutral" and v > 0.02}
            if emotion_data:
                fig_emotion = px.bar(
                    x=list(emotion_data.keys()),
                    y=list(emotion_data.values()),
                    title="Emotion Profile",
                    color=list(emotion_data.values()),
                    color_continuous_scale="Viridis",
                )
                fig_emotion.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=300,
                    xaxis_title="",
                    yaxis_title="Score",
                )
                st.plotly_chart(fig_emotion, use_container_width=True)

    # ── Tab 2: Movie Picks ────────────────────────────────────────────────────
    with tab2:
        st.markdown(f"### {personality['emoji']} Movie Picks for **{personality['name']}**")
        st.markdown("*Personalized explanations powered by Groq AI (Llama 3):*")
        st.markdown("---")

        for i, movie in enumerate(recommendations):
            col1, col2 = st.columns([1, 3])

            with col1:
                if movie.get("poster_url"):
                    st.image(movie["poster_url"], width=150)
                else:
                    st.markdown('<div style="font-size:4rem;text-align:center">🎬</div>', unsafe_allow_html=True)

            with col2:
                genres = ", ".join(movie.get("genres", [])[:3])
                year = movie.get("release_year", "")
                score = movie.get("vote_average", 0)
                similarity = movie.get("similarity_score", 0)

                st.markdown(f"#### {i+1}. {movie['title']} ({year})")
                st.markdown(
                    f'<span class="badge badge-yellow">⭐ {score:.1f}</span>'
                    f'<span class="badge badge-green">🎯 {similarity:.0%} match</span>'
                    f'<span class="badge">{genres}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown(f"> {movie.get('claude_explanation', '')}")

            st.markdown("---")

    # ── Tab 3: Personality Map ────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 Music Personality Map")
        st.markdown("6 personality clusters in 2D PCA space — your position shown as a star:")

        feature_vector = st.session_state.feature_vector
        pca_data = get_pca_coordinates(feature_vector)

        fig = go.Figure()

        # Cluster centroids
        for c in pca_data["centroid_coords"]:
            fig.add_trace(go.Scatter(
                x=[c["x"]], y=[c["y"]],
                mode="markers+text",
                marker=dict(size=22, color=c["color"], opacity=0.75, line=dict(width=1, color="#fff")),
                text=[f"{c['emoji']} {c['name']}"],
                textposition="top center",
                name=c["name"],
            ))

        # User position
        fig.add_trace(go.Scatter(
            x=[pca_data["user_x"]], y=[pca_data["user_y"]],
            mode="markers+text",
            marker=dict(size=18, color="#1DB954", symbol="star", line=dict(width=1, color="#fff")),
            text=["⭐ You"],
            textposition="top center",
            name="You",
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,30,1)",
            font=dict(color="white"),
            height=500,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Match scores for all profiles
        st.markdown("#### Your Match Scores")
        similarities = personality.get("all_similarities", {})
        for name, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            emoji = next(
                (p["emoji"] for p in PERSONALITY_PROFILES.values() if p["name"] == name), "🎵"
            )
            cols = st.columns([3, 7])
            with cols[0]:
                st.markdown(f"{emoji} **{name}**")
            with cols[1]:
                st.progress(score)

    # Re-analyze button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analyze Again", use_container_width=True):
            for key in ["analysis_done", "user_data", "emotion_profile", "feature_vector",
                        "personality", "recommendations", "token_info"]:
                st.session_state[key] = None
            st.session_state.analysis_done = False
            st.session_state.page = "home"
            st.rerun()

    st.markdown(
        '<div class="footer">Built with Streamlit · Spotify API · HuggingFace · Groq (Llama 3) · TMDB</div>',
        unsafe_allow_html=True,
    )


# ─── Main Flow ──────────────────────────────────────────────────────────────────
def main():
    init_session()
    handle_oauth_callback()

    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "analyze":
        page_analyze()
    elif page == "results":
        page_results()
    else:
        page_home()


if __name__ == "__main__":
    main()
