"""
Music Personality → Movie Recommendation System
Full multi-page Streamlit application.
Pages: home · analyze · dashboard · movie_detail · profile · history
"""

import os
import sys
import time
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spotify_collector import get_auth_url, exchange_code_for_token, get_spotify_client, collect_user_data
from nlp_analyzer import fetch_lyrics_for_tracks, compute_emotion_profile, build_user_feature_vector, EMOTIONS
from personality_clustering import assign_personality, get_pca_coordinates, PERSONALITY_PROFILES
from movie_recommender import fetch_movies, load_or_build_embeddings, recommend_movies
from claude_explainer import explain_all_recommendations
from history_store import save_session, load_history

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music → Movie Match",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Base ─── */
    [data-testid="stAppViewContainer"] { background: #0a0a0a; }
    [data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #1a1a1a; }
    [data-testid="stSidebar"] * { color: #ddd !important; }

    /* ─── Typography ─── */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(120deg, #1DB954 0%, #fff 60%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }
    .hero-sub {
        color: #777;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* ─── Buttons ─── */
    .btn-spotify {
        background: #1DB954;
        color: #000 !important;
        padding: 0.85rem 2.2rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 800;
        text-decoration: none !important;
        display: inline-block;
        letter-spacing: 0.02em;
        transition: background 0.2s, transform 0.1s;
    }
    .btn-spotify:hover { background: #1ed760; transform: scale(1.03); }

    /* ─── Cards ─── */
    .card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 14px;
        padding: 1.5rem;
    }
    .card-hover {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 14px;
        padding: 1rem;
        transition: border-color 0.2s;
    }
    .card-hover:hover { border-color: #1DB954; }

    /* ─── Profile card ─── */
    .profile-card {
        background: linear-gradient(135deg, #0f1f15 0%, #111 100%);
        border: 1px solid #1DB954;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    /* ─── Badges ─── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0.15rem;
        background: #1e1e1e;
        color: #bbb;
    }
    .badge-green { background: rgba(29,185,84,0.12); color: #1DB954; border: 1px solid rgba(29,185,84,0.3); }
    .badge-yellow { background: rgba(255,193,7,0.12); color: #ffc107; border: 1px solid rgba(255,193,7,0.3); }
    .badge-blue { background: rgba(99,179,237,0.12); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }

    /* ─── Feature row (landing) ─── */
    .feature-row {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid #1a1a1a;
    }
    .feature-icon {
        font-size: 1.5rem;
        flex-shrink: 0;
        width: 2rem;
        text-align: center;
    }
    .feature-label { font-weight: 700; color: #fff; font-size: 0.95rem; }
    .feature-desc { color: #666; font-size: 0.85rem; margin-top: 0.1rem; }

    /* ─── Sidebar nav buttons ─── */
    .nav-btn {
        display: block;
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        color: #bbb;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: background 0.15s;
        margin-bottom: 0.2rem;
    }
    .nav-btn:hover { background: #1a1a1a; color: #fff; }
    .nav-btn-active { background: rgba(29,185,84,0.1) !important; color: #1DB954 !important; border-left: 3px solid #1DB954; }

    /* ─── Movie grid card ─── */
    .movie-grid-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 12px;
        overflow: hidden;
        transition: border-color 0.2s, transform 0.15s;
        height: 100%;
    }
    .movie-grid-card:hover { border-color: #1DB954; transform: translateY(-2px); }

    /* ─── History card ─── */
    .history-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .history-card:hover { border-color: #333; }

    /* ─── Artist chip ─── */
    .artist-chip {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.82rem;
        color: #ccc;
        margin: 0.2rem;
    }

    /* ─── Progress override ─── */
    .stProgress > div > div > div > div { background: #1DB954 !important; }

    /* ─── Divider ─── */
    .divider { border: none; border-top: 1px solid #1a1a1a; margin: 1.5rem 0; }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        color: #333;
        font-size: 0.78rem;
        margin-top: 4rem;
        padding: 1.5rem 0;
        border-top: 1px solid #141414;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "page": "home",
        "token_info": None,
        "user_data": None,
        "emotion_profile": None,
        "feature_vector": None,
        "personality": None,
        "recommendations": None,
        "analysis_done": False,
        "selected_movie": None,   # movie dict for detail page
        "prev_page": None,         # back navigation
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_to(page: str):
    st.session_state.prev_page = st.session_state.page
    st.session_state.page = page
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  OAUTH
# ═══════════════════════════════════════════════════════════════════════════════

def handle_oauth_callback():
    try:
        code = st.query_params.get("code")
        if code and not st.session_state.token_info:
            with st.spinner("Connecting to Spotify..."):
                token_info = exchange_code_for_token(code)
                st.session_state.token_info = token_info
            st.query_params.clear()
            go_to("analyze")
    except Exception as e:
        st.error(f"OAuth error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — shown after login
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    if not st.session_state.token_info:
        return

    current = st.session_state.page
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem; text-align:center;">
            <span style="font-size:1.5rem;font-weight:900;color:#1DB954;">🎵</span>
            <span style="font-size:1rem;font-weight:700;color:#fff;margin-left:0.4rem;">Music Match</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.user_data:
            name = st.session_state.user_data["user_info"]["display_name"]
            st.markdown(f"<div style='text-align:center;color:#666;font-size:0.82rem;padding-bottom:1rem;'>Logged in as <b style='color:#ccc'>{name}</b></div>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1e1e1e;margin:0 0 0.75rem'>", unsafe_allow_html=True)

        nav_items = [
            ("dashboard",    "🏠", "Dashboard"),
            ("profile",      "👤", "Profile"),
            ("history",      "📚", "History"),
        ]
        for page_key, icon, label in nav_items:
            active = "nav-btn-active" if current == page_key else ""
            if st.button(f"{icon}  {label}", key=f"nav_{page_key}", use_container_width=True,
                         help=label):
                go_to(page_key)

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1e1e1e;margin:1rem 0 0.5rem'>", unsafe_allow_html=True)
        if st.button("🚪  Disconnect", use_container_width=True, key="nav_disconnect"):
            for k in ["token_info", "user_data", "emotion_profile", "feature_vector",
                      "personality", "recommendations", "analysis_done", "selected_movie"]:
                st.session_state[k] = None
            st.session_state.analysis_done = False
            go_to("home")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LANDING
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    col_l, col_main, col_r = st.columns([1, 2.2, 1])
    with col_main:
        st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Your music,<br>your movies.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">Connect Spotify. Get 10 movies built for your taste — analyzed by AI.</div>',
            unsafe_allow_html=True,
        )

        auth_url = get_auth_url()
        st.markdown(f"""
        <a href="{auth_url}" class="btn-spotify" target="_self">
            🎵 &nbsp; Connect with Spotify
        </a>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)

        features = [
            ("🎧", "Connect Spotify", "Read-only access to your top tracks & artists"),
            ("📝", "Lyrics Analysis", "Genius API fetches lyrics for emotion detection"),
            ("🧠", "Personality Profiling", "K-Means clusters you into one of 6 music personalities"),
            ("🎬", "Movie Matching", "Sentence Transformers match your vibe to 5,000+ films"),
            ("✨", "AI Explanations", "Groq (Llama 3) writes a personal reason for every pick"),
        ]
        for icon, label, desc in features:
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-icon">{icon}</div>
                <div>
                    <div class="feature-label">{label}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div style="color:#333;font-size:0.8rem;text-align:center;">🔒 Read-only · Nothing stored on our servers · Open source</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built with Streamlit · Spotify · HuggingFace · Groq · TMDB</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYZE (loading screen)
# ═══════════════════════════════════════════════════════════════════════════════

def page_analyze():
    if st.session_state.analysis_done:
        go_to("dashboard")
        return

    st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)

    col_l, col_main, col_r = st.columns([1, 2, 1])
    with col_main:
        st.markdown("""
        <div style="text-align:center;margin-bottom:2rem;">
            <div style="font-size:2.5rem;font-weight:900;color:#fff;">Analyzing your music</div>
            <div style="color:#555;margin-top:0.5rem;">This takes about 30–60 seconds</div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status = st.empty()

        steps_display = st.empty()

        completed_steps = []

        def update(pct: int, text: str, done_step: str = ""):
            progress_bar.progress(pct)
            status.markdown(f"<div style='text-align:center;color:#1DB954;font-size:0.95rem;margin:0.5rem 0'>{text}</div>", unsafe_allow_html=True)
            if done_step:
                completed_steps.append(done_step)
            html = "".join(
                f"<div style='padding:0.3rem 0;color:#555;font-size:0.85rem;'>✅ {s}</div>"
                for s in completed_steps
            )
            steps_display.markdown(f"<div style='text-align:center;margin-top:1rem'>{html}</div>", unsafe_allow_html=True)

        try:
            # Step 1
            if not st.session_state.user_data:
                update(5, "🎵 Fetching your Spotify data...")
                sp = get_spotify_client(st.session_state.token_info)
                user_data = collect_user_data(sp)
                st.session_state.user_data = user_data
            else:
                user_data = st.session_state.user_data
            update(20, "Spotify data ready", "Fetched top tracks & artists")

            # Step 2 & 3
            if not st.session_state.emotion_profile:
                update(25, "📝 Fetching lyrics via Genius API...")
                tracks_with_lyrics = fetch_lyrics_for_tracks(user_data["tracks_df"], top_n=20)
                update(45, "🧠 Running emotion analysis...")
                emotion_profile = compute_emotion_profile(tracks_with_lyrics)
                st.session_state.emotion_profile = emotion_profile
                feature_vector = build_user_feature_vector(user_data["tracks_df"], emotion_profile)
                st.session_state.feature_vector = feature_vector
            else:
                emotion_profile = st.session_state.emotion_profile
                feature_vector = st.session_state.feature_vector
            update(60, "Lyrics & emotions analyzed", "Emotion profile built")

            # Step 4
            if not st.session_state.personality:
                update(62, "🎭 Determining your music personality...")
                personality = assign_personality(feature_vector)
                st.session_state.personality = personality
            else:
                personality = st.session_state.personality
            update(70, "Personality assigned", f"You are: {personality['emoji']} {personality['name']}")

            # Step 5
            if not st.session_state.recommendations:
                update(72, "🎬 Loading 5,000+ movies...")
                movies = fetch_movies(total=5000)
                embeddings = load_or_build_embeddings(movies)
                raw_recs = recommend_movies(personality["mood_description"], movies, embeddings, top_n=10)
                update(88, "✨ Writing AI explanations...")
                top_artists = user_data["artists_df"]["artist_name"].tolist()[:5]
                recommendations = explain_all_recommendations(personality, emotion_profile, raw_recs, top_artists)
                st.session_state.recommendations = recommendations
            update(95, "Almost done...", "10 movies matched & explained")

            # Save history
            try:
                user_id = user_data["user_info"]["user_id"] or "anonymous"
                user_name = user_data["user_info"]["display_name"]
                all_genres = user_data.get("all_genres", [])
                save_session(user_id, user_name, personality, st.session_state.recommendations, emotion_profile, all_genres)
            except Exception:
                pass  # History saving is non-critical

            update(100, "✅ Done!", "Results ready")
            st.session_state.analysis_done = True
            time.sleep(0.8)
            go_to("dashboard")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Check that all API keys are set in your .env file.")
            if st.button("Back to Home"):
                st.session_state.page = "home"
                st.session_state.token_info = None
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    personality = st.session_state.personality
    recommendations = st.session_state.recommendations
    user_data = st.session_state.user_data

    if not personality:
        go_to("home")
        return

    user_name = user_data["user_info"]["display_name"]

    # ── Top personality banner ────────────────────────────────────────────────
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.markdown(
            f"<div style='font-size:1.8rem;font-weight:900;color:#fff;margin-bottom:0.2rem'>"
            f"Hello, {user_name} 👋</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#555;font-size:0.9rem;'>Your music personality: "
            f"<span style='color:#1DB954;font-weight:700;'>{personality['emoji']} {personality['name']}</span> "
            f"· {personality['confidence']:.0%} confidence</div>",
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("👤 Profile", use_container_width=True):
            go_to("profile")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Two-col layout: personality card left, movies right ──────────────────
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:3.5rem">{personality['emoji']}</div>
            <div style="font-size:1.2rem;font-weight:800;color:#fff;margin:0.6rem 0 0.3rem">{personality['name']}</div>
            <div style="color:#777;font-size:0.85rem;line-height:1.5">{personality['description']}</div>
            <div style="color:#1DB954;font-size:0.8rem;font-weight:600;margin-top:1rem">
                {personality['confidence']:.0%} match confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Mini emotion bars
        emotion_profile = st.session_state.emotion_profile
        emotion_data = {k: v for k, v in emotion_profile.items() if k != "neutral" and v > 0.02}
        if emotion_data:
            st.markdown("<div class='section-title' style='font-size:1rem'>Emotion Profile</div>", unsafe_allow_html=True)
            top_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)[:5]
            for emotion, score in top_emotions:
                col_e, col_s = st.columns([2, 3])
                with col_e:
                    st.markdown(f"<div style='color:#bbb;font-size:0.82rem;padding-top:0.3rem'>{emotion}</div>", unsafe_allow_html=True)
                with col_s:
                    st.progress(score)

    with col_right:
        st.markdown(f"<div class='section-title'>🎬 Your Movie Picks</div>", unsafe_allow_html=True)

        # 2-column movie grid
        movies = recommendations or []
        for row_start in range(0, len(movies), 2):
            g1, g2 = st.columns(2)
            for col_widget, idx in [(g1, row_start), (g2, row_start + 1)]:
                if idx >= len(movies):
                    break
                movie = movies[idx]
                with col_widget:
                    _movie_grid_card(movie, idx)


def _movie_grid_card(movie: dict, idx: int):
    """Renders a compact movie card with a View Details button."""
    poster_url = movie.get("poster_url", "")
    title = movie.get("title", "Unknown")
    year = movie.get("release_year", "")
    score = movie.get("vote_average", 0)
    sim = movie.get("similarity_score", 0)
    genres = movie.get("genres", [])[:2]
    explanation = movie.get("claude_explanation", "")

    with st.container():
        if poster_url:
            col_poster, col_info = st.columns([1, 2])
            with col_poster:
                st.image(poster_url, use_container_width=True)
            with col_info:
                st.markdown(f"**{idx+1}. {title}** ({year})")
                st.markdown(
                    f'<span class="badge badge-yellow">⭐ {score:.1f}</span>'
                    f'<span class="badge badge-green">🎯 {sim:.0%}</span>',
                    unsafe_allow_html=True,
                )
                for g in genres:
                    st.markdown(f'<span class="badge">{g}</span>', unsafe_allow_html=True)
                if st.button("View Details", key=f"movie_btn_{idx}", use_container_width=True):
                    st.session_state.selected_movie = movie
                    go_to("movie_detail")
        else:
            st.markdown(f"**{idx+1}. {title}** ({year})")
            if st.button("View Details", key=f"movie_btn_{idx}", use_container_width=True):
                st.session_state.selected_movie = movie
                go_to("movie_detail")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MOVIE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def page_movie_detail():
    movie = st.session_state.selected_movie
    if not movie:
        go_to("dashboard")
        return

    if st.button("← Back"):
        go_to("dashboard")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_poster, col_info = st.columns([1, 2.5])

    with col_poster:
        if movie.get("poster_url"):
            st.image(movie["poster_url"], use_container_width=True)
        else:
            st.markdown(
                "<div style='font-size:6rem;text-align:center;padding:2rem 0'>🎬</div>",
                unsafe_allow_html=True,
            )

    with col_info:
        title = movie.get("title", "Unknown")
        year = movie.get("release_year", "")
        score = movie.get("vote_average", 0)
        sim = movie.get("similarity_score", 0)
        genres = movie.get("genres", [])
        overview = movie.get("overview", "")
        explanation = movie.get("claude_explanation", "")

        st.markdown(f"<div style='font-size:2rem;font-weight:900;color:#fff;line-height:1.2'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#555;margin-bottom:1rem'>{year}</div>", unsafe_allow_html=True)

        # Badges
        badges = (
            f'<span class="badge badge-yellow">⭐ {score:.1f} / 10</span>'
            f'<span class="badge badge-green">🎯 {sim:.0%} personality match</span>'
        )
        for g in genres:
            badges += f'<span class="badge badge-blue">{g}</span>'
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#1DB954;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem'>Why this film?</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#111;border:1px solid #1e1e1e;border-radius:10px;padding:1.2rem;"
            f"color:#ddd;font-style:italic;line-height:1.7;font-size:0.95rem'>{explanation}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#777;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem'>Overview</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#999;font-size:0.9rem;line-height:1.7'>{overview}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def page_profile():
    user_data = st.session_state.user_data
    personality = st.session_state.personality
    emotion_profile = st.session_state.emotion_profile

    if not user_data:
        go_to("home")
        return

    user_info = user_data["user_info"]
    tracks_df = user_data["tracks_df"]
    artists_df = user_data["artists_df"]
    all_genres = user_data.get("all_genres", [])

    st.markdown(f"<div class='section-title' style='font-size:2rem;'>👤 Your Spotify Profile</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#555;margin-bottom:2rem'>{user_info['display_name']} · {user_info.get('country','')}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Top Artists ───────────────────────────────────────────────────────────
    with col1:
        st.markdown("<div class='section-title'>🎤 Top Artists</div>", unsafe_allow_html=True)
        for i, row in artists_df.head(10).iterrows():
            artist_genres = row.get("genres", [])
            genre_badges = "".join(f'<span class="badge">{g}</span>' for g in artist_genres[:2])
            st.markdown(
                f"<div style='padding:0.6rem 0;border-bottom:1px solid #1a1a1a;'>"
                f"<span style='color:#fff;font-weight:600;font-size:0.9rem'>{i+1}. {row['artist_name']}</span>"
                f"<div style='margin-top:0.2rem'>{genre_badges}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Genres + Audio Radar ─────────────────────────────────────────────────
    with col2:
        st.markdown("<div class='section-title'>🎼 Your Genre Mix</div>", unsafe_allow_html=True)
        if all_genres:
            from collections import Counter
            genre_counts = Counter(all_genres).most_common(12)
            genre_html = "".join(
                f'<span class="badge artist-chip" style="font-size:{0.7 + (count/genre_counts[0][1])*0.25:.2f}rem">{g}</span>'
                for g, count in genre_counts
            )
            st.markdown(f"<div style='line-height:2.5'>{genre_html}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎵 Audio DNA</div>", unsafe_allow_html=True)

        audio_cols = ["energy", "valence", "danceability", "acousticness"]
        audio_values = [float(tracks_df[c].mean()) for c in audio_cols]

        fig = go.Figure(data=go.Scatterpolar(
            r=audio_values + [audio_values[0]],
            theta=["Energy", "Valence", "Danceability", "Acousticness", "Energy"],
            fill="toself",
            fillcolor="rgba(29,185,84,0.18)",
            line=dict(color="#1DB954", width=2),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], color="#333", gridcolor="#222"),
                angularaxis=dict(color="#555"),
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Top Tracks ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🎵 Top Tracks</div>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    tracks = tracks_df[["track_name", "artist_name"]].head(20).to_dict("records")
    for idx, track in enumerate(tracks):
        col = col_t1 if idx < 10 else col_t2
        with col:
            st.markdown(
                f"<div style='padding:0.4rem 0;border-bottom:1px solid #141414;"
                f"color:#bbb;font-size:0.87rem'>"
                f"<span style='color:#555;margin-right:0.5rem'>{idx+1}.</span>"
                f"<b style='color:#ddd'>{track['track_name']}</b> "
                f"<span style='color:#555'>— {track['artist_name']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Emotion chart ─────────────────────────────────────────────────────────
    if emotion_profile:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>😊 Emotion Breakdown</div>", unsafe_allow_html=True)
        emotion_data = {k: v for k, v in emotion_profile.items() if k != "neutral" and v > 0.01}
        if emotion_data:
            fig_e = px.bar(
                x=list(emotion_data.keys()),
                y=list(emotion_data.values()),
                color=list(emotion_data.values()),
                color_continuous_scale=[[0, "#1a1a2e"], [0.5, "#1DB954"], [1, "#fff"]],
            )
            fig_e.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#888"),
                showlegend=False,
                coloraxis_showscale=False,
                height=260,
                margin=dict(t=10, b=10),
                xaxis=dict(title="", gridcolor="#1a1a1a"),
                yaxis=dict(title="Score", gridcolor="#1a1a1a"),
            )
            st.plotly_chart(fig_e, use_container_width=True)

    # ── Personality map ───────────────────────────────────────────────────────
    if personality and st.session_state.feature_vector is not None:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Personality Map</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#555;font-size:0.85rem;margin-bottom:1rem'>Your position in the 6-cluster PCA space</div>", unsafe_allow_html=True)

        pca_data = get_pca_coordinates(st.session_state.feature_vector)
        fig_pca = go.Figure()

        for c in pca_data["centroid_coords"]:
            fig_pca.add_trace(go.Scatter(
                x=[c["x"]], y=[c["y"]],
                mode="markers+text",
                marker=dict(size=20, color=c["color"], opacity=0.6, line=dict(width=1, color="#fff")),
                text=[f"{c['emoji']} {c['name']}"],
                textposition="top center",
                name=c["name"],
            ))

        fig_pca.add_trace(go.Scatter(
            x=[pca_data["user_x"]], y=[pca_data["user_y"]],
            mode="markers+text",
            marker=dict(size=16, color="#1DB954", symbol="star", line=dict(width=1, color="#fff")),
            text=["⭐ You"],
            textposition="top center",
            name="You",
        ))

        fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,15,1)",
            font=dict(color="#888"),
            height=420,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # Match scores
        st.markdown("<div class='section-title' style='font-size:1rem'>Match Scores</div>", unsafe_allow_html=True)
        similarities = personality.get("all_similarities", {})
        for pname, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            emoji = next((p["emoji"] for p in PERSONALITY_PROFILES.values() if p["name"] == pname), "🎵")
            sc1, sc2 = st.columns([2, 5])
            with sc1:
                st.markdown(f"<div style='color:#bbb;font-size:0.85rem;padding-top:0.3rem'>{emoji} {pname}</div>", unsafe_allow_html=True)
            with sc2:
                st.progress(score)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def page_history():
    user_data = st.session_state.user_data
    if not user_data:
        go_to("home")
        return

    user_id = user_data["user_info"].get("user_id", "anonymous")
    user_name = user_data["user_info"]["display_name"]
    history = load_history(user_id)

    st.markdown(f"<div class='section-title' style='font-size:2rem;'>📚 Recommendation History</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#555;margin-bottom:2rem'>Past sessions for {user_name}</div>", unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem">🎬</div>
            <div style="color:#555">No history yet.<br>Your sessions will appear here after your first analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for entry in history:
        ts_raw = entry.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw).strftime("%B %d, %Y · %H:%M")
        except Exception:
            ts = ts_raw

        p = entry.get("personality", {})
        movies = entry.get("movies", [])
        emotions = entry.get("emotion_profile", {})
        top_genres = entry.get("top_genres", [])

        # Top 3 emotion
        top_emo = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        emo_text = " · ".join(f"{e} {s:.0%}" for e, s in top_emo)

        st.markdown(f"""
        <div class="history-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.8rem">
                <div>
                    <span style="font-size:1.4rem">{p.get('emoji','🎵')}</span>
                    <span style="font-size:1rem;font-weight:700;color:#fff;margin-left:0.5rem">{p.get('name','')}</span>
                    <span style="color:#1DB954;font-size:0.8rem;margin-left:0.5rem">{p.get('confidence',0):.0%} match</span>
                </div>
                <div style="color:#444;font-size:0.78rem">{ts}</div>
            </div>
            <div style="color:#555;font-size:0.82rem;margin-bottom:0.8rem">{emo_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Movie pills for this session
        with st.expander(f"View {len(movies)} movies from this session"):
            for m in movies:
                mcol1, mcol2 = st.columns([1, 4])
                with mcol1:
                    if m.get("poster_url"):
                        st.image(m["poster_url"], width=80)
                    else:
                        st.markdown("🎬")
                with mcol2:
                    genres_html = "".join(f'<span class="badge">{g}</span>' for g in m.get("genres", []))
                    st.markdown(
                        f"<b style='color:#fff'>{m['title']}</b> "
                        f"<span style='color:#555'>({m.get('release_year','')})</span>"
                        f'<span class="badge badge-yellow" style="margin-left:0.4rem">⭐ {m.get("vote_average",0):.1f}</span>'
                        f'<span class="badge badge-green">🎯 {m.get("similarity_score",0):.0%}</span>'
                        f"<div style='margin-top:0.2rem'>{genres_html}</div>"
                        f"<div style='color:#666;font-size:0.82rem;font-style:italic;margin-top:0.3rem'>{m.get('explanation','')}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("<hr style='border-color:#1a1a1a;margin:0.5rem 0'>", unsafe_allow_html=True)

    st.markdown('<div class="footer">Music → Movie Match · History stored locally</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    handle_oauth_callback()

    page = st.session_state.page

    # Sidebar on all post-login pages
    if page in ("dashboard", "movie_detail", "profile", "history"):
        render_sidebar()

    routes = {
        "home":         page_home,
        "analyze":      page_analyze,
        "dashboard":    page_dashboard,
        "movie_detail": page_movie_detail,
        "profile":      page_profile,
        "history":      page_history,
    }

    routes.get(page, page_home)()


if __name__ == "__main__":
    main()
