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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,700&display=swap');

:root {
    --bg:      #1A1A2E;
    --surface: #16213E;
    --surface2: #0F3460;
    --border:  rgba(255,255,255,0.07);
    --green:   #1DB954;
    --coral:   #FF6B6B;
    --teal:    #4ECDC4;
    --yellow:  #FFE66D;
    --text:    #EAEAEA;
    --muted:   #6B7280;
    --font:    'Plus Jakarta Sans', sans-serif;
}

/* ─── Base ─── */
* { font-family: var(--font) !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stMain"] { background: var(--bg) !important; }
[data-testid="block-container"] { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] {
    background: #12122a !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stApp { background: var(--bg) !important; }

/* ─── Buttons ─── */
.stButton button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    transition: all 0.18s !important;
}
.stButton button:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
    transform: translateY(-1px) !important;
}

/* ─── Progress bar ─── */
.stProgress > div > div > div > div { background: var(--green) !important; }

/* ─── Inputs / selects ─── */
input, select, textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* ─── Expander ─── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 600 !important; }

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════
   LANDING PAGE
   ═══════════════════════════════════════════════════════════ */
.lp-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.lp-logo {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
}
.lp-logo span { color: var(--green); font-style: italic; }
.lp-nav-links {
    display: flex;
    gap: 2rem;
    font-size: 0.88rem;
    color: var(--muted);
}
.lp-nav-links a { text-decoration: none; color: inherit; cursor: pointer; transition: color 0.15s; }
.lp-nav-links a:hover { color: var(--text); }

.hero-wrap {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    align-items: center;
    padding: 5rem 0 3rem;
    max-width: 1000px;
    margin: 0 auto;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(29,185,84,0.1);
    border: 1px solid rgba(29,185,84,0.25);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: var(--green);
    font-weight: 600;
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
}
.badge-pulse {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.hero-h1 {
    font-size: 3.2rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.08;
    letter-spacing: -0.04em;
    margin-bottom: 1rem;
}
.hero-h1 em {
    font-style: italic;
    color: var(--green);
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    line-height: 1.7;
    font-weight: 400;
    margin-bottom: 2rem;
}
.btn-cta {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--green);
    color: #000 !important;
    padding: 0.85rem 2rem;
    border-radius: 50px;
    font-size: 0.95rem;
    font-weight: 800;
    text-decoration: none !important;
    transition: background 0.18s, transform 0.12s;
    letter-spacing: 0.01em;
}
.btn-cta:hover { background: #1ed760; transform: translateY(-2px); }

.hero-trust {
    margin-top: 1.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.78rem;
    color: var(--muted);
}
.trust-dot { color: var(--teal); }

/* App mockup */
.mockup-wrap {
    background: var(--surface);
    border-radius: 18px;
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: 0 0 60px rgba(29,185,84,0.08), 0 20px 60px rgba(0,0,0,0.4);
    animation: floatUp 0.7s ease both;
}
@keyframes floatUp { from { opacity:0; transform: translateY(18px); } to { opacity:1; transform: translateY(0); } }
.mockup-bar {
    background: rgba(255,255,255,0.04);
    border-bottom: 1px solid var(--border);
    padding: 0.7rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.m-dot { width: 9px; height: 9px; border-radius: 50%; }
.m-dot-r { background: #ff5f57; }
.m-dot-y { background: #febc2e; }
.m-dot-g { background: #28c840; }
.mockup-url { font-size: 0.7rem; color: #4a4a6a; margin-left: 0.5rem; }
.mockup-inner { padding: 1.2rem; }
.mock-profile {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: rgba(29,185,84,0.07);
    border: 1px solid rgba(29,185,84,0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
}
.mock-emoji { font-size: 1.8rem; }
.mock-pname { font-size: 0.85rem; font-weight: 700; color: var(--text); }
.mock-conf { font-size: 0.7rem; color: var(--green); margin-top: 0.1rem; }
.mock-movies { display: flex; flex-direction: column; gap: 0.5rem; }
.mock-movie {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.6rem 0.8rem;
}
.mock-poster {
    width: 32px; height: 44px;
    border-radius: 5px;
    flex-shrink: 0;
}
.mock-title { font-size: 0.78rem; font-weight: 600; color: var(--text); }
.mock-meta { font-size: 0.68rem; color: var(--muted); margin-top: 0.1rem; }
.mock-star { color: var(--yellow); font-size: 0.68rem; }
.mock-coral { color: var(--coral); font-size: 0.68rem; font-weight: 600; }

/* Feature grid */
.feat-section {
    max-width: 1000px;
    margin: 0 auto;
    padding: 3rem 0 2rem;
    border-top: 1px solid var(--border);
}
.feat-section-label {
    font-size: 0.7rem;
    color: var(--green);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.feat-section-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-bottom: 2rem;
    line-height: 1.15;
}
.feat-section-title em { font-style: italic; color: var(--coral); }
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}
.feat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    transition: border-color 0.2s, transform 0.18s;
}
.feat-card:hover { border-color: rgba(255,255,255,0.15); transform: translateY(-2px); }
.feat-icon-box {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 1rem;
}
.feat-card-title { font-size: 0.92rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
.feat-card-desc { font-size: 0.8rem; color: var(--muted); line-height: 1.65; }

/* ═══════════════════════════════════════════════════════════
   SHARED COMPONENTS
   ═══════════════════════════════════════════════════════════ */

/* Badges */
.badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin: 0.1rem;
    background: rgba(255,255,255,0.05);
    color: var(--muted);
}
.badge-green { background: rgba(29,185,84,0.12); color: var(--green); border: 1px solid rgba(29,185,84,0.25); }
.badge-coral { background: rgba(255,107,107,0.12); color: var(--coral); border: 1px solid rgba(255,107,107,0.25); }
.badge-teal  { background: rgba(78,205,196,0.12); color: var(--teal);  border: 1px solid rgba(78,205,196,0.25); }
.badge-yellow{ background: rgba(255,230,109,0.12); color: var(--yellow); border: 1px solid rgba(255,230,109,0.25); }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
}
.profile-card {
    background: linear-gradient(135deg, rgba(29,185,84,0.08) 0%, var(--surface) 100%);
    border: 1px solid rgba(29,185,84,0.25);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}

/* Movie grid card */
.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 13px;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.18s, box-shadow 0.2s;
    margin-bottom: 0.6rem;
}
.movie-card:hover {
    border-color: var(--coral);
    transform: translateY(-3px);
    box-shadow: 0 0 24px rgba(255,107,107,0.12);
}

/* Artist chip */
.artist-chip {
    display: inline-block;
    background: rgba(78,205,196,0.1);
    border: 1px solid rgba(78,205,196,0.2);
    border-radius: 8px;
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    color: var(--teal);
    margin: 0.2rem;
    font-weight: 500;
}

/* Genre badge (coral) */
.genre-pill {
    display: inline-block;
    background: rgba(255,107,107,0.1);
    border: 1px solid rgba(255,107,107,0.2);
    border-radius: 20px;
    padding: 0.18rem 0.6rem;
    font-size: 0.7rem;
    color: var(--coral);
    margin: 0.1rem;
    font-weight: 600;
}

/* Star rating */
.star-rating { color: var(--yellow); font-size: 0.82rem; font-weight: 700; }

/* Section title */
.section-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

/* Divider */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* History card */
.history-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.history-card:hover { border-color: rgba(255,255,255,0.14); }

/* Footer */
.footer {
    text-align: center;
    color: #3a3a5a;
    font-size: 0.75rem;
    margin-top: 4rem;
    padding: 1.5rem 0;
    border-top: 1px solid var(--border);
}

/* Analyze page */
.analyze-wrap {
    max-width: 480px;
    margin: 0 auto;
    padding: 4rem 0;
    text-align: center;
}
.analyze-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
}
.analyze-sub { font-size: 0.9rem; color: var(--muted); margin-bottom: 2.5rem; }
.step-done { padding: 0.25rem 0; color: #4a4a6a; font-size: 0.82rem; }

/* Sidebar nav */
.sb-logo {
    padding: 1.2rem 0 0.4rem;
    font-size: 1rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
}
.sb-logo span { color: var(--green); }
.sb-user { font-size: 0.78rem; color: var(--muted); padding-bottom: 1rem; }

/* Fade animations */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease both; }
.fade-up-d1 { animation: fadeUp 0.5s ease both 0.1s; opacity: 0; animation-fill-mode: both; }
.fade-up-d2 { animation: fadeUp 0.5s ease both 0.2s; opacity: 0; animation-fill-mode: both; }
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
        "selected_movie": None,
        "prev_page": None,
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
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    if not st.session_state.token_info:
        return
    current = st.session_state.page
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">music<span>match</span></div>
        """, unsafe_allow_html=True)
        if st.session_state.user_data:
            name = st.session_state.user_data["user_info"]["display_name"]
            st.markdown(f'<div class="sb-user">Signed in as <b>{name}</b></div>', unsafe_allow_html=True)
        st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0 0 0.8rem'>", unsafe_allow_html=True)

        for page_key, icon, label in [("dashboard","🏠","Dashboard"),("profile","👤","Profile"),("history","📚","History")]:
            btn_style = "border-left: 3px solid var(--green) !important;" if current == page_key else ""
            if st.button(f"{icon}  {label}", key=f"nav_{page_key}", use_container_width=True):
                go_to(page_key)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin-bottom:0.8rem'>", unsafe_allow_html=True)
        if st.button("🚪  Disconnect", use_container_width=True, key="nav_disconnect"):
            for k in ["token_info","user_data","emotion_profile","feature_vector",
                      "personality","recommendations","analysis_done","selected_movie"]:
                st.session_state[k] = None
            st.session_state.analysis_done = False
            go_to("home")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LANDING
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    auth_url = get_auth_url()

    st.markdown(f"""
    <div style="max-width:1000px;margin:0 auto;padding:0 1rem;">

      <!-- NAV -->
      <div class="lp-nav fade-up">
        <div class="lp-logo">music<span>match</span></div>
        <div class="lp-nav-links">
          <a>How it works</a>
          <a>Technology</a>
          <a>Open source</a>
        </div>
        <a href="{auth_url}" class="btn-cta" target="_self" style="padding:0.55rem 1.3rem;font-size:0.82rem;">
          Connect Spotify
        </a>
      </div>

      <!-- HERO -->
      <div class="hero-wrap">
        <div class="fade-up">
          <div class="hero-badge">
            <span class="badge-pulse"></span>
            Powered by Groq · HuggingFace · TMDB
          </div>
          <h1 class="hero-h1">Your music,<br>your <em>movies.</em></h1>
          <p class="hero-sub">
            Connect Spotify. In 60 seconds, get 10 hand-picked films that match
            your exact music personality — analyzed by real AI, not just genre tags.
          </p>
          <a href="{auth_url}" class="btn-cta" target="_self">
            🎵 &nbsp; Connect with Spotify
          </a>
          <div class="hero-trust">
            <span class="trust-dot">✦</span> Read-only access
            <span class="trust-dot">✦</span> Nothing stored
            <span class="trust-dot">✦</span> Free forever
          </div>
        </div>

        <!-- APP MOCKUP -->
        <div class="mockup-wrap fade-up-d1">
          <div class="mockup-bar">
            <span class="m-dot m-dot-r"></span>
            <span class="m-dot m-dot-y"></span>
            <span class="m-dot m-dot-g"></span>
            <span class="mockup-url">musicmatch.app/dashboard</span>
          </div>
          <div class="mockup-inner">
            <div class="mock-profile">
              <div class="mock-emoji">🎭</div>
              <div>
                <div class="mock-pname">Sophisticated &amp; Complex</div>
                <div class="mock-conf">94% personality match</div>
              </div>
            </div>
            <div class="mock-movies">
              <div class="mock-movie">
                <div class="mock-poster" style="background:linear-gradient(135deg,#1a1a3e,#0f3460)"></div>
                <div>
                  <div class="mock-title">Blade Runner 2049</div>
                  <div class="mock-meta">
                    <span class="mock-star">★ 8.0</span>
                    &nbsp;·&nbsp;
                    <span class="mock-coral">97% match</span>
                  </div>
                </div>
              </div>
              <div class="mock-movie">
                <div class="mock-poster" style="background:linear-gradient(135deg,#2e1a1a,#5a1a1a)"></div>
                <div>
                  <div class="mock-title">The Grand Budapest Hotel</div>
                  <div class="mock-meta">
                    <span class="mock-star">★ 8.1</span>
                    &nbsp;·&nbsp;
                    <span class="mock-coral">94% match</span>
                  </div>
                </div>
              </div>
              <div class="mock-movie">
                <div class="mock-poster" style="background:linear-gradient(135deg,#1a2e1a,#1a4a2e)"></div>
                <div>
                  <div class="mock-title">Everything Everywhere</div>
                  <div class="mock-meta">
                    <span class="mock-star">★ 7.8</span>
                    &nbsp;·&nbsp;
                    <span class="mock-coral">91% match</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- FEATURE GRID -->
      <div class="feat-section fade-up-d2">
        <div class="feat-section-label">How it works</div>
        <div class="feat-section-title">Five steps, one <em>perfect</em> watchlist.</div>
        <div class="feat-grid">
          <div class="feat-card">
            <div class="feat-icon-box" style="background:rgba(29,185,84,0.12)">🎧</div>
            <div class="feat-card-title">Spotify Data</div>
            <div class="feat-card-desc">Your top 50 tracks and 20 artists are fetched via read-only OAuth — nothing is stored.</div>
          </div>
          <div class="feat-card">
            <div class="feat-icon-box" style="background:rgba(255,107,107,0.12)">📝</div>
            <div class="feat-card-title">Lyrics Analysis</div>
            <div class="feat-card-desc">Genius API grabs lyrics. HuggingFace DistilRoBERTa extracts joy, sadness, anger and 4 more emotions.</div>
          </div>
          <div class="feat-card">
            <div class="feat-icon-box" style="background:rgba(78,205,196,0.12)">🧠</div>
            <div class="feat-card-title">Personality Cluster</div>
            <div class="feat-card-desc">K-Means assigns you to one of 6 music personalities — from Dark &amp; Introspective to Feel-good &amp; Social.</div>
          </div>
          <div class="feat-card">
            <div class="feat-icon-box" style="background:rgba(255,230,109,0.12)">🎬</div>
            <div class="feat-card-title">Movie Matching</div>
            <div class="feat-card-desc">Sentence Transformers embed your personality and 5,000+ TMDB films. Cosine similarity finds your top 10.</div>
          </div>
          <div class="feat-card">
            <div class="feat-icon-box" style="background:rgba(29,185,84,0.12)">✨</div>
            <div class="feat-card-title">AI Explanations</div>
            <div class="feat-card-desc">Groq (Llama 3) writes a 2-3 sentence personal reason for every recommendation — for free.</div>
          </div>
          <div class="feat-card">
            <div class="feat-icon-box" style="background:rgba(255,107,107,0.12)">📚</div>
            <div class="feat-card-title">Session History</div>
            <div class="feat-card-desc">Every session is saved locally. Come back anytime to revisit your past personality profiles and movie picks.</div>
          </div>
        </div>
      </div>

      <div class="footer">musicmatch · Streamlit · Spotify · HuggingFace · Groq · TMDB</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

def page_analyze():
    if st.session_state.analysis_done:
        go_to("dashboard")
        return

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class="analyze-wrap">
            <div class="analyze-title">Analyzing your music</div>
            <div class="analyze-sub">This takes about 30–60 seconds. Sit tight.</div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status = st.empty()
        steps_log = st.empty()
        completed_steps: list = []

        def update(pct: int, text: str, done: str = ""):
            progress_bar.progress(pct)
            status.markdown(
                f"<div style='text-align:center;color:var(--green);font-size:0.9rem;margin:0.6rem 0'>{text}</div>",
                unsafe_allow_html=True,
            )
            if done:
                completed_steps.append(done)
            log_html = "".join(
                f"<div class='step-done'>✅ {s}</div>" for s in completed_steps
            )
            steps_log.markdown(
                f"<div style='text-align:center;margin-top:0.8rem'>{log_html}</div>",
                unsafe_allow_html=True,
            )

        try:
            if not st.session_state.user_data:
                update(5, "🎵 Fetching your Spotify data...")
                sp = get_spotify_client(st.session_state.token_info)
                user_data = collect_user_data(sp)
                st.session_state.user_data = user_data
            else:
                user_data = st.session_state.user_data
            update(20, "Ready", "Fetched top tracks & artists")

            if not st.session_state.emotion_profile:
                update(25, "📝 Fetching lyrics via Genius API...")
                tracks_with_lyrics = fetch_lyrics_for_tracks(user_data["tracks_df"], top_n=20)
                update(45, "🧠 Running emotion analysis (HuggingFace)...")
                emotion_profile = compute_emotion_profile(tracks_with_lyrics)
                st.session_state.emotion_profile = emotion_profile
                feature_vector = build_user_feature_vector(user_data["tracks_df"], emotion_profile)
                st.session_state.feature_vector = feature_vector
            else:
                emotion_profile = st.session_state.emotion_profile
                feature_vector = st.session_state.feature_vector
            update(60, "Ready", "Emotion profile built")

            if not st.session_state.personality:
                update(62, "🎭 Determining music personality...")
                personality = assign_personality(feature_vector)
                st.session_state.personality = personality
            else:
                personality = st.session_state.personality
            update(70, "Ready", f"Personality: {personality['emoji']} {personality['name']}")

            if not st.session_state.recommendations:
                update(72, "🎬 Loading 5,000+ movies...")
                movies = fetch_movies(total=5000)
                embeddings = load_or_build_embeddings(movies)
                raw_recs = recommend_movies(personality["mood_description"], movies, embeddings, top_n=10)
                update(88, "✨ Writing AI explanations (Groq)...")
                top_artists = user_data["artists_df"]["artist_name"].tolist()[:5]
                recommendations = explain_all_recommendations(personality, emotion_profile, raw_recs, top_artists)
                st.session_state.recommendations = recommendations
            update(95, "Almost there...", "10 movies matched & explained")

            try:
                user_id = user_data["user_info"]["user_id"] or "anonymous"
                save_session(
                    user_id,
                    user_data["user_info"]["display_name"],
                    personality,
                    st.session_state.recommendations,
                    emotion_profile,
                    user_data.get("all_genres", []),
                )
            except Exception:
                pass

            update(100, "✅ Done!")
            st.session_state.analysis_done = True
            time.sleep(0.8)
            go_to("dashboard")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Check that all API keys are set in your .env file.")
            if st.button("← Back to Home"):
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

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_btn = st.columns([6, 1])
    with col_h:
        st.markdown(
            f"<div style='font-size:1.7rem;font-weight:800;color:var(--text);letter-spacing:-0.03em'>"
            f"Hey, {user_name} 👋</div>"
            f"<div style='color:var(--muted);font-size:0.88rem;margin-top:0.2rem;'>Your music personality: "
            f"<span style='color:var(--green);font-weight:700'>{personality['emoji']} {personality['name']}</span>"
            f" · <span style='color:var(--teal)'>{personality['confidence']:.0%} confidence</span></div>",
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("👤 Profile", use_container_width=True):
            go_to("profile")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2.6])

    with col_left:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:3.2rem;margin-bottom:0.5rem">{personality['emoji']}</div>
            <div style="font-size:1.1rem;font-weight:800;color:var(--text);margin-bottom:0.4rem;letter-spacing:-0.02em">
                {personality['name']}
            </div>
            <div style="color:var(--muted);font-size:0.82rem;line-height:1.6;">{personality['description']}</div>
            <div style="margin-top:1rem;">
                <span class="badge badge-green">{personality['confidence']:.0%} match</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        emotion_profile = st.session_state.emotion_profile
        emotion_data = {k: v for k, v in emotion_profile.items() if k != "neutral" and v > 0.02}
        if emotion_data:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem'>Emotion Profile</div>", unsafe_allow_html=True)
            for emotion, score in sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)[:5]:
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.markdown(f"<div style='color:var(--text);font-size:0.8rem;padding-top:0.3rem'>{emotion}</div>", unsafe_allow_html=True)
                with c2:
                    st.progress(score)

    with col_right:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem'>🎬 Your Movie Picks</div>", unsafe_allow_html=True)
        movies = recommendations or []
        for row_start in range(0, len(movies), 2):
            g1, g2 = st.columns(2)
            for col_widget, idx in [(g1, row_start), (g2, row_start + 1)]:
                if idx >= len(movies):
                    break
                with col_widget:
                    _movie_grid_card(movies[idx], idx)


def _movie_grid_card(movie: dict, idx: int):
    poster_url = movie.get("poster_url", "")
    title = movie.get("title", "Unknown")
    year = movie.get("release_year", "")
    score = movie.get("vote_average", 0)
    sim = movie.get("similarity_score", 0)
    genres = movie.get("genres", [])[:2]

    with st.container():
        if poster_url:
            cp, ci = st.columns([1, 2])
            with cp:
                st.image(poster_url, use_container_width=True)
            with ci:
                st.markdown(
                    f"<div style='font-size:0.85rem;font-weight:700;color:var(--text);line-height:1.3;margin-bottom:0.3rem'>"
                    f"{idx+1}. {title} <span style='color:var(--muted);font-weight:400'>({year})</span></div>",
                    unsafe_allow_html=True,
                )
                genre_pills = "".join(f'<span class="genre-pill">{g}</span>' for g in genres)
                st.markdown(
                    f'<span class="star-rating">★ {score:.1f}</span>'
                    f'<span class="badge badge-green" style="margin-left:0.3rem">🎯 {sim:.0%}</span>'
                    f"<div style='margin-top:0.3rem'>{genre_pills}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Details →", key=f"movie_btn_{idx}", use_container_width=True):
                    st.session_state.selected_movie = movie
                    go_to("movie_detail")
        else:
            st.markdown(f"**{idx+1}. {title}** ({year})")
            if st.button("Details →", key=f"movie_btn_{idx}", use_container_width=True):
                st.session_state.selected_movie = movie
                go_to("movie_detail")
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MOVIE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def page_movie_detail():
    movie = st.session_state.selected_movie
    if not movie:
        go_to("dashboard")
        return

    if st.button("← Back to Dashboard"):
        go_to("dashboard")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_poster, col_info = st.columns([1, 2.6])

    with col_poster:
        if movie.get("poster_url"):
            st.image(movie["poster_url"], use_container_width=True)
        else:
            st.markdown(
                "<div style='font-size:5rem;text-align:center;padding:3rem 0;background:var(--surface);border-radius:14px'>🎬</div>",
                unsafe_allow_html=True,
            )

    with col_info:
        title = movie.get("title", "")
        year = movie.get("release_year", "")
        score = movie.get("vote_average", 0)
        sim = movie.get("similarity_score", 0)
        genres = movie.get("genres", [])
        overview = movie.get("overview", "")
        explanation = movie.get("claude_explanation", "")

        st.markdown(
            f"<div style='font-size:2.2rem;font-weight:800;color:var(--text);line-height:1.1;letter-spacing:-0.03em'>{title}</div>"
            f"<div style='color:var(--muted);font-size:0.9rem;margin:0.3rem 0 1rem'>{year}</div>",
            unsafe_allow_html=True,
        )

        genre_pills = "".join(f'<span class="genre-pill">{g}</span>' for g in genres)
        st.markdown(
            f'<span class="star-rating">★ {score:.1f} / 10</span>&nbsp;'
            f'<span class="badge badge-green">🎯 {sim:.0%} personality match</span>'
            f"<div style='margin-top:0.5rem'>{genre_pills}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:var(--green);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Why this film?</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:rgba(29,185,84,0.06);border:1px solid rgba(29,185,84,0.15);"
            f"border-radius:12px;padding:1.2rem;color:var(--text);font-style:italic;line-height:1.75;font-size:0.9rem'>"
            f"{explanation}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Overview</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='color:var(--muted);font-size:0.88rem;line-height:1.75'>{overview}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def page_profile():
    user_data = st.session_state.user_data
    if not user_data:
        go_to("home")
        return

    user_info = user_data["user_info"]
    tracks_df = user_data["tracks_df"]
    artists_df = user_data["artists_df"]
    all_genres = user_data.get("all_genres", [])
    personality = st.session_state.personality
    emotion_profile = st.session_state.emotion_profile

    st.markdown(
        f"<div style='font-size:1.8rem;font-weight:800;color:var(--text);letter-spacing:-0.03em'>"
        f"👤 {user_info['display_name']}</div>"
        f"<div style='color:var(--muted);font-size:0.85rem;margin:0.2rem 0 1.5rem'>"
        f"Spotify · {user_info.get('country','')}</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem'>🎤 Top Artists</div>", unsafe_allow_html=True)
        for i, row in artists_df.head(10).iterrows():
            artist_genres = row.get("genres", [])
            genre_html = "".join(f'<span class="artist-chip">{g}</span>' for g in artist_genres[:2])
            st.markdown(
                f"<div style='padding:0.55rem 0;border-bottom:1px solid var(--border)'>"
                f"<span style='color:var(--muted);font-size:0.75rem;margin-right:0.5rem'>{i+1}</span>"
                f"<span style='color:var(--text);font-weight:600;font-size:0.88rem'>{row['artist_name']}</span>"
                f"<div style='margin-top:0.2rem'>{genre_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem'>🎼 Genre Mix</div>", unsafe_allow_html=True)
        if all_genres:
            from collections import Counter
            genre_counts = Counter(all_genres).most_common(14)
            max_count = genre_counts[0][1] if genre_counts else 1
            genre_html = "".join(
                f'<span class="genre-pill" style="font-size:{0.68 + (c/max_count)*0.2:.2f}rem;opacity:{0.6+(c/max_count)*0.4:.2f}">{g}</span>'
                for g, c in genre_counts
            )
            st.markdown(f"<div style='line-height:2.4'>{genre_html}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem'>🎵 Audio DNA</div>", unsafe_allow_html=True)

        audio_cols = ["energy", "valence", "danceability", "acousticness"]
        audio_values = [float(tracks_df[c].mean()) for c in audio_cols]
        fig_audio = go.Figure(data=go.Scatterpolar(
            r=audio_values + [audio_values[0]],
            theta=["Energy", "Valence", "Danceability", "Acousticness", "Energy"],
            fill="toself",
            fillcolor="rgba(29,185,84,0.12)",
            line=dict(color="#1DB954", width=2),
        ))
        fig_audio.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], color="#3a3a5a", gridcolor="#2a2a4a"),
                angularaxis=dict(color="#6B7280"),
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", family="Plus Jakarta Sans"),
            height=260,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_audio, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem'>🎵 Top Tracks</div>", unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    tracks = tracks_df[["track_name", "artist_name"]].head(20).to_dict("records")
    for i, track in enumerate(tracks):
        col = tc1 if i < 10 else tc2
        with col:
            st.markdown(
                f"<div style='padding:0.35rem 0;border-bottom:1px solid var(--border);font-size:0.83rem'>"
                f"<span style='color:var(--muted);margin-right:0.5rem'>{i+1}.</span>"
                f"<b style='color:var(--text)'>{track['track_name']}</b>"
                f"<span style='color:var(--muted)'> — {track['artist_name']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if emotion_profile:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem'>😊 Emotion Breakdown</div>", unsafe_allow_html=True)
        emotion_data = {k: v for k, v in emotion_profile.items() if k != "neutral" and v > 0.01}
        if emotion_data:
            fig_e = go.Figure(go.Bar(
                x=list(emotion_data.keys()),
                y=list(emotion_data.values()),
                marker_color=["#1DB954","#FF6B6B","#4ECDC4","#FFE66D","#c084fc","#f97316","#60a5fa"][:len(emotion_data)],
                marker_line_width=0,
            ))
            fig_e.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7280", family="Plus Jakarta Sans"),
                showlegend=False,
                height=240,
                margin=dict(t=10, b=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_e, use_container_width=True)

    if personality and st.session_state.feature_vector is not None:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem'>📊 Personality Map</div>", unsafe_allow_html=True)
        pca_data = get_pca_coordinates(st.session_state.feature_vector)
        fig_pca = go.Figure()
        for c in pca_data["centroid_coords"]:
            fig_pca.add_trace(go.Scatter(
                x=[c["x"]], y=[c["y"]],
                mode="markers+text",
                marker=dict(size=20, color=c["color"], opacity=0.55, line=dict(width=1, color="rgba(255,255,255,0.3)")),
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
            plot_bgcolor="rgba(22,33,62,1)",
            font=dict(color="#6B7280", family="Plus Jakarta Sans"),
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem'>Match Scores</div>", unsafe_allow_html=True)
        for pname, score in sorted(personality.get("all_similarities", {}).items(), key=lambda x: x[1], reverse=True):
            emoji = next((p["emoji"] for p in PERSONALITY_PROFILES.values() if p["name"] == pname), "🎵")
            sc1, sc2 = st.columns([2, 5])
            with sc1:
                st.markdown(f"<div style='color:var(--text);font-size:0.82rem;padding-top:0.3rem'>{emoji} {pname}</div>", unsafe_allow_html=True)
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
    history = load_history(user_id)

    st.markdown(
        "<div style='font-size:1.8rem;font-weight:800;color:var(--text);letter-spacing:-0.03em'>📚 History</div>"
        "<div style='color:var(--muted);font-size:0.85rem;margin:0.2rem 0 1.5rem'>Your past recommendation sessions</div>",
        unsafe_allow_html=True,
    )

    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:4rem;">
            <div style="font-size:3rem;margin-bottom:1rem">🎬</div>
            <div style="color:var(--muted)">No sessions yet. Your results will appear here after the first analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for entry in history:
        try:
            ts = datetime.fromisoformat(entry.get("timestamp","")).strftime("%b %d, %Y · %H:%M")
        except Exception:
            ts = entry.get("timestamp", "")

        p = entry.get("personality", {})
        movies = entry.get("movies", [])
        emotions = entry.get("emotion_profile", {})
        top_emo = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        emo_text = " · ".join(f"{e} {s:.0%}" for e, s in top_emo)

        st.markdown(f"""
        <div class="history-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <span style="font-size:1.3rem">{p.get('emoji','🎵')}</span>
                    <span style="font-size:0.95rem;font-weight:800;color:var(--text);margin-left:0.5rem">{p.get('name','')}</span>
                    <span class="badge badge-green" style="margin-left:0.5rem">{p.get('confidence',0):.0%} match</span>
                </div>
                <div style="color:var(--muted);font-size:0.75rem">{ts}</div>
            </div>
            <div style="color:var(--muted);font-size:0.78rem;margin-top:0.5rem">{emo_text}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"🎬 {len(movies)} movies from this session"):
            for m in movies:
                mc1, mc2 = st.columns([1, 5])
                with mc1:
                    if m.get("poster_url"):
                        st.image(m["poster_url"], width=72)
                    else:
                        st.markdown("🎬")
                with mc2:
                    genre_pills = "".join(f'<span class="genre-pill">{g}</span>' for g in m.get("genres",[]))
                    st.markdown(
                        f"<b style='color:var(--text)'>{m['title']}</b>"
                        f"<span style='color:var(--muted)'> ({m.get('release_year','')})</span>"
                        f'<span class="star-rating" style="margin-left:0.5rem">★ {m.get("vote_average",0):.1f}</span>'
                        f'<span class="badge badge-green" style="margin-left:0.3rem">🎯 {m.get("similarity_score",0):.0%}</span>'
                        f"<div style='margin-top:0.25rem'>{genre_pills}</div>"
                        f"<div style='color:var(--muted);font-size:0.8rem;font-style:italic;margin-top:0.3rem;line-height:1.5'>{m.get('explanation','')}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("<hr style='border-color:var(--border);margin:0.4rem 0'>", unsafe_allow_html=True)

    st.markdown('<div class="footer">musicmatch · History stored locally on your machine</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    handle_oauth_callback()

    page = st.session_state.page
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
