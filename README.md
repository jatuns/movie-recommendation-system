# 🎵 Music Personality → Movie Recommendation System

Spotify dinleme alışkanlıklarını analiz ederek müzik kişilik profili oluşturan ve buna göre film öneren bir Data Science web uygulaması.

## Nasıl Çalışır?

```
Spotify OAuth → Top Tracks + Audio Features
     ↓
Genius API → Şarkı Sözleri
     ↓
HuggingFace → Duygu Analizi (joy, sadness, anger...)
     ↓
K-Means → Kişilik Profili (6 tip)
     ↓
TMDB + Sentence Transformers → Top 10 Film
     ↓
Claude API → Kişiselleştirilmiş Açıklamalar
     ↓
Streamlit → Web Arayüzü
```

## Kişilik Profilleri

| Profil | Açıklama |
|--------|----------|
| 🖤 Dark & Introspective | Melankoli, düşük valence, yüksek acousticness |
| ⚡ Energetic & Bold | Yüksek enerji, hızlı tempo, cesur |
| ☀️ Feel-good & Social | Yüksek valence, dans ritmi, neşe |
| 🌙 Moody & Atmospheric | Atmosferik, enstrümantal, gizemli |
| 🎭 Sophisticated & Complex | Karmaşık duygular, entelektüel |
| 🌿 Calm & Reflective | Düşük enerji, huzurlu, yansıtıcı |

## Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. API Anahtarları

`.env.example` dosyasını `.env` olarak kopyala ve anahtarları doldur:

```bash
cp .env.example .env
```

| API | Nereden Alınır | Ücretsiz mi? |
|-----|---------------|--------------|
| Spotify | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) | ✅ Evet |
| Genius | [genius.com/api-clients](https://genius.com/api-clients) | ✅ Evet |
| TMDB | [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) | ✅ Evet |
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com) | 💳 Ücretli (düşük maliyet) |

### 3. Spotify Redirect URI

Spotify Developer Dashboard'da uygulamanın redirect URI'sine şunu ekle:
```
http://localhost:8501/callback
```

### 4. Uygulamayı Çalıştır

```bash
streamlit run app.py
```

## Proje Yapısı

```
movie-recommendation-system/
├── app.py                    # Ana Streamlit uygulaması
├── src/
│   ├── spotify_collector.py  # Phase 1: Spotify OAuth + veri toplama
│   ├── nlp_analyzer.py       # Phase 2: Lyrics + duygu analizi
│   ├── personality_clustering.py  # Phase 3: K-Means kişilik profili
│   ├── movie_recommender.py  # Phase 4: TMDB + embedding tabanlı öneri
│   └── claude_explainer.py   # Phase 5: Claude API açıklamaları
├── data/                     # Cache (gitignore)
├── models/                   # Eğitilmiş modeller (gitignore)
├── requirements.txt
└── .env.example
```

## Tech Stack

- **Streamlit** — Web arayüzü
- **Spotipy** — Spotify OAuth 2.0
- **LyricsGenius** — Şarkı sözleri
- **HuggingFace Transformers** — Duygu analizi (`j-hartmann/emotion-english-distilroberta-base`)
- **Sentence Transformers** — Film embedding'leri (`all-MiniLM-L6-v2`)
- **scikit-learn** — K-Means kümeleme, PCA
- **Anthropic Claude** — Kişiselleştirilmiş açıklamalar
- **TMDB API** — Film veritabanı
- **Plotly** — İnteraktif görselleştirmeler

## Deployment (Streamlit Cloud)

1. GitHub'a push et
2. [share.streamlit.io](https://share.streamlit.io) üzerinden deploy et
3. Secrets bölümüne `.env` değişkenlerini gir
