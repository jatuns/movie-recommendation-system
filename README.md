# musicmatch

Spotify dinleme geçmişini analiz ederek müzik kişiliğini çıkaran ve buna göre kişiselleştirilmiş film öneren full-stack bir Data Science uygulaması.

## Nasıl Çalışır?

```
Spotify OAuth → Top Tracks + Top Artists
      ↓
Genius API → Şarkı sözleri (paralel, 5 thread)
      ↓
HuggingFace → 7 boyutlu duygu analizi (joy, sadness, anger, fear, surprise, disgust, neutral)
      ↓
K-Means (k=6) → Kişilik profili ataması + PCA koordinatları
      ↓
TMDB 5000 film → Sentence Transformer embedding → cosine similarity → Top 10
      ↓
Groq / Llama 3 → Her film için kişiselleştirilmiş açıklama
      ↓
FastAPI backend + Plain HTML/CSS/JS frontend
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

### 1. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 2. API anahtarlarını ayarla

```bash
cp .env.example .env
```

| API | Nereden | Ücretsiz? |
|-----|---------|-----------|
| Spotify | [developer.spotify.com](https://developer.spotify.com/dashboard) | ✅ |
| Genius | [genius.com/api-clients](https://genius.com/api-clients) | ✅ |
| TMDB | [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) | ✅ |
| Groq | [console.groq.com](https://console.groq.com) | ✅ (free tier) |

### 3. Spotify Redirect URI

Spotify Developer Dashboard'da uygulamanın Redirect URI'sine ekle:

```
http://127.0.0.1:8000/api/callback
```

### 4. Çalıştır

```bash
# FastAPI backend + HTML frontend (önerilen)
uvicorn api:app --reload --port 8000
# → http://localhost:8000

# veya Streamlit
streamlit run app.py
# → http://localhost:8501
```

## Proje Yapısı

```
musicmatch/
├── api.py                      # FastAPI backend (7 endpoint, background pipeline)
├── app.py                      # Streamlit arayüzü (alternatif)
├── frontend/
│   ├── index.html              # Landing page
│   ├── dashboard.html          # Film önerileri + analiz ekranı
│   ├── profile.html            # Kişilik profili, görselleştirmeler
│   ├── movie.html              # Film detay sayfası
│   └── history.html            # Geçmiş oturumlar
├── src/
│   ├── spotify_collector.py    # OAuth, top tracks/artists, genre tahmini
│   ├── nlp_analyzer.py         # Genius lyrics + HuggingFace duygu analizi
│   ├── personality_clustering.py  # K-Means, PCA görselleştirme
│   ├── movie_recommender.py    # TMDB fetch, Sentence Transformer embedding
│   ├── claude_explainer.py     # Groq/Llama 3 açıklamaları
│   └── history_store.py        # JSON tabanlı oturum geçmişi
├── models/                     # K-Means + scaler (gitignore)
├── data/                       # Film cache + embedding (gitignore)
├── requirements.txt
└── .env.example
```

## Frontend Sayfaları

| Sayfa | İçerik |
|-------|--------|
| **Dashboard** | 10 film önerisi (poster, açıklama, eşleşme yüzdesi), kişilik kartı, duygu profili |
| **Profile** | Top Artists (Spotify fotoğraflarıyla), Genre Cloud, Audio DNA radar chart, Personality Cluster Map (PCA scatter plot), Personality Affinity, Emotion Breakdown, Top Tracks (albüm kapağıyla) |
| **Movie** | Film detayı, AI açıklaması, benzer filmler |
| **History** | Geçmiş oturumlarda önerilen filmler |

## Tech Stack

- **FastAPI** — Backend API, background pipeline, session yönetimi
- **Spotipy** — Spotify OAuth 2.0
- **LyricsGenius** — Şarkı sözleri (paralel fetch)
- **HuggingFace Transformers** — `j-hartmann/emotion-english-distilroberta-base`
- **Sentence Transformers** — `all-MiniLM-L6-v2` film embedding
- **scikit-learn** — K-Means kümeleme, PCA, StandardScaler
- **Groq API** — Llama 3.1-8b-instant (ücretsiz tier)
- **TMDB API** — 5000+ film verisi
- **Tailwind CSS** — Styling (CDN)
- **Lucide SVG** — İkon sistemi (inline)

## Performans Notları

- İlk çalıştırmada film embedding'leri oluşturulur (~2-3 dk), sonraki çalıştırmalarda `data/movie_embeddings.npy` cache'den yüklenir.
- Lyrics analizi `ThreadPoolExecutor` (5 worker) ile paralel çalışır, süre ~15-25 saniyeye düşer.
- Duygu modeli (`_emotion_classifier`) modül düzeyinde cache'lenir, her istek için yeniden yüklenmez.
- Kullanıcı analiz sırasında sayfadan çıkarsa `navigator.sendBeacon` ile backend pipeline iptal edilir.
