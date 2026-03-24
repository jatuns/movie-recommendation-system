"""
Phase 3: Personality Clustering
K-Means (k=6) ile kullanıcı feature vektörünü bir kişilik profiline atar.
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 6 kişilik profili tanımı
PERSONALITY_PROFILES = {
    0: {
        "name": "Dark & Introspective",
        "emoji": "🖤",
        "description": (
            "Melankoli ve içe dönüklük seni tanımlıyor. Duygusal derinlik, "
            "sessizlik ve karmaşık temalar müziğinin özünde."
        ),
        "mood_description": (
            "dark melancholic introspective emotional depth isolation sadness "
            "atmospheric haunting complex psychological"
        ),
        "color": "#4a4a8a",
    },
    1: {
        "name": "Energetic & Bold",
        "emoji": "⚡",
        "description": (
            "Yüksek enerji ve cesaret seni tanımlıyor. Güçlü ritimler, "
            "sürükleyici tempolar ve kararlı bir ruh hali."
        ),
        "mood_description": (
            "action adventure energetic bold intense thrilling powerful "
            "adrenaline fast-paced exciting dynamic"
        ),
        "color": "#e84c3d",
    },
    2: {
        "name": "Feel-good & Social",
        "emoji": "☀️",
        "description": (
            "Neşe ve sosyallik seni tanımlıyor. Pozitif titreşimler, "
            "dans ritimleri ve hayata dolu dolu sarılmak."
        ),
        "mood_description": (
            "feel-good comedy uplifting joyful friendship social warm "
            "heartwarming optimistic fun lighthearted"
        ),
        "color": "#f39c12",
    },
    3: {
        "name": "Moody & Atmospheric",
        "emoji": "🌙",
        "description": (
            "Atmosferik ve ruh hali değişken. Enstrümantal derinlik, "
            "sinematik genişlik ve gizemli köşeler."
        ),
        "mood_description": (
            "moody atmospheric mysterious cinematic ambient ethereal "
            "dreamlike surreal contemplative visual"
        ),
        "color": "#2ecc71",
    },
    4: {
        "name": "Sophisticated & Complex",
        "emoji": "🎭",
        "description": (
            "Karmaşıklık ve zariflik seni tanımlıyor. Çok katmanlı anlatılar, "
            "entelektüel derinlik ve sanatsal incelik."
        ),
        "mood_description": (
            "sophisticated complex drama intellectual artistic character-driven "
            "nuanced multi-layered thought-provoking literary"
        ),
        "color": "#9b59b6",
    },
    5: {
        "name": "Calm & Reflective",
        "emoji": "🌿",
        "description": (
            "Dinginlik ve yansıma seni tanımlıyor. Yavaş tempo, "
            "doğa sevgisi ve derin bir iç huzur arayışı."
        ),
        "mood_description": (
            "calm peaceful reflective slow-paced nature spiritual meditative "
            "quiet beautiful serene contemplative healing"
        ),
        "color": "#1abc9c",
    },
}

MODEL_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def _load_or_create_model() -> tuple:
    """
    Kaydedilmiş model varsa yükler, yoksa örnek verilerle yeni model oluşturur.
    Gerçek kullanımda model daha büyük bir veri seti ile eğitilmelidir.
    """
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return kmeans, scaler

    # Her profil için temsili sentetik merkez noktalar oluştur
    # Özellik sırası: energy, valence, tempo, danceability, acousticness,
    #                 instrumentalness, speechiness, joy, sadness, anger,
    #                 fear, surprise, disgust
    centroids = np.array([
        # 0: Dark & Introspective
        [0.3, 0.2, 0.3, 0.3, 0.7, 0.3, 0.1, 0.1, 0.6, 0.1, 0.2, 0.05, 0.1],
        # 1: Energetic & Bold
        [0.9, 0.7, 0.8, 0.8, 0.2, 0.1, 0.2, 0.4, 0.05, 0.4, 0.1, 0.3, 0.05],
        # 2: Feel-good & Social
        [0.7, 0.9, 0.7, 0.9, 0.3, 0.05, 0.15, 0.7, 0.05, 0.05, 0.05, 0.1, 0.05],
        # 3: Moody & Atmospheric
        [0.5, 0.4, 0.4, 0.4, 0.5, 0.6, 0.05, 0.1, 0.3, 0.1, 0.15, 0.1, 0.1],
        # 4: Sophisticated & Complex
        [0.5, 0.5, 0.5, 0.4, 0.4, 0.2, 0.15, 0.25, 0.25, 0.15, 0.1, 0.15, 0.1],
        # 5: Calm & Reflective
        [0.2, 0.5, 0.2, 0.3, 0.8, 0.4, 0.05, 0.3, 0.2, 0.05, 0.05, 0.05, 0.05],
    ])

    # Sentetik eğitim verisi: her centroid etrafında 50 nokta
    np.random.seed(42)
    X_train = np.vstack([
        centroids[i] + np.random.normal(0, 0.08, (50, 13))
        for i in range(6)
    ])
    X_train = np.clip(X_train, 0, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=6, init="k-means++", n_init=20, random_state=42)
    kmeans.fit(X_scaled)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(kmeans, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return kmeans, scaler


def assign_personality(feature_vector: np.ndarray) -> dict:
    """
    Kullanıcının feature vektörüne göre kişilik profili atar.
    Returns: profil bilgisi + cluster_id + distances
    """
    kmeans, scaler = _load_or_create_model()

    vector_2d = feature_vector.reshape(1, -1)
    vector_scaled = scaler.transform(vector_2d)

    cluster_id = int(kmeans.predict(vector_scaled)[0])

    # Tüm merkezlere uzaklık (0-1 normalize, en yakın = en yüksek uyum)
    distances = kmeans.transform(vector_scaled)[0]
    max_dist = distances.max()
    similarities = 1 - (distances / (max_dist + 1e-9))

    profile = PERSONALITY_PROFILES[cluster_id].copy()
    profile["cluster_id"] = cluster_id
    profile["confidence"] = float(similarities[cluster_id])
    profile["all_similarities"] = {
        PERSONALITY_PROFILES[i]["name"]: float(similarities[i])
        for i in range(6)
    }

    return profile


def get_pca_coordinates(feature_vector: np.ndarray) -> dict:
    """
    Kullanıcıyı PCA uzayında konumlandırır (görselleştirme için).
    Returns: {"user_x": float, "user_y": float, "centroid_coords": list}
    """
    kmeans, scaler = _load_or_create_model()

    # Merkez noktaları PCA ile 2D'ye indir
    centers_scaled = kmeans.cluster_centers_
    pca = PCA(n_components=2, random_state=42)
    pca.fit(centers_scaled)

    centers_2d = pca.transform(centers_scaled)
    user_2d = pca.transform(scaler.transform(feature_vector.reshape(1, -1)))

    return {
        "user_x": float(user_2d[0, 0]),
        "user_y": float(user_2d[0, 1]),
        "centroid_coords": [
            {
                "x": float(centers_2d[i, 0]),
                "y": float(centers_2d[i, 1]),
                "name": PERSONALITY_PROFILES[i]["name"],
                "emoji": PERSONALITY_PROFILES[i]["emoji"],
                "color": PERSONALITY_PROFILES[i]["color"],
            }
            for i in range(6)
        ],
    }
