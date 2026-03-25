"""
Phase 3: Personality Clustering
Assigns the user's feature vector to a personality profile using K-Means (k=6).
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 6 personality profile definitions
PERSONALITY_PROFILES = {
    0: {
        "name": "Dark & Introspective",
        "emoji": "🖤",
        "description": (
            "Melancholy and introspection define you. Emotional depth, "
            "silence, and complex themes are at the core of your music."
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
            "High energy and boldness define you. Powerful rhythms, "
            "driving tempos, and a fearless attitude."
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
            "Joy and social energy define you. Positive vibes, "
            "dance rhythms, and embracing life to the fullest."
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
            "Atmospheric and mood-shifting. Instrumental depth, "
            "cinematic scope, and mysterious corners."
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
            "Complexity and elegance define you. Multi-layered narratives, "
            "intellectual depth, and artistic refinement."
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
            "Calm and reflection define you. Slow tempos, "
            "a love of nature, and a deep search for inner peace."
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
    Loads a saved model if available, otherwise creates a new one with synthetic data.
    In production, the model should be trained on a larger real dataset.
    """
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return kmeans, scaler

    # Build representative synthetic centroids for each profile
    # Feature order: energy, valence, tempo, danceability, acousticness,
    #                instrumentalness, speechiness, joy, sadness, anger,
    #                fear, surprise, disgust
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

    # Synthetic training data: 50 points around each centroid
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
    Assigns a personality profile based on the user's feature vector.
    Returns: profile info + cluster_id + distances
    """
    kmeans, scaler = _load_or_create_model()

    vector_2d = feature_vector.reshape(1, -1)
    vector_scaled = scaler.transform(vector_2d)

    cluster_id = int(kmeans.predict(vector_scaled)[0])

    # Distance to all centroids (0-1 normalized, closest = highest match)
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
    Positions the user in PCA space (for visualization).
    Returns: {"user_x": float, "user_y": float, "centroid_coords": list}
    """
    kmeans, scaler = _load_or_create_model()

    # Reduce centroids to 2D with PCA
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
