"""
AIn - Machine Learning Models
Real ML-based anomaly detection using IsolationForest and ensemble methods.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """ML model prediction result."""
    anomaly_score: float  # 0-100, higher = more anomalous
    isolation_forest_score: float
    is_anomaly: bool
    feature_contributions: Dict[str, float]
    confidence: float


class ZeroTrustMLModel:
    """
    Machine Learning model for user behavior anomaly detection.

    Uses IsolationForest as the primary anomaly detector with
    feature engineering specific to authentication/session patterns.
    """

    FEATURE_NAMES = [
        "city_mismatch",
        "device_mismatch",
        "is_unusual_time",
        "failed_logins",
        "is_sensitive_service",
        "travel_distance_km",
        "session_velocity",  # Actions per minute
        "login_hour",  # Hour of day (cyclical)
    ]

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML model.

        Args:
            model_path: Path to load pre-trained model. If None, creates new model.
        """
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Expect ~10% anomalies
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def extract_features(self, session_data: Dict) -> np.ndarray:
        """
        Extract ML features from session data.

        Args:
            session_data: Dict with session attributes

        Returns:
            numpy array of features
        """
        # Basic features
        city_mismatch = 1 if session_data.get("city", "").lower() != session_data.get("usual_city", "").lower() else 0
        device_mismatch = 1 if session_data.get("device", "").lower() != session_data.get("usual_device", "").lower() else 0
        is_unusual_time = 1 if session_data.get("is_unusual_time", False) else 0
        failed_logins = min(session_data.get("failed_logins", 0), 10)  # Cap at 10
        is_sensitive = 1 if session_data.get("is_sensitive_service", False) else 0
        travel_distance = min(session_data.get("travel_distance_km", 0), 10000) / 1000  # Normalize to 0-10

        # Session velocity (if available)
        session_velocity = session_data.get("session_velocity", 0.5)

        # Login hour (normalized to 0-1)
        login_hour = session_data.get("login_hour", 12) / 24

        features = np.array([
            city_mismatch,
            device_mismatch,
            is_unusual_time,
            failed_logins / 10,  # Normalize
            is_sensitive,
            travel_distance,
            session_velocity,
            login_hour,
        ])

        return features.reshape(1, -1)

    def fit(self, training_data: List[Dict], labels: Optional[List[int]] = None):
        """
        Train the model on historical session data.

        Args:
            training_data: List of session dictionaries
            labels: Optional labels (1=normal, -1=anomaly) - not used for IF but useful for validation
        """
        # Extract features from all sessions
        X = np.vstack([self.extract_features(session) for session in training_data])

        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True

        logger.info(f"Model trained on {len(training_data)} samples")

    def predict(self, session_data: Dict) -> MLPrediction:
        """
        Predict anomaly score for a session.

        Args:
            session_data: Session attributes dict

        Returns:
            MLPrediction with scores and details
        """
        if not self.is_fitted:
            # Use heuristic scoring if not trained
            return self._heuristic_predict(session_data)

        # Extract and scale features
        features = self.extract_features(session_data)
        features_scaled = self.scaler.transform(features)

        # Get Isolation Forest score
        # score_samples returns negative values, more negative = more anomalous
        if_raw_score = self.isolation_forest.score_samples(features_scaled)[0]

        # Convert to 0-100 scale (more positive raw = more normal)
        # Typical range is -0.5 to 0.5
        if_normalized = 50 - (if_raw_score * 100)
        if_normalized = max(0, min(100, if_normalized))

        # Get prediction (-1 = anomaly, 1 = normal)
        is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1

        # Calculate feature contributions
        contributions = self._calculate_contributions(features[0], features_scaled[0])

        # Confidence based on how far from decision boundary
        confidence = min(1.0, abs(if_raw_score) * 2)

        return MLPrediction(
            anomaly_score=if_normalized,
            isolation_forest_score=if_normalized,
            is_anomaly=is_anomaly,
            feature_contributions=contributions,
            confidence=confidence,
        )

    def _heuristic_predict(self, session_data: Dict) -> MLPrediction:
        """Fallback heuristic prediction when model is not trained."""
        features = self.extract_features(session_data)[0]

        # Weighted heuristic scoring
        weights = {
            "city_mismatch": 20,
            "device_mismatch": 15,
            "is_unusual_time": 15,
            "failed_logins": 25,  # Per normalized unit
            "is_sensitive_service": 10,
            "travel_distance_km": 25,
            "session_velocity": 5,
            "login_hour": 5,
        }

        score = 0
        contributions = {}

        for i, (name, weight) in enumerate(zip(self.FEATURE_NAMES, weights.values())):
            contribution = features[i] * weight
            score += contribution
            contributions[name] = contribution

        score = min(100, max(0, score))

        return MLPrediction(
            anomaly_score=score,
            isolation_forest_score=score,
            is_anomaly=score > 50,
            feature_contributions=contributions,
            confidence=0.7,  # Lower confidence for heuristic
        )

    def _calculate_contributions(self, features: np.ndarray, features_scaled: np.ndarray) -> Dict[str, float]:
        """
        Calculate approximate feature contributions to the anomaly score.

        Uses a simple approach: compare each feature to the mean and weight by importance.
        """
        contributions = {}

        # Importance weights (derived from feature ranges and domain knowledge)
        importance = {
            "city_mismatch": 0.20,
            "device_mismatch": 0.15,
            "is_unusual_time": 0.10,
            "failed_logins": 0.20,
            "is_sensitive_service": 0.10,
            "travel_distance_km": 0.15,
            "session_velocity": 0.05,
            "login_hour": 0.05,
        }

        for i, name in enumerate(self.FEATURE_NAMES):
            # Scaled feature value contributes to anomaly if far from 0
            contrib = abs(features_scaled[i]) * importance.get(name, 0.1) * 100
            contributions[name] = round(contrib, 2)

        return contributions

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model_data = {
            "isolation_forest": self.isolation_forest,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        self.isolation_forest = model_data["isolation_forest"]
        self.scaler = model_data["scaler"]
        self.is_fitted = model_data["is_fitted"]
        logger.info(f"Model loaded from {path}")

    def train_from_logs(self, log_entries: List[Dict]):
        """
        Train model from parsed log entries.

        Converts log entries to session format and trains.
        """
        sessions = []

        for entry in log_entries:
            session = {
                "city": entry.get("city", "Unknown"),
                "usual_city": entry.get("city", "Unknown"),  # Assume same for training
                "device": entry.get("device", "Unknown"),
                "usual_device": entry.get("device", "Unknown"),
                "is_unusual_time": self._is_unusual_time(entry.get("timestamp", "")),
                "failed_logins": 1 if entry.get("status") == "failed" else 0,
                "is_sensitive_service": self._is_sensitive(entry.get("service", "")),
                "travel_distance_km": 0,
                "session_velocity": 0.5,
                "login_hour": self._extract_hour(entry.get("timestamp", "")),
            }
            sessions.append(session)

        if sessions:
            self.fit(sessions)

    def _is_unusual_time(self, timestamp: str) -> bool:
        """Check if timestamp is unusual (night hours)."""
        try:
            hour = int(timestamp.split("T")[1].split(":")[0])
            return hour < 6 or hour >= 23
        except (IndexError, ValueError):
            return False

    def _is_sensitive(self, service: str) -> bool:
        """Check if service is sensitive."""
        sensitive_keywords = ["password", "transfer", "bank", "visa", "exit", "delegation"]
        return any(kw in service.lower() for kw in sensitive_keywords)

    def _extract_hour(self, timestamp: str) -> int:
        """Extract hour from timestamp."""
        try:
            return int(timestamp.split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            return 12


def create_and_train_model(log_file: str = None) -> ZeroTrustMLModel:
    """
    Create and train a model, optionally from a log file.

    If no log file provided, generates synthetic training data.
    """
    model = ZeroTrustMLModel()

    if log_file and os.path.exists(log_file):
        # Train from log file
        with open(log_file, "r") as f:
            if log_file.endswith(".jsonl"):
                entries = [json.loads(line) for line in f if line.strip()]
            else:
                entries = json.load(f)

        model.train_from_logs(entries)
    else:
        # Generate synthetic training data
        logger.info("Generating synthetic training data...")
        training_data = []

        # Normal sessions
        for _ in range(500):
            training_data.append({
                "city": "Riyadh",
                "usual_city": "Riyadh",
                "device": "iPhone 14",
                "usual_device": "iPhone 14",
                "is_unusual_time": np.random.random() < 0.1,
                "failed_logins": 0,
                "is_sensitive_service": np.random.random() < 0.2,
                "travel_distance_km": np.random.uniform(0, 50),
                "session_velocity": np.random.uniform(0.1, 1),
                "login_hour": np.random.randint(8, 22),
            })

        # Anomalous sessions
        for _ in range(50):
            training_data.append({
                "city": "Cairo",
                "usual_city": "Riyadh",
                "device": "Windows PC",
                "usual_device": "iPhone 14",
                "is_unusual_time": True,
                "failed_logins": np.random.randint(2, 6),
                "is_sensitive_service": True,
                "travel_distance_km": np.random.uniform(500, 5000),
                "session_velocity": np.random.uniform(2, 5),
                "login_hour": np.random.choice([2, 3, 4, 23]),
            })

        model.fit(training_data)

    return model


if __name__ == "__main__":
    # Demo: create and test model
    model = create_and_train_model()

    # Test normal session
    normal_session = {
        "city": "Riyadh",
        "usual_city": "Riyadh",
        "device": "iPhone 14",
        "usual_device": "iPhone 14",
        "is_unusual_time": False,
        "failed_logins": 0,
        "is_sensitive_service": False,
        "travel_distance_km": 5,
    }

    print("\n--- Normal Session ---")
    result = model.predict(normal_session)
    print(f"Anomaly Score: {result.anomaly_score:.1f}")
    print(f"Is Anomaly: {result.is_anomaly}")

    # Test suspicious session
    suspicious_session = {
        "city": "Cairo",
        "usual_city": "Riyadh",
        "device": "Windows PC",
        "usual_device": "iPhone 14",
        "is_unusual_time": True,
        "failed_logins": 3,
        "is_sensitive_service": True,
        "travel_distance_km": 1600,
    }

    print("\n--- Suspicious Session ---")
    result = model.predict(suspicious_session)
    print(f"Anomaly Score: {result.anomaly_score:.1f}")
    print(f"Is Anomaly: {result.is_anomaly}")
    print(f"Feature Contributions: {result.feature_contributions}")

    # Save model
    model.save("./trained_model.joblib")
