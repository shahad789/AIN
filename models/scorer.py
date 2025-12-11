"""
AIn - Scoring Engine
Combines rule-based scoring with AI anomaly detection.
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    user_id: str
    ip: str
    city: str
    usual_city: str
    device: str
    usual_device: str
    is_unusual_time: bool
    failed_logins: int
    service: str
    is_sensitive_service: bool
    travel_distance_km: float


@dataclass
class ScoringResult:
    combined_risk: int
    rule_based_score: int
    ai_anomaly_score: int
    trust_level: Literal["High Trust", "Medium Trust", "Low Trust", "Critical"]
    recommended_action: str
    rule_breakdown: dict
    ai_breakdown: dict


class RuleBasedScorer:
    """Deterministic rule-based scoring engine."""

    # Rule weights (configurable)
    WEIGHTS = {
        "city_mismatch": 15,
        "device_mismatch": 10,
        "unusual_time": 10,
        "failed_logins": 8,  # per failed login, capped
        "sensitive_service": 12,
        "travel_distance": 25,  # scaled by distance
    }

    def score(self, session: SessionData) -> tuple[int, dict]:
        """Calculate rule-based risk score (0-100)."""
        breakdown = {}
        total = 0

        # City mismatch (case-insensitive comparison)
        if session.city.lower() != session.usual_city.lower():
            breakdown["city_mismatch"] = self.WEIGHTS["city_mismatch"]
            total += self.WEIGHTS["city_mismatch"]
        else:
            breakdown["city_mismatch"] = 0

        # Device mismatch (case-insensitive comparison)
        if session.device.lower() != session.usual_device.lower():
            breakdown["device_mismatch"] = self.WEIGHTS["device_mismatch"]
            total += self.WEIGHTS["device_mismatch"]
        else:
            breakdown["device_mismatch"] = 0

        # Unusual time
        if session.is_unusual_time:
            breakdown["unusual_time"] = self.WEIGHTS["unusual_time"]
            total += self.WEIGHTS["unusual_time"]
        else:
            breakdown["unusual_time"] = 0

        # Failed logins (max 3 counted)
        failed_score = min(session.failed_logins, 3) * self.WEIGHTS["failed_logins"]
        breakdown["failed_logins"] = failed_score
        total += failed_score

        # Sensitive service
        if session.is_sensitive_service:
            breakdown["sensitive_service"] = self.WEIGHTS["sensitive_service"]
            total += self.WEIGHTS["sensitive_service"]
        else:
            breakdown["sensitive_service"] = 0

        # Travel distance (impossible travel detection)
        if session.travel_distance_km > 0:
            # Scale: 500km+ gets full points, linear below
            distance_score = min(
                self.WEIGHTS["travel_distance"],
                int((session.travel_distance_km / 500) * self.WEIGHTS["travel_distance"])
            )
            breakdown["travel_distance"] = distance_score
            total += distance_score
        else:
            breakdown["travel_distance"] = 0

        # Cap at 100
        return min(total, 100), breakdown


class AIAnomalyScorer:
    """
    AI-based anomaly detection using IsolationForest.

    Uses a real ML model trained on behavioral patterns.
    Falls back to heuristics if model not available.
    """

    def __init__(self, model_path: str = None):
        """Initialize with optional pre-trained model."""
        self.ml_model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        """Load ML model, creating and training if needed."""
        try:
            from .ml_model import ZeroTrustMLModel, create_and_train_model

            if model_path:
                self.ml_model = ZeroTrustMLModel(model_path)
            else:
                # Create and train model with synthetic data
                self.ml_model = create_and_train_model()
        except Exception as e:
            logger.warning(f"Could not load ML model ({e}). Using heuristics.")
            self.ml_model = None

    def score(self, session: SessionData) -> tuple[int, dict]:
        """
        Calculate AI anomaly score (0-100).

        Returns:
            tuple: (score, breakdown_dict)
        """
        features = self._extract_features(session)

        if self.ml_model:
            # Use real ML model
            session_dict = {
                "city": session.city,
                "usual_city": session.usual_city,
                "device": session.device,
                "usual_device": session.usual_device,
                "is_unusual_time": session.is_unusual_time,
                "failed_logins": session.failed_logins,
                "is_sensitive_service": session.is_sensitive_service,
                "travel_distance_km": session.travel_distance_km,
            }

            prediction = self.ml_model.predict(session_dict)

            # Convert numpy types to native Python types for JSON serialization
            feature_contribs = {
                k: float(v) for k, v in prediction.feature_contributions.items()
            }

            breakdown = {
                "isolation_forest_score": int(prediction.isolation_forest_score),
                "deterministic_anomaly": 0,
                "combined_ml_score": int(prediction.anomaly_score),
                "is_anomaly": bool(prediction.is_anomaly),
                "confidence": float(round(prediction.confidence, 2)),
                "features_used": features,
                "feature_contributions": feature_contribs,
            }

            return int(prediction.anomaly_score), breakdown
        else:
            # Fallback: heuristic scoring
            return self._heuristic_score(features)

    def _heuristic_score(self, features: dict) -> tuple[int, dict]:
        """Fallback heuristic scoring when ML model unavailable."""
        anomaly_score = 0

        if features["city_mismatch"]:
            anomaly_score += 20
        if features["device_mismatch"]:
            anomaly_score += 15
        if features["is_unusual_time"]:
            anomaly_score += 15
        if features["failed_logins"] > 0:
            anomaly_score += min(features["failed_logins"] * 10, 25)
        if features["is_sensitive_service"]:
            anomaly_score += 10
        if features["travel_distance_km"] > 100:
            anomaly_score += min(int(features["travel_distance_km"] / 50), 30)

        breakdown = {
            "isolation_forest_score": anomaly_score,
            "deterministic_anomaly": 0,
            "combined_ml_score": anomaly_score,
            "features_used": features,
            "note": "Using heuristic fallback (ML model not loaded)",
        }

        return min(anomaly_score, 100), breakdown

    def _extract_features(self, session: SessionData) -> dict:
        """Extract numerical features for ML model."""
        return {
            "city_mismatch": 1 if session.city != session.usual_city else 0,
            "device_mismatch": 1 if session.device != session.usual_device else 0,
            "is_unusual_time": 1 if session.is_unusual_time else 0,
            "failed_logins": session.failed_logins,
            "is_sensitive_service": 1 if session.is_sensitive_service else 0,
            "travel_distance_km": session.travel_distance_km,
        }


class ZeroTrustScorer:
    """Main scoring engine combining rule-based and AI scoring."""

    def __init__(self, rule_weight: float = 0.4):
        """
        Initialize scorer.

        Args:
            rule_weight: Weight for rule-based score (0-1).
                        AI weight = 1 - rule_weight
        """
        self.rule_weight = rule_weight
        self.rule_scorer = RuleBasedScorer()
        self.ai_scorer = AIAnomalyScorer()

    def evaluate(self, session: SessionData) -> ScoringResult:
        """Evaluate session and return comprehensive risk assessment."""

        # Get individual scores
        rule_score, rule_breakdown = self.rule_scorer.score(session)
        ai_score, ai_breakdown = self.ai_scorer.score(session)

        # Combine scores
        combined = int(
            (rule_score * self.rule_weight) +
            (ai_score * (1 - self.rule_weight))
        )

        # Determine trust level
        trust_level = self._get_trust_level(combined)

        # Determine recommended action
        action = self._get_recommended_action(combined, rule_breakdown)

        return ScoringResult(
            combined_risk=combined,
            rule_based_score=rule_score,
            ai_anomaly_score=ai_score,
            trust_level=trust_level,
            recommended_action=action,
            rule_breakdown=rule_breakdown,
            ai_breakdown=ai_breakdown,
        )

    def _get_trust_level(self, score: int) -> str:
        """Map score to trust level."""
        if score <= 20:
            return "High Trust"
        elif score <= 50:
            return "Medium Trust"
        elif score <= 75:
            return "Low Trust"
        else:
            return "Critical"

    def _get_recommended_action(self, score: int, rule_breakdown: dict) -> str:
        """Determine recommended action based on score and rules triggered."""
        if score <= 15:
            return "Allow"
        elif score <= 30:
            return "Allow and monitor"
        elif score <= 50:
            return "Challenge with MFA"
        elif score <= 75:
            return "Restrict sensitive operations"
        else:
            return "Block and alert SOC"
