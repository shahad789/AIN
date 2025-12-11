"""
AIn - FastAPI Backend
Provides REST API for session risk evaluation and log ingestion.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import json
import threading
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.scorer import SessionData, ZeroTrustScorer
from models.log_parser import LogParser, NormalizedSession
from data.synthetic_generator import SyntheticLogGenerator


app = FastAPI(
    title="AIn",
    description="Predictive security through behavior analysis",
    version="1.0.0",
)

# CORS for Streamlit frontend
# WARNING: allow_origins=["*"] is permissive and suitable for development/demo only.
# For production, restrict to specific origins e.g. ["http://localhost:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scorer and parser
scorer = ZeroTrustScorer(rule_weight=0.4)
scorer_lock = threading.Lock()  # Thread-safe access to scorer
log_parser = LogParser()
synthetic_generator = SyntheticLogGenerator(seed=42)


class SessionInput(BaseModel):
    """Input schema for session evaluation."""
    user_id: str = Field(..., description="User identifier")
    ip: str = Field(..., description="IP address")
    city: str = Field(..., description="Current city")
    usual_city: str = Field(..., description="User's usual city")
    device: str = Field(..., description="Current device")
    usual_device: str = Field(..., description="User's usual device")
    is_unusual_time: bool = Field(False, description="Login at unusual time")
    failed_logins: int = Field(0, ge=0, description="Recent failed login attempts")
    service: str = Field(..., description="Service being accessed")
    is_sensitive_service: bool = Field(False, description="Is this a sensitive service")
    travel_distance_km: float = Field(0, ge=0, description="Distance from usual location")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "Fahad123",
                "ip": "8.8.8.8",
                "city": "Riyadh",
                "usual_city": "Riyadh",
                "device": "iPhone 13",
                "usual_device": "iPhone 13",
                "is_unusual_time": False,
                "failed_logins": 0,
                "service": "View Profile",
                "is_sensitive_service": False,
                "travel_distance_km": 0
            }
        }


class EvaluationResponse(BaseModel):
    """Response schema for session evaluation."""
    combined_risk: int = Field(..., ge=0, le=100)
    rule_based_score: int = Field(..., ge=0, le=100)
    ai_anomaly_score: int = Field(..., ge=0, le=100)
    trust_level: str
    recommended_action: str
    rule_breakdown: dict
    ai_breakdown: dict


class ConfigInput(BaseModel):
    """Configuration for scoring weights."""
    rule_weight: float = Field(0.4, ge=0, le=1, description="Weight for rule-based scoring")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "service": "ZeroTrust AI"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_session(session: SessionInput):
    """
    Evaluate a user session and return risk assessment.

    Returns combined risk score, individual scores, trust level,
    and recommended action.
    """
    try:
        session_data = SessionData(
            user_id=session.user_id,
            ip=session.ip,
            city=session.city,
            usual_city=session.usual_city,
            device=session.device,
            usual_device=session.usual_device,
            is_unusual_time=session.is_unusual_time,
            failed_logins=session.failed_logins,
            service=session.service,
            is_sensitive_service=session.is_sensitive_service,
            travel_distance_km=session.travel_distance_km,
        )

        result = scorer.evaluate(session_data)

        return EvaluationResponse(
            combined_risk=result.combined_risk,
            rule_based_score=result.rule_based_score,
            ai_anomaly_score=result.ai_anomaly_score,
            trust_level=result.trust_level,
            recommended_action=result.recommended_action,
            rule_breakdown=result.rule_breakdown,
            ai_breakdown=result.ai_breakdown,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
async def update_config(config: ConfigInput):
    """Update scoring configuration."""
    global scorer
    with scorer_lock:
        scorer = ZeroTrustScorer(rule_weight=config.rule_weight)
    return {"status": "updated", "rule_weight": config.rule_weight}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "scorer_initialized": scorer is not None,
        "rule_weight": scorer.rule_weight,
        "ml_model_loaded": scorer.ai_scorer.ml_model is not None,
    }


# ============== Log Ingestion Endpoints ==============

class LogIngestionRequest(BaseModel):
    """Request for log ingestion."""
    logs: str = Field(..., description="Raw log data")
    format: Optional[str] = Field("auto", description="Log format: auto, json, jsonl, syslog, apache, csv, elk")
    user_id: Optional[str] = Field(None, description="User ID to analyze (if not in logs)")


class LogIngestionResponse(BaseModel):
    """Response from log ingestion."""
    entries_parsed: int
    sessions_analyzed: int
    evaluations: List[dict]


class BulkEvaluationRequest(BaseModel):
    """Request for bulk session evaluation."""
    sessions: List[SessionInput]


class GenerateSyntheticRequest(BaseModel):
    """Request for synthetic data generation."""
    num_normal: int = Field(10, ge=1, le=100)
    num_impossible_travel: int = Field(2, ge=0, le=20)
    num_credential_stuffing: int = Field(1, ge=0, le=10)
    num_account_takeover: int = Field(1, ge=0, le=10)


@app.post("/ingest/logs", response_model=LogIngestionResponse)
async def ingest_logs(request: LogIngestionRequest):
    """
    Ingest raw logs, parse them, and evaluate sessions.

    Supports multiple log formats:
    - JSON array
    - JSONL (one JSON per line)
    - Syslog format
    - Apache/Nginx combined log
    - CSV
    - ELK/Elasticsearch format

    Returns evaluation results for each unique user session.
    """
    try:
        # Parse logs
        entries = log_parser.parse(request.logs, request.format)

        if not entries:
            return LogIngestionResponse(
                entries_parsed=0,
                sessions_analyzed=0,
                evaluations=[],
            )

        # Group by user
        users = {}
        for entry in entries:
            user_id = entry.user_id or request.user_id or "unknown"
            if user_id not in users:
                users[user_id] = []
            users[user_id].append(entry)

        # Evaluate each user's session
        evaluations = []
        for user_id, user_entries in users.items():
            try:
                # Normalize to session
                normalized = log_parser.normalize_to_session(user_entries, user_id)

                # Convert to SessionData
                session_data = SessionData(
                    user_id=normalized.user_id,
                    ip=normalized.ip,
                    city=normalized.city,
                    usual_city=normalized.usual_city,
                    device=normalized.device,
                    usual_device=normalized.usual_device,
                    is_unusual_time=normalized.is_unusual_time,
                    failed_logins=normalized.failed_logins,
                    service=normalized.service,
                    is_sensitive_service=normalized.is_sensitive_service,
                    travel_distance_km=normalized.travel_distance_km,
                )

                # Evaluate
                result = scorer.evaluate(session_data)

                evaluations.append({
                    "user_id": user_id,
                    "session": normalized.to_dict(),
                    "combined_risk": result.combined_risk,
                    "rule_based_score": result.rule_based_score,
                    "ai_anomaly_score": result.ai_anomaly_score,
                    "trust_level": result.trust_level,
                    "recommended_action": result.recommended_action,
                })
            except Exception as e:
                evaluations.append({
                    "user_id": user_id,
                    "error": str(e),
                })

        return LogIngestionResponse(
            entries_parsed=len(entries),
            sessions_analyzed=len(users),
            evaluations=evaluations,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Log parsing error: {str(e)}")


@app.post("/ingest/file")
async def ingest_log_file(
    file: UploadFile = File(...),
    format: Optional[str] = "auto",
):
    """
    Upload and process a log file.

    Accepts .json, .jsonl, .csv, .log, .txt files.
    """
    try:
        content = await file.read()
        log_data = content.decode("utf-8")

        # Auto-detect format from extension if not specified
        if format == "auto" and file.filename:
            ext = file.filename.split(".")[-1].lower()
            format_map = {
                "json": "json",
                "jsonl": "jsonl",
                "csv": "csv",
                "log": "syslog",
                "txt": "auto",
            }
            format = format_map.get(ext, "auto")

        # Use the log ingestion endpoint
        request = LogIngestionRequest(logs=log_data, format=format)
        return await ingest_logs(request)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")


@app.post("/evaluate/bulk")
async def evaluate_bulk(request: BulkEvaluationRequest):
    """
    Evaluate multiple sessions at once.

    Useful for batch processing.
    """
    results = []

    for session in request.sessions:
        try:
            session_data = SessionData(
                user_id=session.user_id,
                ip=session.ip,
                city=session.city,
                usual_city=session.usual_city,
                device=session.device,
                usual_device=session.usual_device,
                is_unusual_time=session.is_unusual_time,
                failed_logins=session.failed_logins,
                service=session.service,
                is_sensitive_service=session.is_sensitive_service,
                travel_distance_km=session.travel_distance_km,
            )

            result = scorer.evaluate(session_data)
            results.append({
                "user_id": session.user_id,
                "combined_risk": result.combined_risk,
                "trust_level": result.trust_level,
                "recommended_action": result.recommended_action,
            })
        except Exception as e:
            results.append({
                "user_id": session.user_id,
                "error": str(e),
            })

    return {"evaluations": results, "total": len(results)}


@app.post("/generate/synthetic")
async def generate_synthetic_logs(request: GenerateSyntheticRequest):
    """
    Generate synthetic Absher-style logs for testing.

    Returns logs in JSONL format with various attack scenarios.
    """
    entries = synthetic_generator.generate_mixed_dataset(
        num_normal=request.num_normal,
        num_impossible_travel=request.num_impossible_travel,
        num_credential_stuffing=request.num_credential_stuffing,
        num_account_takeover=request.num_account_takeover,
    )

    # Convert to JSON-serializable format
    from dataclasses import asdict
    logs = [asdict(e) for e in entries]

    return {
        "total_entries": len(logs),
        "breakdown": {
            "normal": request.num_normal,
            "impossible_travel": request.num_impossible_travel,
            "credential_stuffing": request.num_credential_stuffing,
            "account_takeover": request.num_account_takeover,
        },
        "logs": logs,
    }


@app.get("/demo/scenarios")
async def get_demo_scenarios():
    """
    Get pre-built demo scenarios for testing.

    Returns sample sessions for different attack types.
    """
    scenarios = {
        "normal_safe_session": {
            "user_id": "Lama22",
            "ip": "212.138.45.12",
            "city": "Riyadh",
            "usual_city": "Riyadh",
            "device": "iPhone 14",
            "usual_device": "iPhone 14",
            "is_unusual_time": False,
            "failed_logins": 0,
            "service": "View Profile",
            "is_sensitive_service": False,
            "travel_distance_km": 2,
        },
        "impossible_travel": {
            "user_id": "Fahad123",
            "ip": "197.32.45.100",
            "city": "Cairo",
            "usual_city": "Riyadh",
            "device": "iPhone 13",
            "usual_device": "iPhone 13",
            "is_unusual_time": False,
            "failed_logins": 1,
            "service": "View Profile",
            "is_sensitive_service": False,
            "travel_distance_km": 1600,
        },
        "suspicious_login": {
            "user_id": "Ahmed99",
            "ip": "185.220.100.252",
            "city": "Unknown",
            "usual_city": "Jeddah",
            "device": "Windows PC",
            "usual_device": "Samsung Galaxy S22",
            "is_unusual_time": True,
            "failed_logins": 3,
            "service": "Change Password",
            "is_sensitive_service": True,
            "travel_distance_km": 500,
        },
        "account_takeover_attempt": {
            "user_id": "Sara_M",
            "ip": "95.173.200.100",
            "city": "Moscow",
            "usual_city": "Dammam",
            "device": "Windows PC",
            "usual_device": "iPhone 12",
            "is_unusual_time": True,
            "failed_logins": 5,
            "service": "Update Bank Account",
            "is_sensitive_service": True,
            "travel_distance_km": 5000,
        },
    }

    return scenarios


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
