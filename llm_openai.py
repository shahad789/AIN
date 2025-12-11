import os
import json
from dotenv import load_dotenv

# Load the .env file automatically
load_dotenv()

# Try to import OpenAI (optional)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Read API key from .env (only if it exists)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client= None

# Optional API key from environment (ONLY for you if you set it)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None  # No real LLM → we will use local logic


SYSTEM_PROMPT = """
You are a senior SOC analyst in a Zero Trust environment.
You receive login session details and risk engine outputs.
Write a short, clear explanation for another analyst:
- explain why the risk is high, medium, or low
- reference specific signals (city/device mismatch, failed logins, travel distance, sensitive service, unusual time)
- end with 1–2 concrete recommendations (block, MFA, monitor).
Keep it short: 1–2 paragraphs. No bullet points.
"""


def _fallback_explanation(session: dict, result: dict) -> str:
    """
    Local explanation with NO external API.
    This is what runs for judges / anyone without a key.
    """

    risk = result.get("combined_risk", 0)
    rule_score = result.get("rule_based_score", 0)
    ml_score = result.get("ai_anomaly_score", 0)
    trust = result.get("trust_level", "Unknown")
    action = result.get("recommended_action", "No action")

    # Very rough banding
    if risk >= 80:
        severity = "critical"
    elif risk >= 60:
        severity = "high"
    elif risk >= 40:
        severity = "medium"
    else:
        severity = "low"

    reasons = []

    if session.get("city") and session.get("usual_city") and session.get("city") != session.get("usual_city"):
        reasons.append("login from a different city than usual")

    if session.get("device") and session.get("usual_device") and session.get("device") != session.get("usual_device"):
        reasons.append("new or unusual device for this user")

    if session.get("is_unusual_time"):
        reasons.append("activity at an unusual time")

    failed = session.get("failed_logins", 0) or 0
    if failed >= 5:
        reasons.append(f"{failed} failed login attempts (possible brute-force)")
    elif failed >= 1:
        reasons.append(f"{failed} failed login attempts before success")

    if session.get("is_sensitive_service"):
        reasons.append("accessing a sensitive / high-impact service")

    travel = session.get("travel_distance_km", 0) or 0
    if travel and travel > 200:
        reasons.append(f"large travel distance (~{int(travel)} km) suggesting impossible travel")

    if reasons:
        reasons_text = "; ".join(reasons)
    else:
        reasons_text = "no strong anomalies in city, device, time, or service."

    return (
        f"The engine rates this session as {severity} risk (score {risk}/100) "
        f"with trust level '{trust}'. The rule-based engine contributed {rule_score}/100, "
        f"and the ML anomaly model contributed {ml_score}/100.\n\n"
        f"Key signals observed: {reasons_text}. "
        f"Recommended action: {action}."
    )


def explain_session(session: dict, result: dict) -> str:
    """
    Main function used by the frontend.

    - If OpenAI + OPENAI_API_KEY are available → call real LLM.
    - If not → fallback to local explanation so it ALWAYS works.
    """

    # If no client configured, just use local "AI"
    if client is None:
        return _fallback_explanation(session, result)

    # Real OpenAI path (only for you if you export OPENAI_API_KEY)
    user_content = f"""
Session JSON:
{json.dumps(session, indent=2)}

Combined Risk Score: {result.get('combined_risk')}
Rule-based Score: {result.get('rule_based_score')}
AI Anomaly Score: {result.get('ai_anomaly_score')}
Trust Level: {result.get('trust_level')}
Recommended Action: {result.get('recommended_action')}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",  "content": user_content},
        ],
        temperature=0.25,
        max_tokens=350,
    )

    return completion.choices[0].message.content.strip()