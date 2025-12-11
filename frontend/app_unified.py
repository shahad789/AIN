"""
AIn - Unified Frontend with Style Selector
All 4 styles in one app with dropdown to switch between them.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import requests
import json
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from llm_openai import explain_session


try:
    from components import (
        create_geolocation_map,
        create_realtime_feed,
        generate_demo_sessions,
        display_threat_stats,
    )
except ImportError as e:
    st.error(f"Failed to import components: {e}. Ensure components.py is in the same directory.")
    st.stop()

# Configuration
API_URL = "http://localhost:8000"
API_TIMEOUT = 10  # seconds

# Logo paths for favicon and header
import os
LOGO_PATH = os.path.join(os.path.dirname(__file__), "ain-logo.png")
LOGO_WHITE_PATH = os.path.join(os.path.dirname(__file__), "ain-logo-white.png")

# Must be first Streamlit command
st.set_page_config(
    page_title="AIn",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# STYLE DEFINITIONS
# ============================================================================

STYLES = {
    "Claymorphism": {
        "css": """
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
        /* === Force light mode for this theme - override Streamlit dark theme === */
        :root, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
            color-scheme: light !important;
            background: linear-gradient(145deg, #e8e4e1 0%, #d4cfc9 100%) !important;
            font-family: 'Nunito', sans-serif;
        }
        /* Force all text dark (override Streamlit dark theme) */
        .stApp *, .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown, .stMarkdown p {
            color: #2d2520 !important;
        }
        /* Our custom header */
        .main-header {
            background: linear-gradient(145deg, #f0ebe6, #d9d4cf);
            padding: 2rem; border-radius: 30px; margin-bottom: 2rem;
            text-align: center;
            box-shadow: 20px 20px 60px #c5c1bc, -20px -20px 60px #ffffff;
        }
        .header-logo { max-width: 240px; height: auto; margin-bottom: 0.5rem; }
        .main-title { display: none; }
        .subtitle { color: #6b5d52 !important; font-size: 0.85rem; margin-top: 0; letter-spacing: 1px; text-transform: uppercase; }
        .section-header { color: #2d2520 !important; font-size: 1.1rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; }
        /* Sidebar */
        .stSidebar, .stSidebar [data-testid="stSidebarContent"], [data-testid="stSidebar"] {
            background: linear-gradient(145deg, #e8e4e1, #d4cfc9) !important;
        }
        .stSidebar p, .stSidebar label, .stSidebar h1, .stSidebar h2, .stSidebar h3 { color: #2d2520 !important; }
        /* Dropdown - neutral gray works on both light/dark */
        .stSelectbox [data-baseweb="select"] * { color: #555555 !important; }
        .stSelectbox [data-baseweb="select"] svg { fill: #555555 !important; }
        /* Metrics */
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #2d2520 !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #4a3f37 !important; }
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color: #333333 !important; }
        /* Form elements - high specificity */
        .stSelectbox label, .stTextArea label, .stSlider label { color: #2d2520 !important; }
        /* Dropdown menu - high specificity */
        .stApp [data-baseweb="popover"], [data-baseweb="popover"] { background-color: #f0ebe6 !important; border: 1px solid #8b7355 !important; }
        .stApp [data-baseweb="menu"], [data-baseweb="menu"] { background-color: #f0ebe6 !important; }
        .stApp [data-baseweb="menu"] li, [data-baseweb="menu"] li { color: #2d2520 !important; background-color: #f0ebe6 !important; }
        .stApp [data-baseweb="menu"] li:hover { background-color: #e8e4e1 !important; }
        .stApp [role="listbox"], [role="listbox"] { background-color: #f0ebe6 !important; }
        .stApp [role="option"], [role="option"] { color: #2d2520 !important; background-color: #f0ebe6 !important; }
        .stApp [role="option"]:hover { background-color: #e8e4e1 !important; }
        /* Text area */
        .stApp .stTextArea textarea { color: #2d2520 !important; background-color: #faf8f5 !important; border: 1px solid #8b7355 !important; }
        /* Expander - dark bg, white text for visibility */
        .stApp [data-testid="stExpander"] { background-color: #5c4a3a !important; border-radius: 20px; }
        .stApp [data-testid="stExpander"] summary { background-color: #5c4a3a !important; }
        .stApp [data-testid="stExpander"] summary span { color: #ffffff !important; }
        .stApp [data-testid="stExpander"] summary:hover span { color: #c9a87c !important; }
        .stApp [data-testid="stExpander"] svg { fill: #ffffff !important; stroke: #ffffff !important; }
        .stApp .streamlit-expanderContent { background-color: #4a3d30 !important; }
        .stApp .streamlit-expanderContent, .stApp .streamlit-expanderContent * { color: #ffffff !important; }
        /* JSON display - dark bg, light text */
        .stApp [data-testid="stJson"] { background-color: #3d3228 !important; padding: 1rem; border-radius: 15px; }
        .stApp [data-testid="stJson"] *, [data-testid="stJson"] * { color: #e8dfd5 !important; }
        .stApp pre, .stApp code { color: #e8dfd5 !important; background-color: #3d3228 !important; }
        /* Button */
        .stApp .stButton button { background-color: #8b7355 !important; color: #f5f5f0 !important; border: none !important; border-radius: 20px !important; }
        .stApp .stButton button:hover { background-color: #a67c52 !important; }
        """,
        "theme": "light",
        "header_title": "AIn",
        "header_subtitle": "Intelligent Session Risk Assessment",
        "colors": {"primary": "#8b7355", "secondary": "#a67c52", "accent": "#c9a87c"},
    },
    "Saudi": {
        "css": """
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
        .stApp { background-color: #006633; font-family: 'Tajawal', sans-serif; }
        .main-header {
            background: linear-gradient(135deg, #004d26 0%, #006633 50%, #008844 100%);
            border: 3px solid #c9a227; padding: 1.5rem 2rem; border-radius: 0;
            text-align: center;
        }
        .header-logo { max-width: 240px; height: auto; margin-bottom: 0.5rem; }
        .main-title { display: none; }
        .subtitle { color: rgba(245, 245, 220, 0.8) !important; font-size: 0.85rem; margin-top: 0; letter-spacing: 2px; text-transform: uppercase; font-family: 'Tajawal', sans-serif; }
        .section-header { color: #c9a227 !important; font-size: 1.1rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #c9a227; font-family: 'Tajawal', sans-serif; }
        [data-testid="stMetricValue"] { color: #c9a227 !important; font-family: 'Tajawal', sans-serif !important; }
        [data-testid="stMetricLabel"] { color: #f5f5dc !important; font-family: 'Tajawal', sans-serif !important; }
        .stButton > button { background: linear-gradient(135deg, #c9a227 0%, #a88b1f 100%); color: #004d26; border: none; padding: 0.75rem 2rem; font-weight: 700; border-radius: 0; font-family: 'Tajawal', sans-serif; }
        .stAlert { background-color: rgba(201, 162, 39, 0.2); border-left: 4px solid #c9a227; }
        .stAlert p { color: #f5f5dc !important; }
        .stTextArea textarea { color: #f5f5dc !important; background-color: rgba(0,0,0,0.2) !important; border: 1px solid #c9a227; }
        .stSelectbox [data-baseweb="select"] { background-color: rgba(0,0,0,0.3) !important; border: 1px solid #c9a227 !important; border-radius: 0 !important; }
        .stSelectbox [data-baseweb="select"] > div { border-radius: 0 !important; }
        .stSelectbox [data-baseweb="select"] * { color: #f5f5dc !important; }
        .stSelectbox [data-baseweb="select"] svg { fill: #f5f5dc !important; }
        .stSelectbox div[data-baseweb="popover"] { border-radius: 0 !important; background-color: #004d26 !important; border: 1px solid #c9a227 !important; }
        .stSelectbox ul { border-radius: 0 !important; background-color: #004d26 !important; }
        .stSelectbox li { color: #f5f5dc !important; }
        .stSelectbox li:hover { background-color: #006633 !important; }
        .stMarkdown p, .stMarkdown span, .stMarkdown li { color: #f5f5dc !important; }
        .stSidebar, .stSidebar [data-testid="stSidebarContent"] { background-color: #004d26 !important; }
        .stSidebar p, .stSidebar span, .stSidebar label { color: #f5f5dc !important; }
        """,
        "theme": "dark",
        "header_title": "AIn",
        "header_subtitle": "Absher Security Platform",
        "colors": {"primary": "#006633", "secondary": "#c9a227", "accent": "#f5f5dc"},
    },
    "Neon Cyberpunk": {
        "css": """
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');
        .stApp {
            background-color: #000000;
            background-image: linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
        }
        .stApp::after {
            content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.15), rgba(0, 0, 0, 0.15) 1px, transparent 1px, transparent 2px);
            pointer-events: none; z-index: 1000;
        }
        .main-header {
            background: linear-gradient(180deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 0, 0, 0.9) 100%);
            border: 2px solid #00ffff; padding: 1.5rem; margin-bottom: 2rem;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), inset 0 0 20px rgba(0, 255, 255, 0.1);
        }
        .header-logo { max-width: 240px; height: auto; margin-bottom: 0.5rem; }
        .main-title { display: none; }
        .subtitle { color: #ff00ff !important; font-size: 0.8rem; font-family: 'Share Tech Mono', monospace; margin-top: 0; letter-spacing: 2px; }
        .section-header { color: #00ffff !important; font-size: 1rem; font-family: 'Share Tech Mono', monospace; text-transform: uppercase; letter-spacing: 2px; margin-top: 1.5rem; margin-bottom: 1rem; padding-left: 10px; border-left: 3px solid #ff00ff; text-shadow: 0 0 5px #00ffff; }
        [data-testid="stMetricValue"] { color: #00ff00 !important; font-family: 'Orbitron', sans-serif !important; text-shadow: 0 0 10px #00ff00; }
        [data-testid="stMetricLabel"] { color: #00ffff !important; font-family: 'Share Tech Mono', monospace !important; text-transform: uppercase; }
        .stButton > button { background: transparent; color: #00ffff; border: 2px solid #00ffff; padding: 0.75rem 2rem; font-family: 'Share Tech Mono', monospace; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.3); transition: all 0.2s ease; }
        .stButton > button:hover { background: #00ffff; color: #000000 !important; box-shadow: 0 0 20px rgba(0, 255, 255, 0.8); }
        .stButton > button:hover span, .stButton > button:hover p { color: #000000 !important; }
        .stAlert { background-color: rgba(255, 0, 255, 0.1); border: 1px solid #ff00ff; }
        .stAlert p { color: #ff00ff !important; font-family: 'Share Tech Mono', monospace; }
        .stTextArea textarea { color: #00ff00 !important; background-color: #0a0a0a !important; border: 1px solid #00ff00; font-family: 'Share Tech Mono', monospace; }
        .stSelectbox [data-baseweb="select"] { background-color: #0a0a0a !important; border: 1px solid #00ffff !important; border-radius: 0 !important; }
        .stSelectbox [data-baseweb="select"] > div { border-radius: 0 !important; }
        .stSelectbox [data-baseweb="select"] * { color: #00ffff !important; }
        .stSelectbox [data-baseweb="select"] svg { fill: #00ffff !important; }
        .stSelectbox div[data-baseweb="popover"] { border-radius: 0 !important; border: 1px solid #00ffff !important; background-color: #0a0a0a !important; }
        .stSelectbox ul { border-radius: 0 !important; background-color: #0a0a0a !important; }
        .stSelectbox li { color: #00ffff !important; }
        p, span, label, .stMarkdown { color: #00ffff !important; font-family: 'Share Tech Mono', monospace; }
        """,
        "theme": "dark",
        "header_title": "AIn",
        "header_subtitle": ">> SECURITY_TERMINAL v2.0 | SESSION_RISK_ANALYZER",
        "colors": {"primary": "#00ffff", "secondary": "#ff00ff", "accent": "#00ff00"},
    },
    "Neo-Brutalism": {
        "css": """
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        /* === Force light mode for this theme - override Streamlit dark theme === */
        :root, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
            color-scheme: light !important;
            background-color: #f5f5dc !important;
            font-family: 'Space Grotesk', sans-serif;
        }
        /* Our custom header - brutalist style */
        .main-header {
            background: #006633;
            padding: 2rem; border-radius: 0; margin-bottom: 2rem;
            text-align: center;
            border: 4px solid #000000;
            box-shadow: 8px 8px 0px #000000;
        }
        .header-logo { max-width: 240px; height: auto; margin-bottom: 0.5rem; }
        .main-title { display: none; }
        .subtitle { color: #f5f5dc !important; font-size: 0.9rem; margin-top: 0; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
        .section-header {
            color: #004d26 !important; font-size: 1.2rem; font-weight: 700;
            margin-top: 1.5rem; margin-bottom: 1rem;
            text-transform: uppercase; border-bottom: 3px solid #004d26; padding-bottom: 0.5rem;
        }
        /* Sidebar */
        .stSidebar, .stSidebar [data-testid="stSidebarContent"], [data-testid="stSidebar"] {
            background: #e8e4d9 !important;
        }
        .stSidebar p, .stSidebar label, .stSidebar h1, .stSidebar h2, .stSidebar h3 { color: #1a1a1a !important; }
        /* Dropdown - neutral gray works on both light/dark */
        .stSelectbox [data-baseweb="select"] * { color: #555555 !important; }
        .stSelectbox [data-baseweb="select"] svg { fill: #555555 !important; }
        /* Metrics */
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #1a1a1a !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #004d26 !important; }
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color: #333333 !important; }
        /* Form elements - high specificity */
        .stSelectbox label, .stTextArea label, .stSlider label { color: #1a1a1a !important; }
        /* Text area */
        .stApp .stTextArea textarea { color: #1a1a1a !important; background-color: #faf8f5 !important; border: 1px solid #004d26 !important; }
        /* Expander - dark bg, white text for visibility */
        .stApp [data-testid="stExpander"] { background-color: #004d26 !important; border-radius: 4px; }
        .stApp [data-testid="stExpander"] summary { background-color: #004d26 !important; }
        .stApp [data-testid="stExpander"] summary span { color: #ffffff !important; }
        .stApp [data-testid="stExpander"] summary:hover span { color: #c9a227 !important; }
        .stApp [data-testid="stExpander"] svg { fill: #ffffff !important; stroke: #ffffff !important; }
        .stApp .streamlit-expanderContent { background-color: #003d1f !important; }
        .stApp .streamlit-expanderContent, .stApp .streamlit-expanderContent * { color: #ffffff !important; }
        /* JSON display - dark bg, white text */
        .stApp [data-testid="stJson"] { background-color: #002211 !important; padding: 1rem; border-radius: 4px; }
        .stApp [data-testid="stJson"] *, [data-testid="stJson"] * { color: #00ff88 !important; }
        .stApp pre, .stApp code { color: #00ff88 !important; background-color: #002211 !important; }
        /* Button */
        .stApp .stButton button { background-color: #006633 !important; color: #f5f5dc !important; border: 2px solid #000 !important; }
        .stApp .stButton button:hover { background-color: #004d26 !important; }
        """,
        "theme": "light",
        "header_title": "AIn",
        "header_subtitle": "Session Risk Assessment",
        "colors": {"primary": "#006633", "secondary": "#c9a227", "accent": "#f5f5dc"},
    },
}

# ============================================================================
# SCENARIOS
# ============================================================================

SCENARIOS = {
    "Select a scenario...": None,
    "Normal Safe Session": {
        "user_id": "Lama22",
        "ip": "10.0.0.1",
        "city": "Riyadh",
        "usual_city": "Riyadh",
        "device": "iPhone 14",
        "usual_device": "iPhone 14",
        "is_unusual_time": False,
        "failed_logins": 0,
        "service": "View Profile",
        "is_sensitive_service": False,
        "travel_distance_km": 2
    },
    "Impossible Travel Attack": {
        "user_id": "Fahad123",
        "ip": "8.8.8.8",
        "city": "Cairo",
        "usual_city": "Riyadh",
        "device": "iPhone 13",
        "usual_device": "iPhone 13",
        "is_unusual_time": False,
        "failed_logins": 1,
        "service": "View Profile",
        "is_sensitive_service": False,
        "travel_distance_km": 1600
    },
    "Suspicious Login": {
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
        "travel_distance_km": 500
    },
    "Account Takeover (Critical)": {
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
        "travel_distance_km": 5000
    },
}

# ============================================================================
# API CALLS
# ============================================================================

def call_api(session_data: dict, rule_weight: float) -> dict:
    """Call the FastAPI backend."""
    try:
        requests.post(f"{API_URL}/config", json={"rule_weight": rule_weight}, timeout=API_TIMEOUT)
        response = requests.post(f"{API_URL}/evaluate", json=session_data, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend connection failed. Ensure the backend is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Backend request timed out. The server may be overloaded.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_trust_color(trust_level: str, style: dict) -> str:
    """Get color based on trust level."""
    if trust_level == "High Trust":
        return "#00aa00"
    elif trust_level == "Medium Trust":
        return "#ffaa00"
    elif trust_level == "Low Trust":
        return "#ff6600"
    else:  # Critical
        return "#ff0000"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar for style selection
    style_options = ["Claymorphism", "Saudi", "Neon Cyberpunk", "Neo-Brutalism"]
    default_style = "Neo-Brutalism"

    with st.sidebar:
        st.markdown("### Theme Settings")
        selected_style = st.selectbox(
            "Select UI Style",
            style_options,
            index=style_options.index(default_style),
            key="style_selector"
        )

    # Get current style config
    style = STYLES[selected_style]

    # Inject CSS
    st.markdown(f"<style>{style['css']}</style>", unsafe_allow_html=True)

    # Load logo as base64 for embedding in HTML
    # Green logo for Claymorphism, white logo for all others
    logo_html = ""
    logo_file = LOGO_PATH if selected_style == "Claymorphism" else LOGO_WHITE_PATH
    if os.path.exists(logo_file):
        with open(logo_file, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" style="max-width: 240px; height: auto;" alt="AIn Logo">'

    # Header
    st.markdown(f"""
    <div class="main-header">
        {logo_html}
        <p class="main-title">{style['header_title']}</p>
        <p class="subtitle">{style['header_subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)

        scenario = st.selectbox(
            "Test Scenario",
            list(SCENARIOS.keys()),
            help="Choose a pre-defined scenario or edit the JSON below"
        )

        if scenario != "Select a scenario..." and SCENARIOS[scenario]:
            default_json = json.dumps(SCENARIOS[scenario], indent=2)
        else:
            default_json = json.dumps(list(SCENARIOS.values())[1], indent=2)

        session_json = st.text_area(
            "Session Data (JSON)",
            value=default_json,
            height=200,
            help="Edit the session data to test different scenarios"
        )

        rule_weight = st.slider(
            "Rule vs ML Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Balance between rule-based scoring and ML anomaly detection"
        )
        st.caption(f"Rules: {rule_weight:.0%} | ML: {1-rule_weight:.0%}")

        st.markdown("")
        evaluate_btn = st.button("Evaluate Session", use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

        if evaluate_btn:
            try:
                session_data = json.loads(session_json)
                with st.spinner("Analyzing..."):
                    result = call_api(session_data, rule_weight)
                if result:
                    st.session_state.result = result
                    st.session_state.session_data = session_data
            except json.JSONDecodeError:
                st.error("Invalid JSON format")

        if "result" in st.session_state:
            result = st.session_state.result
            session_data = st.session_state.session_data

            # Metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Risk Score", result["combined_risk"])
            with metric_cols[1]:
                st.metric("Rule Score", result["rule_based_score"])
            with metric_cols[2]:
                st.metric("AI Score", result["ai_anomaly_score"])
            with metric_cols[3]:
                st.metric("Trust Level", result["trust_level"])

            # Recommendation with dynamic colors
            st.markdown('<p class="section-header">Recommended Action</p>', unsafe_allow_html=True)
            action = result["recommended_action"]
            action_lower = action.lower()

            # Determine color based on action severity
            if "block" in action_lower or "alert soc" in action_lower:
                # Critical - red tones
                if style["theme"] == "dark":
                    bg_color = "rgba(220, 53, 69, 0.3)"
                    border_color = "#dc3545"
                    text_color = "#ff6b6b"
                else:
                    bg_color = "#f8d7da"
                    border_color = "#dc3545"
                    text_color = "#721c24"
            elif "deny" in action_lower or "require" in action_lower:
                # High - orange tones
                if style["theme"] == "dark":
                    bg_color = "rgba(255, 152, 0, 0.3)"
                    border_color = "#ff9800"
                    text_color = "#ffb74d"
                else:
                    bg_color = "#fff3cd"
                    border_color = "#ff9800"
                    text_color = "#856404"
            elif "monitor" in action_lower or "flag" in action_lower:
                # Medium - yellow tones
                if style["theme"] == "dark":
                    bg_color = "rgba(255, 193, 7, 0.2)"
                    border_color = "#ffc107"
                    text_color = "#ffd54f"
                else:
                    bg_color = "#fff9e6"
                    border_color = "#ffc107"
                    text_color = "#856404"
            else:
                # Safe - green tones
                if style["theme"] == "dark":
                    bg_color = "rgba(40, 167, 69, 0.3)"
                    border_color = "#28a745"
                    text_color = "#81c784"
                else:
                    bg_color = "#d4edda"
                    border_color = "#28a745"
                    text_color = "#155724"

            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {border_color};
                padding: 1rem 1.5rem;
                border-radius: 4px;
                margin: 0.5rem 0;
            ">
                <p style="color: {text_color}; font-weight: 600; margin: 0; font-size: 1.1rem;">
                    {action}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Geolocation Map
            st.markdown('<p class="section-header">Location Analysis</p>', unsafe_allow_html=True)
            geo_fig = create_geolocation_map(
                usual_city=session_data.get("usual_city", "Riyadh"),
                current_city=session_data.get("city", "Riyadh"),
                risk_score=result["combined_risk"],
                travel_distance_km=session_data.get("travel_distance_km", 0),
                theme=style["theme"]
            )
            st.plotly_chart(geo_fig, use_container_width=True)

            # Score chart
            st.markdown('<p class="section-header">Score Breakdown</p>', unsafe_allow_html=True)
            scores_df = pd.DataFrame({
                "Component": ["Rule Engine", "AI Model", "Combined"],
                "Score": [result["rule_based_score"], result["ai_anomaly_score"], result["combined_risk"]]
            })

            chart_colors = [style["colors"]["primary"], style["colors"]["secondary"], style["colors"]["accent"]]
            # Determine chart text color based on theme
            chart_text_color = "#ffffff" if style["theme"] == "dark" else "#1a1a1a"
            chart_grid_color = "rgba(255,255,255,0.1)" if style["theme"] == "dark" else "rgba(0,0,0,0.15)"

            fig = px.bar(
                scores_df, y="Component", x="Score", orientation="h",
                color="Component",
                color_discrete_sequence=chart_colors
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_text_color, size=13),
                xaxis=dict(
                    range=[0, 100],
                    gridcolor=chart_grid_color,
                    tickfont=dict(color=chart_text_color, size=12),
                    title=dict(font=dict(color=chart_text_color))
                ),
                yaxis=dict(
                    tickfont=dict(color=chart_text_color, size=12),
                    title=dict(font=dict(color=chart_text_color))
                ),
                height=180,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            fig.update_traces(textfont=dict(color=chart_text_color))
            st.plotly_chart(fig, use_container_width=True)

            # Detailed breakdown
            with st.expander("Detailed Analysis"):
                breakdown_cols = st.columns(2)
                with breakdown_cols[0]:
                    st.markdown("**Rule Analysis**")
                    for rule, score in result["rule_breakdown"].items():
                        st.write(f"• {rule.replace('_', ' ').title()}: {score}")
                with breakdown_cols[1]:
                    st.markdown("**ML Analysis**")
                    ai_breakdown = result["ai_breakdown"]
                    st.write(f"• Combined: {ai_breakdown.get('combined_ml_score', 0)}")
                    st.write(f"• IsolationForest: {ai_breakdown.get('isolation_forest_score', 0)}")
                    st.write(f"• Deterministic: {ai_breakdown.get('deterministic_anomaly', 0)}")

                    # 🔹 AI Explanation 
            with st.expander("AI Security Explanation (OpenAI)"):
                if st.button("Generate AI Explanation", key="openai_explain_button"):
                    with st.spinner("Generating explanation with OpenAI..."):
                        try:
                            explanation = explain_session(session_data, result)
                            st.write(explanation)
                        except Exception as e:
                            st.error(f"OpenAI error: {e}")

            with st.expander("Raw Session Data"):
                st.json(session_data)

        else:
            # Theme-aware empty state
            if style["theme"] == "dark":
                empty_border = "rgba(255,255,255,0.3)"
                empty_text = "rgba(255,255,255,0.9)"
                empty_muted = "rgba(255,255,255,0.6)"
            else:
                empty_border = "rgba(0,0,0,0.3)"
                empty_text = "#212529"
                empty_muted = "#666666"
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem; border: 1px dashed {empty_border}; border-radius: 12px;">
                <p style="font-size: 1.2rem; color: {empty_text};">Select a scenario and click Evaluate</p>
                <p style="color: {empty_muted};">to see risk assessment results</p>
            </div>
            """, unsafe_allow_html=True)

    # Real-time Feed Section
    st.markdown("---")
    st.markdown('<p class="section-header">Live Session Feed</p>', unsafe_allow_html=True)

    if "feed_sessions" not in st.session_state:
        st.session_state.feed_sessions = generate_demo_sessions(15)

    feed_cols = st.columns([1, 4])
    with feed_cols[0]:
        if st.button("Refresh Feed", use_container_width=True):
            st.session_state.feed_sessions = generate_demo_sessions(15)
            st.rerun()

    display_threat_stats(st.session_state.feed_sessions, theme=style["theme"])
    create_realtime_feed(st.session_state.feed_sessions, theme=style["theme"], max_display=8)


if __name__ == "__main__":
    main()
