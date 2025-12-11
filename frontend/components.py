"""
AIn - Additional UI Components
- Geolocation Map with impossible travel visualization
- Real-time session feed
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# City coordinates (lat, lon) for map visualization
# Comprehensive list of major world cities
CITY_COORDINATES = {
    # Saudi Arabia
    "Riyadh": (24.7136, 46.6753),
    "Jeddah": (21.4858, 39.1925),
    "Mecca": (21.3891, 39.8579),
    "Medina": (24.5247, 39.5692),
    "Dammam": (26.4207, 50.0888),
    "Khobar": (26.2172, 50.1971),
    "Tabuk": (28.3838, 36.5550),
    "Abha": (18.2164, 42.5053),
    "Taif": (21.2703, 40.4158),
    "Buraidah": (26.3260, 43.9750),
    "Jubail": (27.0046, 49.6601),
    "Yanbu": (24.0895, 38.0618),
    "Najran": (17.4933, 44.1277),
    "Jizan": (16.8892, 42.5706),
    "Hail": (27.5114, 41.7208),

    # Middle East
    "Cairo": (30.0444, 31.2357),
    "Dubai": (25.2048, 55.2708),
    "Abu Dhabi": (24.4539, 54.3773),
    "Doha": (25.2854, 51.5310),
    "Kuwait City": (29.3759, 47.9774),
    "Manama": (26.2285, 50.5860),
    "Muscat": (23.5880, 58.3829),
    "Amman": (31.9454, 35.9284),
    "Beirut": (33.8938, 35.5018),
    "Damascus": (33.5138, 36.2765),
    "Baghdad": (33.3152, 44.3661),
    "Tehran": (35.6892, 51.3890),
    "Jerusalem": (31.7683, 35.2137),
    "Tel Aviv": (32.0853, 34.7818),

    # Africa
    "Lagos": (6.5244, 3.3792),
    "Johannesburg": (-26.2041, 28.0473),
    "Cape Town": (-33.9249, 18.4241),
    "Nairobi": (-1.2921, 36.8219),
    "Addis Ababa": (9.0320, 38.7469),
    "Casablanca": (33.5731, -7.5898),
    "Algiers": (36.7538, 3.0588),
    "Tunis": (36.8065, 10.1815),
    "Accra": (5.6037, -0.1870),
    "Dakar": (14.7167, -17.4677),

    # Europe - UK & Ireland
    "London": (51.5074, -0.1278),
    "Manchester": (53.4808, -2.2426),
    "Birmingham": (52.4862, -1.8904),
    "Liverpool": (53.4084, -2.9916),
    "Leeds": (53.8008, -1.5491),
    "Glasgow": (55.8642, -4.2518),
    "Edinburgh": (55.9533, -3.1883),
    "Bristol": (51.4545, -2.5879),
    "Newcastle": (54.9783, -1.6178),
    "Sheffield": (53.3811, -1.4701),
    "Norwich": (52.6309, 1.2974),
    "Cambridge": (52.2053, 0.1218),
    "Oxford": (51.7520, -1.2577),
    "Brighton": (50.8225, -0.1372),
    "Cardiff": (51.4816, -3.1791),
    "Belfast": (54.5973, -5.9301),
    "Dublin": (53.3498, -6.2603),

    # Europe - Western
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3851, 2.1734),
    "Rome": (41.9028, 12.4964),
    "Milan": (45.4642, 9.1900),
    "Amsterdam": (52.3676, 4.9041),
    "Brussels": (50.8503, 4.3517),
    "Vienna": (48.2082, 16.3738),
    "Zurich": (47.3769, 8.5417),
    "Geneva": (46.2044, 6.1432),
    "Munich": (48.1351, 11.5820),
    "Frankfurt": (50.1109, 8.6821),
    "Hamburg": (53.5511, 9.9937),
    "Lisbon": (38.7223, -9.1393),
    "Athens": (37.9838, 23.7275),
    "Copenhagen": (55.6761, 12.5683),
    "Stockholm": (59.3293, 18.0686),
    "Oslo": (59.9139, 10.7522),
    "Helsinki": (60.1699, 24.9384),

    # Europe - Eastern
    "Moscow": (55.7558, 37.6173),
    "St Petersburg": (59.9343, 30.3351),
    "Istanbul": (41.0082, 28.9784),
    "Ankara": (39.9334, 32.8597),
    "Warsaw": (52.2297, 21.0122),
    "Prague": (50.0755, 14.4378),
    "Budapest": (47.4979, 19.0402),
    "Bucharest": (44.4268, 26.1025),
    "Sofia": (42.6977, 23.3219),
    "Belgrade": (44.7866, 20.4489),
    "Zagreb": (45.8150, 15.9819),
    "Kiev": (50.4501, 30.5234),
    "Kyiv": (50.4501, 30.5234),
    "Minsk": (53.9006, 27.5590),

    # North America - USA
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431),
    "San Francisco": (37.7749, -122.4194),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "Boston": (42.3601, -71.0589),
    "Washington": (38.9072, -77.0369),
    "Washington DC": (38.9072, -77.0369),
    "Miami": (25.7617, -80.1918),
    "Atlanta": (33.7490, -84.3880),
    "Las Vegas": (36.1699, -115.1398),
    "Detroit": (42.3314, -83.0458),
    "Minneapolis": (44.9778, -93.2650),
    "Portland": (45.5152, -122.6784),
    "Orlando": (28.5383, -81.3792),
    "Charlotte": (35.2271, -80.8431),
    "Nashville": (36.1627, -86.7816),
    "New Orleans": (29.9511, -90.0715),
    "Cleveland": (41.4993, -81.6944),
    "Pittsburgh": (40.4406, -79.9959),
    "Cincinnati": (39.1031, -84.5120),
    "Kansas City": (39.0997, -94.5786),
    "Indianapolis": (39.7684, -86.1581),
    "Columbus": (39.9612, -82.9988),
    "Milwaukee": (43.0389, -87.9065),
    "Baltimore": (39.2904, -76.6122),
    "Tampa": (27.9506, -82.4572),
    "Raleigh": (35.7796, -78.6382),
    "Salt Lake City": (40.7608, -111.8910),
    "Honolulu": (21.3069, -157.8583),
    "Anchorage": (61.2181, -149.9003),

    # North America - Canada
    "Toronto": (43.6532, -79.3832),
    "Montreal": (45.5017, -73.5673),
    "Vancouver": (49.2827, -123.1207),
    "Calgary": (51.0447, -114.0719),
    "Edmonton": (53.5461, -113.4938),
    "Ottawa": (45.4215, -75.6972),
    "Winnipeg": (49.8951, -97.1384),
    "Quebec City": (46.8139, -71.2080),

    # Central America & Caribbean
    "Mexico City": (19.4326, -99.1332),
    "Guadalajara": (20.6597, -103.3496),
    "Monterrey": (25.6866, -100.3161),
    "Havana": (23.1136, -82.3666),
    "San Juan": (18.4655, -66.1057),
    "Panama City": (8.9824, -79.5199),
    "Guatemala City": (14.6349, -90.5069),
    "San Jose CR": (9.9281, -84.0907),

    # South America
    "Sao Paulo": (-23.5505, -46.6333),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Buenos Aires": (-34.6037, -58.3816),
    "Lima": (-12.0464, -77.0428),
    "Bogota": (4.7110, -74.0721),
    "Santiago": (-33.4489, -70.6693),
    "Caracas": (10.4806, -66.9036),
    "Quito": (-0.1807, -78.4678),
    "Montevideo": (-34.9011, -56.1645),
    "Medellin": (6.2442, -75.5812),
    "Brasilia": (-15.7975, -47.8919),

    # Asia - East
    "Tokyo": (35.6762, 139.6503),
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Hong Kong": (22.3193, 114.1694),
    "Seoul": (37.5665, 126.9780),
    "Osaka": (34.6937, 135.5023),
    "Taipei": (25.0330, 121.5654),
    "Guangzhou": (23.1291, 113.2644),
    "Shenzhen": (22.5431, 114.0579),
    "Chengdu": (30.5728, 104.0668),
    "Hangzhou": (30.2741, 120.1551),
    "Nanjing": (32.0603, 118.7969),
    "Wuhan": (30.5928, 114.3055),
    "Xian": (34.3416, 108.9398),
    "Busan": (35.1796, 129.0756),
    "Pyongyang": (39.0392, 125.7625),

    # Asia - South
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "New Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Islamabad": (33.6844, 73.0479),
    "Dhaka": (23.8103, 90.4125),
    "Colombo": (6.9271, 79.8612),
    "Kathmandu": (27.7172, 85.3240),

    # Asia - Southeast
    "Singapore": (1.3521, 103.8198),
    "Bangkok": (13.7563, 100.5018),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Jakarta": (-6.2088, 106.8456),
    "Manila": (14.5995, 120.9842),
    "Ho Chi Minh City": (10.8231, 106.6297),
    "Hanoi": (21.0278, 105.8342),
    "Yangon": (16.8661, 96.1951),
    "Phnom Penh": (11.5564, 104.9282),

    # Oceania
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Perth": (-31.9505, 115.8605),
    "Auckland": (-36.8509, 174.7645),
    "Wellington": (-41.2866, 174.7756),

    # Special cases
    "Unknown": (0, 0),
    "VPN": (45.0, 25.0),  # Generic Eastern Europe for suspicious VPN
    "Tor": (45.0, 25.0),
    "Proxy": (45.0, 25.0),
}

# Risk level colors
RISK_COLORS = {
    "safe": "#00ff00",      # Green
    "low": "#90EE90",       # Light green
    "medium": "#ffaa00",    # Orange
    "high": "#ff6600",      # Dark orange
    "critical": "#ff0000",  # Red
}


def get_risk_level(score: int) -> str:
    """Convert risk score to risk level."""
    if score < 20:
        return "safe"
    elif score < 40:
        return "low"
    elif score < 60:
        return "medium"
    elif score < 80:
        return "high"
    else:
        return "critical"


# Create case-insensitive lookup dictionary
_CITY_LOOKUP = {k.lower(): v for k, v in CITY_COORDINATES.items()}


def get_city_coords(city_name: str) -> tuple:
    """
    Get coordinates for a city with case-insensitive matching.
    Falls back to (0, 0) for unknown cities.
    """
    if not city_name:
        return (0, 0)
    # Try exact match first
    if city_name in CITY_COORDINATES:
        return CITY_COORDINATES[city_name]
    # Try case-insensitive match
    lower_name = city_name.lower()
    if lower_name in _CITY_LOOKUP:
        return _CITY_LOOKUP[lower_name]
    # Try partial match (e.g., "New York City" -> "New York")
    for key in _CITY_LOOKUP:
        if key in lower_name or lower_name in key:
            return _CITY_LOOKUP[key]
    # Unknown city
    return (0, 0)


def create_geolocation_map(
    usual_city: str,
    current_city: str,
    risk_score: int,
    travel_distance_km: float,
    theme: str = "dark"
) -> go.Figure:
    """
    Create a geolocation map showing usual location, current location,
    and impossible travel line if applicable.

    Args:
        usual_city: User's usual city
        current_city: Current session city
        risk_score: Combined risk score (0-100)
        travel_distance_km: Distance between locations
        theme: "dark" or "light" for map styling
    """
    # Get coordinates with case-insensitive matching
    usual_coords = get_city_coords(usual_city)
    if usual_coords == (0, 0):
        usual_coords = CITY_COORDINATES.get("Riyadh")  # Default fallback

    current_coords = get_city_coords(current_city)

    # Handle Unknown city - place it somewhere suspicious
    if current_city.lower() == "unknown" or current_coords == (0, 0):
        # Place unknown at a random suspicious location
        current_coords = (45.0, 25.0)  # Somewhere in Eastern Europe

    risk_level = get_risk_level(risk_score)
    risk_color = RISK_COLORS[risk_level]

    # Determine if this is same location or travel
    is_travel = usual_city.lower() != current_city.lower() and travel_distance_km > 50

    # Create figure
    fig = go.Figure()

    # Add travel line if applicable
    if is_travel:
        # Add a curved line between locations
        fig.add_trace(go.Scattergeo(
            lon=[usual_coords[1], current_coords[1]],
            lat=[usual_coords[0], current_coords[0]],
            mode='lines',
            line=dict(
                width=3,
                color=risk_color,
                dash='dot' if risk_score > 60 else 'solid'
            ),
            opacity=0.8,
            name='Travel Path',
            hoverinfo='skip'
        ))

    # Add usual location marker (green home icon)
    fig.add_trace(go.Scattergeo(
        lon=[usual_coords[1]],
        lat=[usual_coords[0]],
        mode='markers+text',
        marker=dict(
            size=16,
            color='#00aa00',
            symbol='circle',
            line=dict(width=2, color='white' if theme == 'dark' else '#333333')
        ),
        text=[f"🏠 {usual_city}"],
        textposition='top center',
        textfont=dict(size=12, color='white' if theme == 'dark' else '#212529'),
        name=f'Usual: {usual_city}',
        hovertemplate=f"<b>Usual Location</b><br>{usual_city}<br>Home base<extra></extra>"
    ))

    # Add current location marker (colored by risk)
    marker_symbol = 'circle' if risk_score < 40 else 'x' if risk_score > 70 else 'diamond'
    fig.add_trace(go.Scattergeo(
        lon=[current_coords[1]],
        lat=[current_coords[0]],
        mode='markers+text',
        marker=dict(
            size=20 if is_travel else 16,
            color=risk_color,
            symbol=marker_symbol,
            line=dict(width=2, color='white' if theme == 'dark' else '#333333')
        ),
        text=[f"📍 {current_city}"],
        textposition='bottom center',
        textfont=dict(size=12, color='white' if theme == 'dark' else '#212529'),
        name=f'Current: {current_city}',
        hovertemplate=f"<b>Current Session</b><br>{current_city}<br>Risk: {risk_score}/100<br>Distance: {travel_distance_km:.0f} km<extra></extra>"
    ))

    # Configure map layout
    if theme == "dark":
        geo_config = dict(
            showland=True,
            landcolor='rgb(30, 30, 40)',
            countrycolor='rgb(60, 60, 80)',
            coastlinecolor='rgb(60, 60, 80)',
            showocean=True,
            oceancolor='rgb(20, 20, 30)',
            showlakes=True,
            lakecolor='rgb(20, 20, 30)',
            bgcolor='rgba(0,0,0,0)',
        )
        paper_bgcolor = 'rgba(0,0,0,0)'
        font_color = 'white'
    else:
        geo_config = dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(200, 200, 200)',
            coastlinecolor='rgb(150, 150, 150)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showlakes=True,
            lakecolor='rgb(200, 230, 255)',
            bgcolor='rgba(0,0,0,0)',
        )
        paper_bgcolor = 'rgba(0,0,0,0)'
        font_color = '#212529'

    # Center map on the relevant area
    center_lat = (usual_coords[0] + current_coords[0]) / 2
    center_lon = (usual_coords[1] + current_coords[1]) / 2

    # Calculate zoom based on distance
    if travel_distance_km > 3000:
        projection_scale = 1
    elif travel_distance_km > 1000:
        projection_scale = 2
    elif travel_distance_km > 500:
        projection_scale = 3
    else:
        projection_scale = 4

    fig.update_layout(
        geo=dict(
            **geo_config,
            projection_type='natural earth',
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=projection_scale,
            showframe=False,
        ),
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        showlegend=False,
        font=dict(color=font_color),
        title=dict(
            text=f"Session Location {'⚠️' if is_travel else '✓'}" +
                 (f" | {travel_distance_km:.0f} km from usual" if is_travel else ""),
            font=dict(size=14, color=risk_color if is_travel else '#00aa00'),
            x=0.5
        )
    )

    return fig


def create_realtime_feed(
    sessions: List[Dict],
    theme: str = "dark",
    max_display: int = 8
) -> None:
    """
    Display a real-time feed of session evaluations.

    Args:
        sessions: List of session dicts with evaluation results
        theme: "dark" or "light"
        max_display: Maximum number of sessions to display
    """
    if theme == "dark":
        bg_color = "rgba(255,255,255,0.05)"
        text_color = "#ffffff"
        text_muted = "rgba(255,255,255,0.7)"
        border_color = "rgba(255,255,255,0.1)"
    else:
        bg_color = "rgba(0,0,0,0.08)"
        text_color = "#1a1a1a"
        text_muted = "#555555"
        border_color = "rgba(0,0,0,0.15)"

    st.markdown(f"""
    <style>
        .feed-container {{
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
        }}
        .feed-item {{
            background: {bg_color};
            border: 2px solid {border_color};
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease-out;
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-20px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        .feed-user {{
            font-weight: bold;
            color: {text_color} !important;
        }}
        .feed-details {{
            font-size: 0.85rem;
            color: {text_muted} !important;
        }}
        .feed-risk {{
            font-weight: bold;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
        }}
        .risk-safe {{ background: #00aa00; color: white !important; }}
        .risk-low {{ background: #90EE90; color: #212529 !important; }}
        .risk-medium {{ background: #ffaa00; color: #212529 !important; }}
        .risk-high {{ background: #ff6600; color: white !important; }}
        .risk-critical {{ background: #ff0000; color: white !important; animation: pulse 1s infinite; }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Display sessions
    for session in sessions[:max_display]:
        risk_score = session.get('combined_risk', 0)
        risk_level = get_risk_level(risk_score)
        user_id = session.get('user_id', 'Unknown')
        city = session.get('city', 'Unknown')
        service = session.get('service', 'Unknown')
        timestamp = session.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        trust_level = session.get('trust_level', 'Unknown')

        st.markdown(f"""
        <div class="feed-item">
            <div>
                <div class="feed-user">👤 {user_id}</div>
                <div class="feed-details">📍 {city} | 🔧 {service} | ⏰ {timestamp}</div>
            </div>
            <div class="feed-risk risk-{risk_level}">{risk_score}</div>
        </div>
        """, unsafe_allow_html=True)


def generate_demo_sessions(count: int = 10) -> List[Dict]:
    """Generate demo session data for the real-time feed."""
    users = ["Fahad123", "Lama22", "Ahmed99", "Sara_M", "Khalid_A", "Noura_S", "Omar77", "Maha_K"]
    cities = ["Riyadh", "Jeddah", "Dammam", "Mecca", "Cairo", "Dubai", "Unknown"]
    services = ["View Profile", "Check Violations", "Change Password", "Transfer Vehicle", "View Appointments"]

    sessions = []
    base_time = datetime.now()

    for i in range(count):
        is_suspicious = random.random() < 0.3  # 30% suspicious

        if is_suspicious:
            risk_score = random.randint(50, 95)
            city = random.choice(["Cairo", "Dubai", "Unknown", "Moscow"])
            trust_level = "Low Trust" if risk_score < 70 else "Critical"
        else:
            risk_score = random.randint(5, 40)
            city = random.choice(["Riyadh", "Jeddah", "Dammam", "Mecca"])
            trust_level = "High Trust" if risk_score < 25 else "Medium Trust"

        session_time = base_time - timedelta(seconds=i * random.randint(5, 30))

        sessions.append({
            "user_id": random.choice(users),
            "city": city,
            "usual_city": "Riyadh",
            "service": random.choice(services),
            "combined_risk": risk_score,
            "trust_level": trust_level,
            "timestamp": session_time.strftime('%H:%M:%S'),
            "travel_distance_km": random.randint(0, 5000) if is_suspicious else random.randint(0, 50)
        })

    return sessions


def display_threat_stats(sessions: List[Dict], theme: str = "dark") -> None:
    """Display threat statistics from recent sessions."""
    if not sessions:
        return

    total = len(sessions)
    critical = sum(1 for s in sessions if s.get('combined_risk', 0) >= 80)
    high = sum(1 for s in sessions if 60 <= s.get('combined_risk', 0) < 80)
    medium = sum(1 for s in sessions if 40 <= s.get('combined_risk', 0) < 60)
    safe = sum(1 for s in sessions if s.get('combined_risk', 0) < 40)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Sessions", total)
    with cols[1]:
        st.metric("🟢 Safe", safe)
    with cols[2]:
        st.metric("🟡 Medium", medium + high)
    with cols[3]:
        st.metric("🔴 Critical", critical, delta=f"+{critical}" if critical > 0 else None, delta_color="off")
