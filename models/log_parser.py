"""
AIn - Log Parser
Multi-format log parser that normalizes different log formats into SessionData.

Supports:
- JSON logs (structured)
- Syslog format
- Apache/Nginx access logs
- CSV logs
- ELK/Elasticsearch format
"""

import re
import json
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from io import StringIO
from math import radians, sin, cos, sqrt, atan2


@dataclass
class RawLogEntry:
    """Raw parsed log entry before normalization."""
    timestamp: datetime
    user_id: Optional[str] = None
    ip: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    status: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class NormalizedSession:
    """Normalized session data matching the scoring engine input."""
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

    def to_dict(self) -> dict:
        return asdict(self)


class LogParser:
    """Multi-format log parser."""

    # Sensitive services that require elevated scrutiny
    SENSITIVE_SERVICES = {
        "change_password", "change-password", "password_reset", "password-reset",
        "transfer_money", "transfer-money", "bank_transfer", "bank-transfer",
        "update_profile", "update-profile", "edit_profile", "edit-profile",
        "add_dependent", "add-dependent", "remove_dependent", "remove-dependent",
        "issue_visa", "issue-visa", "cancel_visa", "cancel-visa",
        "vehicle_transfer", "vehicle-transfer", "property_transfer",
        "delegation", "power_of_attorney", "power-of-attorney",
        "travel_permit", "travel-permit", "exit_reentry", "exit-reentry",
        "civil_status", "civil-status", "marriage", "divorce",
    }

    # Device detection patterns
    DEVICE_PATTERNS = {
        r"iPhone\s*\d*": "iPhone",
        r"iPad": "iPad",
        r"Android": "Android",
        r"Windows NT": "Windows PC",
        r"Macintosh|Mac OS": "Mac",
        r"Linux": "Linux PC",
        r"Samsung": "Samsung",
        r"Huawei": "Huawei",
    }

    # City coordinates for distance calculation (Saudi Arabia focus)
    CITY_COORDS = {
        "riyadh": (24.7136, 46.6753),
        "jeddah": (21.4858, 39.1925),
        "mecca": (21.3891, 39.8579),
        "medina": (24.5247, 39.5692),
        "dammam": (26.4207, 50.0888),
        "khobar": (26.2172, 50.1971),
        "dhahran": (26.2361, 50.0393),
        "tabuk": (28.3838, 36.5550),
        "abha": (18.2164, 42.5053),
        "taif": (21.4373, 40.5127),
        "qatif": (26.5196, 50.0115),
        "jubail": (27.0046, 49.6225),
        "yanbu": (24.0895, 38.0618),
        "najran": (17.4933, 44.1277),
        "jazan": (16.8892, 42.5706),
        "hail": (27.5114, 41.7208),
        "cairo": (30.0444, 31.2357),
        "dubai": (25.2048, 55.2708),
        "london": (51.5074, -0.1278),
        "new york": (40.7128, -74.0060),
        "unknown": (0, 0),
    }

    def __init__(self):
        self.user_profiles: Dict[str, Dict] = {}  # Track user history

    def parse(self, log_data: str, format_type: str = "auto") -> List[RawLogEntry]:
        """
        Parse log data in various formats.

        Args:
            log_data: Raw log string
            format_type: One of 'auto', 'json', 'jsonl', 'syslog', 'apache', 'csv', 'elk'

        Returns:
            List of RawLogEntry objects
        """
        if format_type == "auto":
            format_type = self._detect_format(log_data)

        parsers = {
            "json": self._parse_json,
            "jsonl": self._parse_jsonl,
            "syslog": self._parse_syslog,
            "apache": self._parse_apache,
            "csv": self._parse_csv,
            "elk": self._parse_elk,
        }

        parser = parsers.get(format_type)
        if not parser:
            raise ValueError(f"Unknown format: {format_type}")

        return parser(log_data)

    def _detect_format(self, log_data: str) -> str:
        """Auto-detect log format."""
        stripped = log_data.strip()

        # JSON array
        if stripped.startswith("[") and stripped.endswith("]"):
            return "json"

        # JSONL (one JSON per line)
        if stripped.startswith("{"):
            return "jsonl"

        # CSV (has header with commas)
        first_line = stripped.split("\n")[0]
        if "," in first_line and not first_line.startswith("{"):
            if any(h in first_line.lower() for h in ["timestamp", "user", "ip", "action"]):
                return "csv"

        # Syslog format
        if re.match(r"^[A-Z][a-z]{2}\s+\d+\s+\d+:\d+:\d+", stripped):
            return "syslog"

        # Apache/Nginx format
        if re.match(r'^\d+\.\d+\.\d+\.\d+\s+-\s+-\s+\[', stripped):
            return "apache"

        # Default to JSONL
        return "jsonl"

    def _parse_json(self, log_data: str) -> List[RawLogEntry]:
        """Parse JSON array of logs."""
        entries = []
        data = json.loads(log_data)

        for item in data:
            entry = self._json_to_entry(item)
            if entry:
                entries.append(entry)

        return entries

    def _parse_jsonl(self, log_data: str) -> List[RawLogEntry]:
        """Parse JSON Lines format (one JSON object per line)."""
        entries = []

        for line in log_data.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                entry = self._json_to_entry(item)
                if entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

        return entries

    def _json_to_entry(self, item: dict) -> Optional[RawLogEntry]:
        """Convert JSON dict to RawLogEntry."""
        # Try various timestamp field names
        timestamp_str = (
            item.get("timestamp") or
            item.get("@timestamp") or
            item.get("time") or
            item.get("datetime") or
            item.get("created_at")
        )

        if timestamp_str:
            timestamp = self._parse_timestamp(timestamp_str)
        else:
            timestamp = datetime.now()

        return RawLogEntry(
            timestamp=timestamp,
            user_id=item.get("user_id") or item.get("user") or item.get("username"),
            ip=item.get("ip") or item.get("ip_address") or item.get("client_ip") or item.get("source_ip"),
            action=item.get("action") or item.get("event") or item.get("event_type"),
            resource=item.get("resource") or item.get("service") or item.get("endpoint") or item.get("path"),
            status=item.get("status") or item.get("result") or item.get("outcome"),
            user_agent=item.get("user_agent") or item.get("userAgent") or item.get("agent"),
            location=item.get("location") or item.get("city") or item.get("geo_city"),
            extra=item,
        )

    def _parse_syslog(self, log_data: str) -> List[RawLogEntry]:
        """Parse syslog format."""
        entries = []
        # Syslog pattern: Month Day HH:MM:SS hostname process[pid]: message
        pattern = r'^(\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+?)(?:\[\d+\])?:\s+(.*)$'

        for line in log_data.strip().split("\n"):
            match = re.match(pattern, line)
            if match:
                timestamp_str, hostname, process, message = match.groups()
                # Add year to timestamp
                timestamp_str = f"{datetime.now().year} {timestamp_str}"
                timestamp = datetime.strptime(timestamp_str, "%Y %b %d %H:%M:%S")

                # Extract fields from message
                user_match = re.search(r'user[=:\s]+(\S+)', message, re.I)
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', message)
                action_match = re.search(r'(login|logout|failed|success|access|denied)', message, re.I)

                entries.append(RawLogEntry(
                    timestamp=timestamp,
                    user_id=user_match.group(1) if user_match else None,
                    ip=ip_match.group(1) if ip_match else None,
                    action=action_match.group(1) if action_match else process,
                    resource=process,
                    status="failed" if "failed" in message.lower() else "success",
                    extra={"raw": message, "hostname": hostname},
                ))

        return entries

    def _parse_apache(self, log_data: str) -> List[RawLogEntry]:
        """Parse Apache/Nginx combined log format."""
        entries = []
        # Combined log format: IP - user [timestamp] "METHOD path HTTP/x.x" status size "referer" "user_agent"
        pattern = r'^(\S+)\s+\S+\s+(\S+)\s+\[([^\]]+)\]\s+"(\S+)\s+(\S+)[^"]*"\s+(\d+)\s+\S+\s+"[^"]*"\s+"([^"]*)"'

        for line in log_data.strip().split("\n"):
            match = re.match(pattern, line)
            if match:
                ip, user, timestamp_str, method, path, status, user_agent = match.groups()

                # Parse Apache timestamp format
                try:
                    timestamp = datetime.strptime(timestamp_str.split()[0], "%d/%b/%Y:%H:%M:%S")
                except ValueError:
                    timestamp = datetime.now()

                entries.append(RawLogEntry(
                    timestamp=timestamp,
                    user_id=user if user != "-" else None,
                    ip=ip,
                    action=method,
                    resource=path,
                    status="success" if status.startswith("2") else "failed",
                    user_agent=user_agent,
                ))

        return entries

    def _parse_csv(self, log_data: str) -> List[RawLogEntry]:
        """Parse CSV format logs."""
        entries = []
        reader = csv.DictReader(StringIO(log_data))

        for row in reader:
            # Normalize column names to lowercase
            row = {k.lower().strip(): v for k, v in row.items()}

            timestamp_str = (
                row.get("timestamp") or
                row.get("time") or
                row.get("datetime") or
                row.get("date")
            )
            timestamp = self._parse_timestamp(timestamp_str) if timestamp_str else datetime.now()

            entries.append(RawLogEntry(
                timestamp=timestamp,
                user_id=row.get("user_id") or row.get("user") or row.get("username"),
                ip=row.get("ip") or row.get("ip_address"),
                action=row.get("action") or row.get("event"),
                resource=row.get("resource") or row.get("service") or row.get("endpoint"),
                status=row.get("status") or row.get("result"),
                user_agent=row.get("user_agent"),
                location=row.get("location") or row.get("city"),
                extra=dict(row),
            ))

        return entries

    def _parse_elk(self, log_data: str) -> List[RawLogEntry]:
        """Parse ELK/Elasticsearch JSON format."""
        # ELK format is essentially JSONL with @timestamp and _source
        entries = []

        for line in log_data.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)

                # ELK wraps data in _source
                if "_source" in item:
                    item = item["_source"]

                entry = self._json_to_entry(item)
                if entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

        return entries

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse various timestamp formats."""
        if isinstance(timestamp_str, datetime):
            return timestamp_str

        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d/%b/%Y:%H:%M:%S",
            "%b %d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # Fallback: try to extract datetime-like pattern
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})\D+(\d{2}):(\d{2}):(\d{2})', timestamp_str)
        if match:
            return datetime(*[int(g) for g in match.groups()])

        return datetime.now()

    def normalize_to_session(
        self,
        entries: List[RawLogEntry],
        user_id: str,
        user_history: Optional[Dict] = None
    ) -> NormalizedSession:
        """
        Convert raw log entries for a user into a NormalizedSession.

        Args:
            entries: List of log entries for this session
            user_id: The user to analyze
            user_history: Optional dict with usual_city, usual_device, etc.

        Returns:
            NormalizedSession ready for scoring
        """
        if not entries:
            raise ValueError("No entries to normalize")

        # Filter entries for this user
        user_entries = [e for e in entries if e.user_id == user_id]
        if not user_entries:
            user_entries = entries  # Use all if no user match

        # Get latest entry
        latest = max(user_entries, key=lambda e: e.timestamp)

        # Extract current values
        current_ip = latest.ip or "0.0.0.0"
        current_city = self._extract_city(latest)
        current_device = self._extract_device(latest)
        current_service = self._extract_service(latest)

        # Get user history or use defaults
        if user_history is None:
            user_history = self.user_profiles.get(user_id, {})

        usual_city = user_history.get("usual_city", current_city)
        usual_device = user_history.get("usual_device", current_device)

        # Calculate metrics
        failed_logins = sum(
            1 for e in user_entries
            if e.status and "fail" in e.status.lower()
            and e.action and "login" in e.action.lower()
        )

        is_unusual_time = self._is_unusual_time(latest.timestamp)
        is_sensitive = self._is_sensitive_service(current_service)
        travel_distance = self._calculate_distance(usual_city, current_city)

        # Update user profile
        self.user_profiles[user_id] = {
            "usual_city": usual_city,
            "usual_device": usual_device,
            "last_seen": latest.timestamp,
        }

        return NormalizedSession(
            user_id=user_id,
            ip=current_ip,
            city=current_city,
            usual_city=usual_city,
            device=current_device,
            usual_device=usual_device,
            is_unusual_time=is_unusual_time,
            failed_logins=failed_logins,
            service=current_service,
            is_sensitive_service=is_sensitive,
            travel_distance_km=travel_distance,
        )

    def _extract_city(self, entry: RawLogEntry) -> str:
        """Extract city from log entry."""
        if entry.location:
            return entry.location.title()

        # Try to extract from extra data
        if entry.extra:
            for key in ["city", "geo_city", "location", "geo.city"]:
                if key in entry.extra and entry.extra[key]:
                    return str(entry.extra[key]).title()

        return "Unknown"

    def _extract_device(self, entry: RawLogEntry) -> str:
        """Extract device type from user agent."""
        if not entry.user_agent:
            return "Unknown"

        ua = entry.user_agent
        for pattern, device in self.DEVICE_PATTERNS.items():
            if re.search(pattern, ua, re.I):
                # Try to get specific model
                if device == "iPhone":
                    match = re.search(r"iPhone\s*(\d+)", ua)
                    if match:
                        return f"iPhone {match.group(1)}"
                return device

        return "Unknown"

    def _extract_service(self, entry: RawLogEntry) -> str:
        """Extract service/endpoint name."""
        if entry.resource:
            # Clean up path to service name
            path = entry.resource.strip("/").split("/")[-1]
            path = re.sub(r"[_-]", " ", path)
            return path.title() if path else "View Profile"

        if entry.action:
            return entry.action.title()

        return "View Profile"

    def _is_unusual_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is outside normal hours (6 AM - 11 PM)."""
        hour = timestamp.hour
        return hour < 6 or hour >= 23

    def _is_sensitive_service(self, service: str) -> bool:
        """Check if service is sensitive."""
        service_lower = service.lower().replace(" ", "_")
        return any(s in service_lower for s in self.SENSITIVE_SERVICES)

    def _calculate_distance(self, city1: str, city2: str) -> float:
        """Calculate approximate distance between cities in km."""
        coords1 = self.CITY_COORDS.get(city1.lower(), self.CITY_COORDS["unknown"])
        coords2 = self.CITY_COORDS.get(city2.lower(), self.CITY_COORDS["unknown"])

        if coords1 == (0, 0) or coords2 == (0, 0):
            return 0 if city1.lower() == city2.lower() else 500  # Default distance for unknown

        # Haversine formula
        lat1, lon1 = radians(coords1[0]), radians(coords1[1])
        lat2, lon2 = radians(coords2[0]), radians(coords2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return round(6371 * c, 1)  # Earth radius in km
