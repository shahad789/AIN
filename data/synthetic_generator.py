"""
AIn - Synthetic Log Generator
Generates realistic Absher-style logs for testing and demo purposes.

Creates various scenarios:
- Normal user behavior
- Impossible travel attacks
- Credential stuffing
- Account takeover attempts
- Session hijacking patterns
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import uuid


@dataclass
class SyntheticLogEntry:
    """A synthetic log entry mimicking Absher platform logs."""
    timestamp: str
    user_id: str
    ip: str
    city: str
    device: str
    user_agent: str
    action: str
    service: str
    status: str
    session_id: str
    request_id: str
    response_time_ms: int
    extra: Dict


# Saudi Arabian cities with realistic IP ranges
SAUDI_CITIES = {
    "Riyadh": {"ip_prefix": "212.138", "population_weight": 0.35},
    "Jeddah": {"ip_prefix": "188.247", "population_weight": 0.25},
    "Mecca": {"ip_prefix": "94.99", "population_weight": 0.10},
    "Medina": {"ip_prefix": "95.187", "population_weight": 0.08},
    "Dammam": {"ip_prefix": "86.111", "population_weight": 0.12},
    "Khobar": {"ip_prefix": "86.110", "population_weight": 0.05},
    "Tabuk": {"ip_prefix": "94.98", "population_weight": 0.03},
    "Abha": {"ip_prefix": "95.186", "population_weight": 0.02},
}

# Foreign cities for suspicious activity
FOREIGN_CITIES = {
    "Cairo": {"ip_prefix": "197.32", "country": "Egypt"},
    "Dubai": {"ip_prefix": "94.200", "country": "UAE"},
    "London": {"ip_prefix": "51.140", "country": "UK"},
    "Moscow": {"ip_prefix": "95.173", "country": "Russia"},
    "Lagos": {"ip_prefix": "41.58", "country": "Nigeria"},
    "Unknown": {"ip_prefix": "185.220", "country": "VPN/Tor"},
}

# Absher services
ABSHER_SERVICES = {
    "normal": [
        ("View Profile", "view_profile"),
        ("Check Traffic Violations", "traffic_violations"),
        ("Renew Iqama", "renew_iqama"),
        ("Check Passport Status", "passport_status"),
        ("View Dependents", "view_dependents"),
        ("Download Certificate", "download_certificate"),
        ("Check Visa Status", "visa_status"),
        ("View Appointments", "view_appointments"),
    ],
    "sensitive": [
        ("Transfer Vehicle", "vehicle_transfer"),
        ("Issue Exit Reentry", "exit_reentry"),
        ("Change Password", "change_password"),
        ("Update Mobile Number", "update_mobile"),
        ("Add Dependent", "add_dependent"),
        ("Remove Dependent", "remove_dependent"),
        ("Issue Delegation", "delegation"),
        ("Cancel Visa", "cancel_visa"),
        ("Update Bank Account", "update_bank"),
    ],
}

# Device configurations
DEVICES = {
    "iPhone 14": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
    "iPhone 13": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    "iPhone 12": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
    "Samsung Galaxy S23": "Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36",
    "Samsung Galaxy S22": "Mozilla/5.0 (Linux; Android 12; SM-S901B) AppleWebKit/537.36",
    "Huawei P40": "Mozilla/5.0 (Linux; Android 10; ELS-NX9) AppleWebKit/537.36",
    "Windows PC": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mac": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# Arabic first names
ARABIC_NAMES = [
    "Mohammed", "Ahmed", "Abdullah", "Fahad", "Khalid", "Omar", "Ali", "Hassan",
    "Ibrahim", "Yousef", "Saad", "Nasser", "Faisal", "Sultan", "Majed", "Turki",
    "Sara", "Fatima", "Aisha", "Noura", "Maha", "Lama", "Reem", "Dana", "Haya",
]


class SyntheticLogGenerator:
    """Generates synthetic Absher-style logs."""

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.users = self._generate_users(100)

    def _generate_users(self, count: int) -> Dict[str, Dict]:
        """Generate user profiles with consistent attributes."""
        users = {}
        for i in range(count):
            name = random.choice(ARABIC_NAMES)
            user_id = f"{name}{random.randint(100, 999)}"

            # Assign home city based on population weight
            city = random.choices(
                list(SAUDI_CITIES.keys()),
                weights=[c["population_weight"] for c in SAUDI_CITIES.values()]
            )[0]

            # Assign primary device
            device = random.choice(list(DEVICES.keys()))

            users[user_id] = {
                "user_id": user_id,
                "home_city": city,
                "primary_device": device,
                "work_hours": (8, 18) if random.random() > 0.3 else (22, 6),  # Some night workers
                "risk_profile": random.choice(["low", "low", "low", "medium", "high"]),
            }

        return users

    def _generate_ip(self, city: str, is_foreign: bool = False) -> str:
        """Generate realistic IP for a city."""
        cities = FOREIGN_CITIES if is_foreign else SAUDI_CITIES
        city_data = cities.get(city, SAUDI_CITIES["Riyadh"])
        prefix = city_data["ip_prefix"]
        return f"{prefix}.{random.randint(1, 254)}.{random.randint(1, 254)}"

    def generate_normal_session(
        self,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        num_actions: int = None
    ) -> List[SyntheticLogEntry]:
        """Generate a normal user session."""
        if user_id is None:
            user_id = random.choice(list(self.users.keys()))

        user = self.users[user_id]
        if timestamp is None:
            timestamp = datetime.now() - timedelta(hours=random.randint(0, 24))

        if num_actions is None:
            num_actions = random.randint(2, 8)

        session_id = str(uuid.uuid4())[:8]
        entries = []

        # Login
        entries.append(self._create_entry(
            timestamp=timestamp,
            user_id=user_id,
            city=user["home_city"],
            device=user["primary_device"],
            action="login",
            service="Authentication",
            status="success",
            session_id=session_id,
        ))

        # Browse services
        current_time = timestamp + timedelta(seconds=random.randint(5, 30))
        for _ in range(num_actions - 1):
            service_name, service_endpoint = random.choice(ABSHER_SERVICES["normal"])
            entries.append(self._create_entry(
                timestamp=current_time,
                user_id=user_id,
                city=user["home_city"],
                device=user["primary_device"],
                action="access",
                service=service_name,
                status="success",
                session_id=session_id,
            ))
            current_time += timedelta(seconds=random.randint(10, 120))

        return entries

    def generate_impossible_travel(
        self,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> List[SyntheticLogEntry]:
        """
        Generate impossible travel scenario.
        User logs in from home city, then shortly after from a distant location.
        """
        if user_id is None:
            user_id = random.choice(list(self.users.keys()))

        user = self.users[user_id]
        if timestamp is None:
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 12))

        session_id_1 = str(uuid.uuid4())[:8]
        session_id_2 = str(uuid.uuid4())[:8]
        entries = []

        # Normal login from home
        entries.append(self._create_entry(
            timestamp=timestamp,
            user_id=user_id,
            city=user["home_city"],
            device=user["primary_device"],
            action="login",
            service="Authentication",
            status="success",
            session_id=session_id_1,
        ))

        # Normal activity
        entries.append(self._create_entry(
            timestamp=timestamp + timedelta(minutes=5),
            user_id=user_id,
            city=user["home_city"],
            device=user["primary_device"],
            action="access",
            service="View Profile",
            status="success",
            session_id=session_id_1,
        ))

        # Suspicious login from foreign location (30 min later - impossible travel)
        foreign_city = random.choice(list(FOREIGN_CITIES.keys()))
        foreign_device = random.choice([d for d in DEVICES.keys() if d != user["primary_device"]])

        entries.append(self._create_entry(
            timestamp=timestamp + timedelta(minutes=30),
            user_id=user_id,
            city=foreign_city,
            device=foreign_device,
            action="login",
            service="Authentication",
            status="success",
            session_id=session_id_2,
            is_foreign=True,
        ))

        # Attempt sensitive action
        sensitive_service = random.choice(ABSHER_SERVICES["sensitive"])
        entries.append(self._create_entry(
            timestamp=timestamp + timedelta(minutes=32),
            user_id=user_id,
            city=foreign_city,
            device=foreign_device,
            action="access",
            service=sensitive_service[0],
            status="success",
            session_id=session_id_2,
            is_foreign=True,
        ))

        return entries

    def generate_credential_stuffing(
        self,
        target_user: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        num_attempts: int = None
    ) -> List[SyntheticLogEntry]:
        """
        Generate credential stuffing attack pattern.
        Multiple failed logins from same IP targeting different users.
        """
        if timestamp is None:
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 6))

        if num_attempts is None:
            num_attempts = random.randint(5, 15)

        attacker_city = random.choice(list(FOREIGN_CITIES.keys()))
        attacker_ip = self._generate_ip(attacker_city, is_foreign=True)
        attacker_device = "Windows PC"  # Usually automated

        entries = []
        current_time = timestamp

        # Multiple failed attempts
        for i in range(num_attempts):
            victim = random.choice(list(self.users.keys()))
            entries.append(self._create_entry(
                timestamp=current_time,
                user_id=victim,
                city=attacker_city,
                device=attacker_device,
                action="login",
                service="Authentication",
                status="failed",
                session_id=str(uuid.uuid4())[:8],
                is_foreign=True,
                override_ip=attacker_ip,
            ))
            current_time += timedelta(seconds=random.randint(1, 5))

        # One success (compromised account)
        if target_user is None:
            target_user = random.choice(list(self.users.keys()))

        session_id = str(uuid.uuid4())[:8]
        entries.append(self._create_entry(
            timestamp=current_time,
            user_id=target_user,
            city=attacker_city,
            device=attacker_device,
            action="login",
            service="Authentication",
            status="success",
            session_id=session_id,
            is_foreign=True,
            override_ip=attacker_ip,
        ))

        # Immediate sensitive action attempt
        entries.append(self._create_entry(
            timestamp=current_time + timedelta(seconds=10),
            user_id=target_user,
            city=attacker_city,
            device=attacker_device,
            action="access",
            service="Change Password",
            status="success",
            session_id=session_id,
            is_foreign=True,
            override_ip=attacker_ip,
        ))

        return entries

    def generate_account_takeover(
        self,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> List[SyntheticLogEntry]:
        """
        Generate account takeover scenario.
        Attacker gains access and rapidly performs sensitive operations.
        """
        if user_id is None:
            user_id = random.choice(list(self.users.keys()))

        user = self.users[user_id]
        if timestamp is None:
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 4))

        # Unusual time (3 AM)
        timestamp = timestamp.replace(hour=3, minute=random.randint(0, 59))

        attacker_city = random.choice(list(FOREIGN_CITIES.keys()))
        attacker_device = random.choice([d for d in DEVICES.keys() if d != user["primary_device"]])
        session_id = str(uuid.uuid4())[:8]

        entries = []

        # Failed attempts first
        for i in range(random.randint(2, 4)):
            entries.append(self._create_entry(
                timestamp=timestamp + timedelta(seconds=i * 30),
                user_id=user_id,
                city=attacker_city,
                device=attacker_device,
                action="login",
                service="Authentication",
                status="failed",
                session_id=str(uuid.uuid4())[:8],
                is_foreign=True,
            ))

        # Successful login
        login_time = timestamp + timedelta(minutes=2)
        entries.append(self._create_entry(
            timestamp=login_time,
            user_id=user_id,
            city=attacker_city,
            device=attacker_device,
            action="login",
            service="Authentication",
            status="success",
            session_id=session_id,
            is_foreign=True,
        ))

        # Rapid sensitive actions (account takeover behavior)
        current_time = login_time + timedelta(seconds=15)
        sensitive_actions = [
            "Change Password",
            "Update Mobile Number",
            "Update Bank Account",
            "Issue Exit Reentry",
        ]

        for action in sensitive_actions:
            entries.append(self._create_entry(
                timestamp=current_time,
                user_id=user_id,
                city=attacker_city,
                device=attacker_device,
                action="access",
                service=action,
                status="success",
                session_id=session_id,
                is_foreign=True,
            ))
            current_time += timedelta(seconds=random.randint(5, 15))

        return entries

    def _create_entry(
        self,
        timestamp: datetime,
        user_id: str,
        city: str,
        device: str,
        action: str,
        service: str,
        status: str,
        session_id: str,
        is_foreign: bool = False,
        override_ip: Optional[str] = None,
    ) -> SyntheticLogEntry:
        """Create a single log entry."""
        ip = override_ip or self._generate_ip(city, is_foreign)
        user_agent = DEVICES.get(device, DEVICES["Windows PC"])

        return SyntheticLogEntry(
            timestamp=timestamp.isoformat() + "Z",
            user_id=user_id,
            ip=ip,
            city=city,
            device=device,
            user_agent=user_agent,
            action=action,
            service=service,
            status=status,
            session_id=session_id,
            request_id=str(uuid.uuid4()),
            response_time_ms=random.randint(50, 500) if status == "success" else random.randint(100, 2000),
            extra={
                "platform": "absher_web" if "PC" in device or "Mac" in device else "absher_app",
                "api_version": "v2",
                "country": FOREIGN_CITIES.get(city, {}).get("country", "Saudi Arabia"),
            }
        )

    def generate_mixed_dataset(
        self,
        num_normal: int = 100,
        num_impossible_travel: int = 10,
        num_credential_stuffing: int = 5,
        num_account_takeover: int = 5,
    ) -> List[SyntheticLogEntry]:
        """Generate a mixed dataset with various scenarios."""
        all_entries = []

        # Normal sessions
        for _ in range(num_normal):
            all_entries.extend(self.generate_normal_session())

        # Attack scenarios
        for _ in range(num_impossible_travel):
            all_entries.extend(self.generate_impossible_travel())

        for _ in range(num_credential_stuffing):
            all_entries.extend(self.generate_credential_stuffing())

        for _ in range(num_account_takeover):
            all_entries.extend(self.generate_account_takeover())

        # Sort by timestamp
        all_entries.sort(key=lambda e: e.timestamp)

        return all_entries

    def export_jsonl(self, entries: List[SyntheticLogEntry], filepath: str):
        """Export entries to JSONL format."""
        with open(filepath, "w") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry)) + "\n")

    def export_json(self, entries: List[SyntheticLogEntry], filepath: str):
        """Export entries to JSON array format."""
        with open(filepath, "w") as f:
            json.dump([asdict(e) for e in entries], f, indent=2)

    def export_csv(self, entries: List[SyntheticLogEntry], filepath: str):
        """Export entries to CSV format."""
        import csv

        if not entries:
            return

        fieldnames = list(asdict(entries[0]).keys())
        fieldnames.remove("extra")  # Handle nested dict separately

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames + ["platform", "country"])
            writer.writeheader()

            for entry in entries:
                row = asdict(entry)
                extra = row.pop("extra")
                row["platform"] = extra.get("platform", "")
                row["country"] = extra.get("country", "")
                writer.writerow(row)


def generate_sample_files(output_dir: str = "."):
    """Generate sample log files in various formats."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    generator = SyntheticLogGenerator(seed=42)

    # Generate dataset
    entries = generator.generate_mixed_dataset(
        num_normal=50,
        num_impossible_travel=5,
        num_credential_stuffing=3,
        num_account_takeover=3,
    )

    # Export in multiple formats
    generator.export_jsonl(entries, os.path.join(output_dir, "absher_logs.jsonl"))
    generator.export_json(entries, os.path.join(output_dir, "absher_logs.json"))
    generator.export_csv(entries, os.path.join(output_dir, "absher_logs.csv"))

    print(f"Generated {len(entries)} log entries")
    print(f"Files saved to {output_dir}/")

    return entries


if __name__ == "__main__":
    generate_sample_files("./sample_logs")
