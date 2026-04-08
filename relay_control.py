"""Relay Control Logic Module.

Manages relay connection threading, channel control, and class-to-relay mapping.
Hardware communication is delegated to Relay_B.py (Hardware Output layer).
"""

import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QThread, Signal


try:
    from Relay_B import Relay  # type: ignore

    RELAY_AVAILABLE = True
except Exception:
    RELAY_AVAILABLE = False


class RelayConnectionWorker(QThread):
    """Background thread for establishing a relay board connection."""

    connection_result = Signal(bool, str)  # (success, message)

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.relay: Optional[Any] = None

    def run(self) -> None:
        try:
            self.relay = Relay(host=self.host, port=self.port)
            self.relay.connect()
            self.connection_result.emit(True, f"Connected to {self.host}:{self.port}")
        except Exception as e:
            self.connection_result.emit(False, str(e))


def create_default_mappings(count: int = 8) -> List[Dict]:
    """Create the default class-to-relay-channel mapping list."""
    return [
        {"class": None, "channel": i + 1, "last_state": False, "last_on_time": 0.0}
        for i in range(count)
    ]


def evaluate_mappings(
    mappings: List[Dict],
    class_counts: Dict[str, int],
    relay: Optional[Any],
    relay_connected: bool,
    min_on_seconds: float = 1.0,
) -> List[str]:
    """Evaluate all mappings and toggle relay channels as needed.

    Returns a list of matched descriptions like ``"person→CH1"``.
    """
    matched: List[str] = []
    now = time.time()

    for mapping in mappings:
        class_name = mapping["class"]
        channel = mapping["channel"]

        if class_name is None:
            if mapping["last_state"]:
                _set_channel(relay, relay_connected, channel, False)
                mapping["last_state"] = False
            continue

        detected = class_name in class_counts and class_counts[class_name] > 0

        if detected:
            matched.append(f"{class_name}→CH{channel}")

        if detected and not mapping["last_state"]:
            _set_channel(relay, relay_connected, channel, True)
            mapping["last_state"] = True
            mapping["last_on_time"] = now
        elif not detected and mapping["last_state"]:
            elapsed_on = now - mapping["last_on_time"]
            if elapsed_on >= min_on_seconds:
                _set_channel(relay, relay_connected, channel, False)
                mapping["last_state"] = False

    return matched


def _set_channel(relay: Optional[Any], connected: bool, channel: int, on: bool) -> Optional[str]:
    """Send a relay on/off command. Returns an error string or ``None``."""
    if not connected or relay is None:
        return None
    try:
        if on:
            relay.on(channel)
        else:
            relay.off(channel)
        return None
    except Exception as e:
        return f"Relay CH{channel} error: {e}"


def disconnect_relay(relay: Optional[Any], mappings: List[Dict]) -> None:
    """Turn off all active channels and disconnect."""
    if relay is None:
        return
    for mapping in mappings:
        if mapping["last_state"]:
            try:
                relay.off(mapping["channel"])
            except Exception:
                pass
            mapping["last_state"] = False
            mapping["last_on_time"] = 0.0
    try:
        relay.disconnect()
    except Exception:
        pass
