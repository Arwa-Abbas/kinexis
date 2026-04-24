from config import *
from collections import deque


class RulesEngine:
    """
    Stateful rule engine. Tracks error history for debouncing.
    Usage:
        engine = RulesEngine()
        errors = engine.check(features)  # call every frame
    """

    RULES = [
        {
            "name": "knee_valgus",
            "display": "⚠ Knees Caving In!",
            "voice": "Knees caving in, push them outward",
            "color": (0, 0, 255),  # BGR red
            "check": lambda f: (
                f.get("left_knee") is not None
                and f["left_knee"] < KNEE_VALGUS_THRESHOLD
            ),
        },
        {
            "name": "forward_lean",
            "display": "⚠ Chest Up!",
            "voice": "Keep your chest up, reduce forward lean",
            "color": (0, 128, 255),
            "check": lambda f: (
                f.get("trunk_angle") is not None
                and f["trunk_angle"] > FORWARD_LEAN_THRESHOLD
            ),
        },
        {
            "name": "asymmetry",
            "display": "⚠ Asymmetric Movement!",
            "voice": "Uneven movement detected, balance left and right",
            "color": (255, 0, 128),
            "check": lambda f: (
                f.get("knee_symmetry") is not None
                and f["knee_symmetry"] > ASYMMETRY_THRESHOLD
            ),
        },
        {
            "name": "shallow_depth",
            "display": "↓ Go Deeper!",
            "voice": "Go deeper, aim for ninety degrees at the knee",
            "color": (0, 255, 128),
            "check": lambda f: (
                f.get("left_knee") is not None
                and f["left_knee"] > SHALLOW_DEPTH_THRESHOLD
            ),
        },
    ]

    def __init__(self):
        # Track how many consecutive frames each error has been active
        self.counters = {r["name"]: deque(maxlen=DEBOUNCE_FRAMES) for r in self.RULES}
        self.active_errors = {}  # currently sustained errors
        self.last_spoken = None

    def check(self, features: dict) -> list:
        """
        Run all rules on current frame features.
        Returns list of active error dicts (name, display, voice, color).
        """
        active = []
        for rule in self.RULES:
            triggered = rule["check"](features)
            self.counters[rule["name"]].append(1 if triggered else 0)

            # Error confirmed if triggered in all last DEBOUNCE_FRAMES frames
            confirmed = sum(self.counters[rule["name"]]) == DEBOUNCE_FRAMES
            if confirmed:
                active.append(rule)

        self.active_errors = {r["name"]: r for r in active}
        return active

    def new_errors_to_speak(self) -> list:
        """Return errors that haven't been spoken yet this cycle."""
        return [r for r in self.active_errors.values() if r["name"] != self.last_spoken]
