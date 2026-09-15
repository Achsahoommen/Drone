import time
import logging
from typing import Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from utils.config import CONFIG, SPRAY_PROFILES, CLASSES, setup_logging

logger = setup_logging("spray_control")


class SprayAction(Enum):
    NO_SPRAY = "no_spray"
    LOW_SPRAY = "low_spray"
    MEDIUM_SPRAY = "medium_spray"
    HIGH_SPRAY = "high_spray"


@dataclass
class SprayCommand:
    action: SprayAction
    duration_sec: float
    flow_rate_lpm: float
    severity_class: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {
            "action": self.action.value,
            "duration_sec": self.duration_sec,
            "flow_rate_lpm": self.flow_rate_lpm,
            "severity_class": self.severity_class,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class SprayDecisionEngine:
    def __init__(self, config: dict = None):
        if config is None:
            config = CONFIG

        self.spray_profiles = config["spray"]
        self.confidence_threshold = config["inference"]["confidence_threshold"]
        self.cooldown_sec = self.spray_profiles.get("cooldown_sec", 1.0)
        self.last_spray_time = 0
        self.spray_log = []

        self._on_spray: Optional[Callable] = None
        self._on_no_spray: Optional[Callable] = None

    def register_callbacks(
        self,
        on_spray: Callable = None,
        on_no_spray: Callable = None,
    ):
        self._on_spray = on_spray
        self._on_no_spray = on_no_spray

    def decide(self, prediction: dict) -> SprayCommand:
        severity_class = prediction["class"]
        confidence = prediction["confidence"]

        if severity_class not in self.spray_profiles:
            logger.warning("Unknown class: {}".format(severity_class))
            return SprayCommand(
                action=SprayAction.NO_SPRAY,
                duration_sec=0,
                flow_rate_lpm=0,
                severity_class=severity_class,
                confidence=confidence,
            )

        if confidence < self.confidence_threshold:
            logger.info(
                "Below confidence threshold ({:.2f} < {:.2f}). No spray.".format(
                    confidence, self.confidence_threshold
                )
            )
            return SprayCommand(
                action=SprayAction.NO_SPRAY,
                duration_sec=0,
                flow_rate_lpm=0,
                severity_class=severity_class,
                confidence=confidence,
            )

        if severity_class == "healthy":
            command = SprayCommand(
                action=SprayAction.NO_SPRAY,
                duration_sec=0,
                flow_rate_lpm=0,
                severity_class=severity_class,
                confidence=confidence,
            )
            logger.info("HEALTHY -> No spray required")
            if self._on_no_spray:
                self._on_no_spray(command)
            self._log_decision(command)
            return command

        current_time = time.time()
        elapsed = current_time - self.last_spray_time
        if elapsed < self.cooldown_sec:
            remaining = self.cooldown_sec - elapsed
            logger.info("Cooldown active ({:.1f}s remaining)".format(remaining))
            return SprayCommand(
                action=SprayAction.NO_SPRAY,
                duration_sec=0,
                flow_rate_lpm=0,
                severity_class=severity_class,
                confidence=confidence,
            )

        profile = self.spray_profiles[severity_class]
        action_map = {
            "mild": SprayAction.LOW_SPRAY,
            "moderate": SprayAction.MEDIUM_SPRAY,
            "severe": SprayAction.HIGH_SPRAY,
        }

        command = SprayCommand(
            action=action_map[severity_class],
            duration_sec=profile["duration_sec"],
            flow_rate_lpm=profile["flow_rate_lpm"],
            severity_class=severity_class,
            confidence=confidence,
        )

        logger.info(
            "SPRAY: {} | Duration: {}s | Flow: {} LPM | Conf: {:.1f}%".format(
                command.action.value, command.duration_sec,
                command.flow_rate_lpm, confidence * 100
            )
        )

        self.last_spray_time = current_time
        self._log_decision(command)

        if self._on_spray:
            self._on_spray(command)

        return command

    def _log_decision(self, command: SprayCommand):
        self.spray_log.append(command.to_dict())

    def get_log(self):
        return self.spray_log.copy()

    def get_stats(self):
        total = len(self.spray_log)
        sprays = sum(
            1 for d in self.spray_log if d["action"] != "no_spray"
        )
        return {
            "total_decisions": total,
            "spray_commands_issued": sprays,
            "no_spray_decisions": total - sprays,
        }


class SpraySimulator:
    def __init__(self):
        self.is_spraying = False
        self.spray_start_time = None
        logger.info("SpraySimulator initialized (for testing without real drone)")

    def execute_spray(self, command: SprayCommand) -> bool:
        if command.action == SprayAction.NO_SPRAY:
            return False

        self.is_spraying = True
        self.spray_start_time = time.time()

        logger.info(
            ">>> SIMULATING SPRAY: {} for {:.1f}s at {} LPM <<<".format(
                command.action.value, command.duration_sec, command.flow_rate_lpm
            )
        )

        time.sleep(command.duration_sec)

        self.is_spraying = False
        self.spray_start_time = None
        logger.info(">>> SPRAY COMPLETE <<<")
        return True

    def stop_spray(self):
        self.is_spraying = False
        self.spray_start_time = None


if __name__ == "__main__":
    engine = SprayDecisionEngine()
    simulator = SpraySimulator()

    engine.register_callbacks(on_spray=simulator.execute_spray)

    test_predictions = [
        {"class": "healthy", "confidence": 0.95},
        {"class": "mild", "confidence": 0.82},
        {"class": "moderate", "confidence": 0.78},
        {"class": "severe", "confidence": 0.91},
        {"class": "mild", "confidence": 0.40},
    ]

    for pred in test_predictions:
        print("\n--- Prediction: {} ({:.1f}%) ---".format(
            pred["class"], pred["confidence"] * 100
        ))
        cmd = engine.decide(pred)
        print("  Decision: {}".format(cmd.to_dict()))
        time.sleep(0.1)

    print("\n=== Spray Statistics ===")
    print(engine.get_stats())
