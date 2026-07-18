# src/detection/drift_detector.py
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

class DriftDetector:
    """
    ADWIN-style drift detector.
    Monitors attack rate over a sliding window.
    Triggers alert if recent window differs significantly from older window.
    """
    def __init__(self, window_size=1000, threshold=0.05):
        self.window_size = window_size
        self.threshold   = threshold
        self.window      = deque(maxlen=window_size)
        self.drift_count = 0
        self.total_checks = 0

    def update(self, is_attack: bool) -> bool:
        self.window.append(int(is_attack))

        if len(self.window) < self.window_size:
            return False

        self.total_checks += 1
        window_list = list(self.window)
        mid    = len(window_list) // 2
        mean1  = sum(window_list[:mid]) / mid
        mean2  = sum(window_list[mid:]) / (len(window_list) - mid)
        drift  = abs(mean1 - mean2) > self.threshold

        if drift:
            self.drift_count += 1
            logger.warning(
                f"🔄 DRIFT DETECTED! "
                f"Attack rate changed: {mean1:.2%} → {mean2:.2%} "
                f"(diff: {abs(mean1-mean2):.2%})"
            )
        return drift

    def get_stats(self) -> dict:
        if not self.window:
            return {}
        return {
            'window_size':    len(self.window),
            'current_attack_rate': sum(self.window) / len(self.window),
            'drift_count':    self.drift_count,
            'total_checks':   self.total_checks
        }


class FeedbackLoop:
    """
    Stores misclassified samples for future retraining.
    Saves to retraining_queue.jsonl — one JSON per line.
    """
    def __init__(self):
        self.queue_path = BASE_DIR / 'data/processed/retraining_queue.jsonl'
        self.queue_path.parent.mkdir(exist_ok=True)

    def add_sample(self, features: list, predicted: str, actual: str):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted,
            'actual':    actual,
            'features':  features[:10]  # save first 10 features only
        }
        with open(self.queue_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def queue_size(self) -> int:
        if not self.queue_path.exists():
            return 0
        with open(self.queue_path) as f:
            return sum(1 for _ in f)


def test_drift_detector():
    logger.info("Testing drift detector...")
    detector = DriftDetector(window_size=200, threshold=0.05)
    feedback = FeedbackLoop()

    # Phase 1: Normal traffic (10% attack rate)
    logger.info("Phase 1: Normal traffic (10% attack rate)")
    for i in range(200):
        is_attack = np.random.random() < 0.10
        detector.update(is_attack)

    stats = detector.get_stats()
    logger.info(f"Phase 1 stats: attack_rate={stats['current_attack_rate']:.2%}")

    # Phase 2: Simulate attack surge (50% attack rate)
    logger.info("Phase 2: Attack surge (50% attack rate)")
    drifts_detected = 0
    for i in range(200):
        is_attack = np.random.random() < 0.50
        if detector.update(is_attack):
            drifts_detected += 1
            # Log to feedback queue
            feedback.add_sample(
                features=[float(np.random.random()) for _ in range(10)],
                predicted='BENIGN',
                actual='DDoS'
            )

    stats = detector.get_stats()
    logger.info(f"Phase 2 stats: attack_rate={stats['current_attack_rate']:.2%}")
    logger.info(f"Total drifts detected: {detector.drift_count}")
    logger.info(f"Retraining queue size: {feedback.queue_size()}")

    if detector.drift_count > 0:
        logger.info("Drift detector working correctly!")
    else:
        logger.warning("No drift detected — try adjusting threshold")

if __name__ == '__main__':
    test_drift_detector()