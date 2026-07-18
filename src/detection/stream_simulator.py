import asyncio
import numpy as np
import joblib
import logging
import time
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR    = BASE_DIR / 'src/models'

class NetworkStreamSimulator:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.total_processed = 0
        self.attack_count = 0

        logger.info("Loading models and data...")
        self.rf  = joblib.load(MODELS_DIR / 'random_forest.pkl')
        self.xgb = joblib.load(MODELS_DIR / 'xgboost.pkl')
        self.le  = joblib.load(PROCESSED_DIR / 'label_encoder.pkl')

        _, self.X_test, _, self.y_test = joblib.load(
            PROCESSED_DIR / 'scaled_data.pkl'
        )
        self.current_idx = 0
        logger.info(f"Simulator ready — {len(self.X_test):,} test flows available")

    def get_next_batch(self):
        end_idx = min(self.current_idx + self.batch_size, len(self.X_test))
        batch   = self.X_test[self.current_idx:end_idx]
        self.current_idx = end_idx
        if self.current_idx >= len(self.X_test):
            self.current_idx = 0  # loop back
        return batch

    def ensemble_predict(self, X):
        rf_proba  = self.rf.predict_proba(X)
        xgb_proba = self.xgb.predict_proba(X)
        combined  = 0.4 * rf_proba + 0.6 * xgb_proba
        return np.argmax(combined, axis=1)

    async def process_batch(self, batch_id):
        batch      = self.get_next_batch()
        start_time = time.time()
        preds      = self.ensemble_predict(batch)
        elapsed    = time.time() - start_time

        # Count attacks
        attack_mask   = self.le.inverse_transform(preds) != 'BENIGN'
        attacks_found = attack_mask.sum()
        self.attack_count     += attacks_found
        self.total_processed  += len(batch)

        throughput = len(batch) / elapsed if elapsed > 0 else 0

        if attacks_found > 0:
            attack_types = self.le.inverse_transform(preds[attack_mask])
            unique_attacks = dict(zip(*np.unique(attack_types, return_counts=True)))
            logger.warning(
                f"[Batch {batch_id}] ATTACKS DETECTED: {unique_attacks} "
                f"| throughput: {throughput:,.0f} flows/sec"
            )
        else:
            logger.info(
                f"[Batch {batch_id}] All {len(batch)} flows BENIGN "
                f"| throughput: {throughput:,.0f} flows/sec"
            )

        return {
            'batch_id':      batch_id,
            'flows':         len(batch),
            'attacks':       int(attacks_found),
            'throughput':    throughput,
            'timestamp':     datetime.now().isoformat()
        }

    async def run(self, n_batches=20):
        logger.info(f"Starting stream — {n_batches} batches of {self.batch_size} flows each")
        print("="*60)

        tasks   = [self.process_batch(i) for i in range(n_batches)]
        results = await asyncio.gather(*tasks)

        print("="*60)
        avg_throughput = np.mean([r['throughput'] for r in results])
        logger.info(f"Total flows processed: {self.total_processed:,}")
        logger.info(f"Total attacks found:   {self.attack_count:,}")
        logger.info(f"Avg throughput:        {avg_throughput:,.0f} flows/sec")

        # Check target
        if avg_throughput >= 10000:
            logger.info("TARGET MET: >10,000 flows/sec!")
        else:
            logger.warning(f"Below target: {avg_throughput:,.0f} < 10,000 flows/sec")

        return results

if __name__ == '__main__':
    simulator = NetworkStreamSimulator(batch_size=10000)
    asyncio.run(simulator.run(n_batches=20))