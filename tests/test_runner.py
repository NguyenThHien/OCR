import cv2
import numpy as np

from benchmark.run import benchmark


def test_failed_inference_counts_ground_truth_as_missed(tmp_path):
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))

    class BrokenDetector:
        def predict(self, image):
            raise RuntimeError("inference failed")

        def synchronize(self):
            pass

    sample = {"id": "one", "image": str(path), "annotations": [
        {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "ignore": False}]}
    _, summary = benchmark(BrokenDetector(), [sample], tmp_path / "run", warmup_iterations=0)
    assert summary["errors"] == 1
    assert summary["fn"] == 1
    assert summary["latency_p50_ms"] is None
