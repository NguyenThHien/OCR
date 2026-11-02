"""Only this module depends on the inference framework."""
from typing import Protocol
import os
from pathlib import Path


class Detector(Protocol):
    def predict(self, image) -> list[dict]: ...

    def synchronize(self) -> None: ...


class PaddleDetector:
    def __init__(self, model_name: str, device: str = "cpu", predict_options=None, **kwargs):
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path("artifacts/checkpoints/paddlex").resolve()))
        import paddle
        from paddleocr import TextDetection

        self.paddle = paddle
        self.device = device
        self.predict_options = predict_options or {}
        self.model = TextDetection(model_name=model_name, device=device, **kwargs)

    def synchronize(self):
        if self.device.startswith("gpu"):
            self.paddle.device.cuda.synchronize()

    def predict(self, image):
        results = list(self.model.predict(input=image, batch_size=1, **self.predict_options))
        if len(results) != 1:
            raise ValueError(f"Expected one image result, got {len(results)}")
        result = results[0]
        polys, scores = result["dt_polys"], result["dt_scores"]
        if len(polys) != len(scores):
            raise ValueError("Polygon/score count mismatch")
        return [{"polygon": p.tolist(), "score": float(s)} for p, s in zip(polys, scores)]


def load_detector(model_name: str, device: str = "cpu", **kwargs) -> Detector:
    return PaddleDetector(model_name, device, **kwargs)
