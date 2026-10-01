import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fastapi.testclient import TestClient
from quantbench.api import create_app


class FakeRunner:
    def __init__(self, variant):
        self.loads = 0
        self.manifest = {"model_revision": "fixture-revision", "graph": "fixture.onnx",
                         "files": {"fixture.onnx": "a" * 64}}

    def load(self):
        self.loads += 1

    def preprocess(self, texts, *args, **kwargs):
        return texts

    def prepare(self, inputs):
        return inputs

    def predict(self, inputs):
        return np.array([[-3., 3.] if "good" in text else [3., -3.] for text in inputs])


class ApiTests(unittest.TestCase):
    def test_lifecycle_batch_labels_and_metadata(self):
        runner = FakeRunner("onnx_int8")
        app = create_app(runner_factory=lambda variant: runner)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health/ready").status_code, 200)
            response = client.post("/v1/predict", json={"texts": ["good", "bad"], "variant": "onnx_int8"})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual([row["label"] for row in body["predictions"]], ["POSITIVE", "NEGATIVE"])
            self.assertEqual(body["artifact_hash"], "a" * 64)
            self.assertEqual(body["model_revision"], "fixture-revision")
            self.assertEqual(runner.loads, 1)
            for row in body["predictions"]:
                self.assertAlmostEqual(sum(row["scores"]), 1, places=6)
        self.assertFalse(app.state.ready)
        self.assertIsNone(app.state.runner)

    def test_rejection_rules_and_server_variant(self):
        with TestClient(create_app(runner_factory=FakeRunner)) as client:
            for texts in ([], [" "], ["x" * 4097], ["x"] * 9, [123]):
                response = client.post("/v1/predict", json={"texts": texts, "variant": "onnx_int8"})
                self.assertEqual(response.status_code, 422)
            response = client.post("/v1/predict", json={"texts": ["good"], "variant": "onnx_fp32"})
            self.assertEqual(response.status_code, 400)
            response = client.post("/v1/predict", json={"texts": ["good"], "variant": "onnx_int8", "path": "../model.onnx"})
            self.assertEqual(response.status_code, 422)

    def test_busy_request_is_rejected_without_queueing(self):
        entered, release = threading.Event(), threading.Event()

        class SlowRunner(FakeRunner):
            def predict(self, inputs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test release timed out")
                return super().predict(inputs)

        with TestClient(create_app(runner_factory=SlowRunner)) as client, ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, "/v1/predict", json={"texts": ["good"], "variant": "onnx_int8"})
            try:
                self.assertTrue(entered.wait(3))
                second = client.post("/v1/predict", json={"texts": ["good"], "variant": "onnx_int8"})
                self.assertEqual(second.status_code, 503)
                self.assertEqual(client.get("/health/ready").status_code, 200)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=3).status_code, 200)
