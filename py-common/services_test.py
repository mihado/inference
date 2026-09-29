# Endpoint-level checks for the services importable without a GPU toolchain:
#
#   uv run --no-project --with fastapi --with httpx python py-common/services_test.py
#
# laya, jina, jina-embed and agentjev import torch at module scope, so they are
# not covered here; their /info is produced by the same helper and their request
# bodies go through the same check_model_match / check_rerank_body, both covered
# in stack_service_test.py. What is pinned here is the per-service wiring: the
# discovery shape, the validation branches, and the error envelope around a
# model call.
import importlib.util
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SERVICES = ["julia", "laya", "omnijev", "jina", "jina-embed", "agentjev"]


def load(service):
    # Every service's entrypoint is called server.py, so importing by name
    # returns whichever was cached first. Load by path under a unique name.
    directory = os.path.join(ROOT, service)
    path = os.path.join(directory, "server.py")
    spec = importlib.util.spec_from_file_location(f"{service}_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, directory)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class DiscoveryShapeTest(unittest.TestCase):
    """No service may hand-roll /info again. This is the guard for the drift
    the extraction was meant to end: a hand-rolled copy is a second place for
    the shape, the 503, and the field set to disagree."""

    def test_every_service_delegates_its_info_to_the_helper(self):
        for service in SERVICES:
            with self.subTest(service=service):
                source = open(os.path.join(ROOT, service, "server.py"), encoding="utf8").read()
                match = re.search(r"@app\.get\(\"/info\"\).*?(?=\n@|\Z)", source, re.S)
                self.assertIsNotNone(match, f"{service}: no /info route")
                block = match.group(0)
                self.assertIn("stack_service.info_response(", block, f"{service}: /info is hand-rolled")
                self.assertNotIn('"model_id": SERVED_ID', block, f"{service}: /info builds the shape itself")
                self.assertNotIn("the model is still loading", block, f"{service}: /info repeats the 503")

    def test_no_service_repeats_the_readiness_rationale(self):
        # One copy, in make_app's docstring.
        for service in SERVICES:
            with self.subTest(service=service):
                source = open(os.path.join(ROOT, service, "server.py"), encoding="utf8").read()
                self.assertNotIn("Readiness lives at /info", source, f"{service}: duplicates make_app's docstring")


class ServiceCase(unittest.TestCase):
    service = ""
    endpoint = ""
    bound_name = ""
    paths = ()
    state_ok = "text"

    def setUp(self):
        if not self.service:
            self.skipTest("base class")
        from fastapi.testclient import TestClient

        self.mod = load(self.service)
        self.client = TestClient(self.mod.app)
        self.mod.raise_exc = None
        # julia resolves its device through torch when the env var is empty;
        # there is no torch here, and the value is only echoed by /info.
        if hasattr(self.mod, "DEVICE"):
            self.mod.DEVICE = "cpu"
        self.loaded()

    def stub(self, result):
        mod = self.mod

        def call(self, *args, **kwargs):
            if mod.raise_exc is not None:
                raise mod.raise_exc
            return result

        # The /info fields are read off the model, so the stub needs them.
        return type(
            "Stub",
            (),
            {
                "predict": call,
                "system_one": call,
                "dtype": "torch.float16",
                "device": "cuda:0",
                "dev": "cuda:0",
            },
        )()

    def loaded(self, result=None):
        """Put a stub in every model global the service reads, so the request
        reaches the validation branches instead of the 503."""
        if result is None:
            result = {"answers": {"q": {"choice": 0}}}
        for name in ("model", "agent", "engine"):
            if hasattr(self.mod, name):
                setattr(self.mod, name, self.stub(result))

    def unload(self):
        for name in ("model", "agent", "engine"):
            if hasattr(self.mod, name):
                setattr(self.mod, name, None)

    def body(self, **over):
        base = {"model": self.mod.SERVED_ID, "state": self.state_ok, "questions": {"q": ["a", "b"]}}
        base.update(over)
        return base

    def test_info_is_503_until_the_model_is_loaded(self):
        self.unload()
        got = self.client.get("/info")
        self.assertEqual(got.status_code, 503)
        self.assertEqual(got.json()["error"]["code"], "model_loading")

    def test_info_carries_the_served_shape(self):
        got = self.client.get("/info").json()
        self.assertEqual(got["model_id"], self.mod.SERVED_ID)
        self.assertEqual(got["max_client_batch_size"], getattr(self.mod, self.bound_name))
        self.assertEqual(got["paths"], list(self.paths))
        self.assertIn("device", got)

    def test_model_mismatch_is_404(self):
        got = self.client.post(self.endpoint, json=self.body(model="somebody-elses-model"))
        self.assertEqual(got.status_code, 404)

    def test_empty_state_and_questions_are_400(self):
        for over in ({"state": ""}, {"state": None}, {"questions": {}}, {"questions": None}):
            got = self.client.post(self.endpoint, json=self.body(**over))
            self.assertEqual(got.status_code, 400, str(over))
            self.assertEqual(got.json()["error"]["type"], "invalid_request_error")

    def test_too_many_questions_is_413(self):
        bound = getattr(self.mod, self.bound_name)
        got = self.client.post(
            self.endpoint, json=self.body(questions={f"q{i}": ["a", "b"] for i in range(bound + 1)})
        )
        self.assertEqual(got.status_code, 413)
        self.assertIn(str(bound), got.json()["error"]["message"])

    def test_every_model_call_exception_becomes_a_400(self):  # noqa: D401
        # The regression: laya's /rerank caught ValueError alone, so a TypeError
        # or RuntimeError from the engine escaped as a bare 500 with a stack
        # trace while every other service returned the error envelope.
        for exc in (ValueError, KeyError, TypeError, RuntimeError, OSError):
            with self.subTest(exc=exc.__name__):
                self.mod.raise_exc = exc("engine refused")
                got = self.client.post(self.endpoint, json=self.body())
                self.assertEqual(got.status_code, 400, exc.__name__)
                self.assertEqual(got.json()["error"]["type"], "invalid_request_error")
                self.mod.raise_exc = None

    def test_a_happy_request_answers(self):
        got = self.client.post(self.endpoint, json=self.body())
        self.assertEqual(got.status_code, 200, got.text)
        self.assertIn("model", got.json())


class JuliaTest(ServiceCase):
    service = "julia"
    endpoint = "/v1/predict"
    bound_name = "MAX_QUESTIONS"
    paths = ("/v1/predict",)


class OmniJevTest(ServiceCase):
    service = "omnijev"
    endpoint = "/v1/systemone"
    bound_name = "MAX_QUESTIONS"
    paths = ("/v1/systemone",)
    state_ok = {"images": ["data:image/png;base64,iVBORw0KGgo="]}

    def test_state_must_carry_exactly_one_image(self):
        for images in ([], ["data:image/png;base64,iVBORw0KGgo="] * 2, ["not a data url"], [7]):
            got = self.client.post(self.endpoint, json=self.body(state={"images": images}))
            self.assertEqual(got.status_code, 400, str(images))

    def test_a_bad_image_is_400_not_500(self):
        for bad in ("data:image/png;base64,%%%", "https://example.com/x.png"):
            got = self.client.post(self.endpoint, json=self.body(state={"images": [bad]}))
            self.assertEqual(got.status_code, 400, bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
