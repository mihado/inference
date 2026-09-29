# Unit tests for stack_service.py. Stdlib unittest only; fastapi comes from
# the caller (`uv run --no-project --with fastapi`), never from this repo.
#   uv run --no-project --with fastapi python py-common/stack_service_test.py
#   make test
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stack_service


def body_of(response):
    return json.loads(response.body.decode("utf8"))


class ErrorTest(unittest.TestCase):
    def test_envelope(self):
        response = stack_service.error(404, "Nope.", "model_not_found")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            body_of(response),
            {"error": {"message": "Nope.", "type": "model_not_found", "code": "model_not_found"}},
        )


class MakeAppTest(unittest.TestCase):
    def test_health_route(self):
        app = stack_service.make_app(lambda: None, "test-load")
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        self.assertIn("/health", paths)


class InfoResponseTest(unittest.TestCase):
    def test_loading(self):
        response = stack_service.info_response(loaded=False, model_id="m", max_client_batch_size=1)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(body_of(response)["error"]["code"], "model_loading")

    def test_loaded_shape(self):
        response = stack_service.info_response(
            loaded=True,
            model_id="m",
            max_client_batch_size=32,
            extra=lambda: {"device": "cuda:0", "paths": ["/rerank"]},
        )
        self.assertEqual(
            response,
            {
                "model_id": "m",
                "device": "cuda:0",
                "paths": ["/rerank"],
                "max_client_batch_size": 32,
            },
        )


class CheckModelMatchTest(unittest.TestCase):
    def test_match_and_missing_pass(self):
        self.assertIsNone(stack_service.check_model_match({"model": "m"}, "m"))
        self.assertIsNone(stack_service.check_model_match({}, "m"))

    def test_mismatch_404s_on_either_key(self):
        for payload in ({"model": "x"}, {"model_id": "x"}):
            with self.subTest(payload=payload):
                response = stack_service.check_model_match(payload, "m")
                self.assertEqual(response.status_code, 404)
                self.assertIn("x", body_of(response)["error"]["message"])


class CheckRerankBodyTest(unittest.TestCase):
    def test_valid_texts(self):
        parsed, err = stack_service.check_rerank_body({"query": "q", "texts": ["a", "b"]}, 64)
        self.assertIsNone(err)
        self.assertEqual(parsed, ("q", ["a", "b"]))

    def test_documents_alias(self):
        parsed, err = stack_service.check_rerank_body({"query": "q", "documents": ["a"]}, 64)
        self.assertIsNone(err)
        self.assertEqual(parsed, ("q", ["a"]))

    def test_empty_passes_through_for_the_caller(self):
        parsed, err = stack_service.check_rerank_body({"query": "q", "texts": []}, 64)
        self.assertIsNone(err)
        self.assertEqual(parsed, ("q", []))

    def test_bad_shapes(self):
        for payload in (
            {"texts": ["a"]},
            {"query": "q"},
            {"query": "q", "texts": "a"},
            {"query": "q", "texts": [1]},
        ):
            with self.subTest(payload=payload):
                parsed, err = stack_service.check_rerank_body(payload, 64)
                self.assertIsNone(parsed)
                self.assertEqual(err.status_code, 400)

    def test_over_limit(self):
        parsed, err = stack_service.check_rerank_body({"query": "q", "texts": ["a"] * 65}, 64)
        self.assertIsNone(parsed)
        self.assertEqual(err.status_code, 413)


class DecodeDataUrlTest(unittest.TestCase):
    def test_valid(self):
        payload = "aGk="  # "hi"
        raw, refusal = stack_service.decode_data_url("data:image/png;base64,%s" % payload)
        self.assertIsNone(refusal)
        self.assertEqual(raw, b"hi")

    def test_refusals(self):
        for url in (
            "https://example.com/x.png",
            "data:image/png,AAAA",
            "data:image/png;base64,!!!",
            "nodata",
        ):
            with self.subTest(url=url):
                raw, refusal = stack_service.decode_data_url(url)
                self.assertIsNone(raw)
                self.assertTrue(refusal)

    def test_oversize_refused_before_decoding(self):
        # Must be valid base64: "QQ==" repeated carries mid-string padding
        # and fails validate=True, which is what left this suite red.
        url = "data:image/png;base64,%s" % ("QUJD" * 100)
        raw, refusal = stack_service.decode_data_url(url, max_bytes=10)
        self.assertIsNone(raw)
        self.assertIn("10", refusal)
        raw, refusal = stack_service.decode_data_url(url)
        self.assertIsNone(refusal)
        self.assertTrue(raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
