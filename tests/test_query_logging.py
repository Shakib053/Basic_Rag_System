import json
import logging
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.services import rag_service
from retrieval.result import AnswerMode, AnswerResult


class QueryLoggingTests(unittest.TestCase):
    def setUp(self):
        self.records = []

        class Capture(logging.Handler):
            def emit(inner, record):
                self.records.append(json.loads(record.getMessage()))

        self.handler = Capture()
        logging.getLogger("app.query_events").addHandler(self.handler)
        self.addCleanup(logging.getLogger("app.query_events").removeHandler, self.handler)

    def test_api_and_cli_each_emit_one_event(self):
        secret = "private query text"
        for entry_point, mode in [("api", AnswerMode.GROUNDED), ("cli", AnswerMode.GENERAL)]:
            with patch.object(rag_service, "_answer_query", return_value=AnswerResult("private answer", mode)):
                rag_service.answer_query(secret, entry_point=entry_point)

        self.assertEqual(len(self.records), 2)
        self.assertEqual([record["entry_point"] for record in self.records], ["api", "cli"])
        self.assertEqual([record["mode"] for record in self.records], ["grounded", "general"])
        self.assertNotEqual(self.records[0]["query_id"], self.records[1]["query_id"])
        for record in self.records:
            self.assertEqual(record["event"], "query_completed")
            self.assertEqual(record["query_length"], len(secret))
            self.assertGreaterEqual(record["duration_ms"], 0)
            self.assertIn("+00:00", record["timestamp"])
            self.assertNotIn(secret, json.dumps(record))
            self.assertNotIn("private answer", json.dumps(record))

    def test_handled_and_unexpected_errors_emit_one_event_each(self):
        with patch.object(rag_service, "_answer_query", return_value=AnswerResult(
            "error", AnswerMode.ERROR, reason="retrieval system unavailable"
        )):
            rag_service.answer_query("secret", entry_point="api")
        with patch.object(rag_service, "_answer_query", side_effect=RuntimeError("secret exception")):
            with self.assertRaises(RuntimeError):
                rag_service.answer_query("secret", entry_point="cli")

        self.assertEqual([record["reason"] for record in self.records], [
            "retrieval system unavailable", "unexpected error"
        ])
        self.assertEqual([record["mode"] for record in self.records], ["error", "error"])
        self.assertNotIn("secret", json.dumps(self.records))

    def test_internal_evaluation_call_does_not_emit_user_event(self):
        with patch.object(rag_service, "_answer_query", return_value=AnswerResult("ok", AnswerMode.GENERAL)):
            rag_service.answer_query("evaluation")
        self.assertEqual(self.records, [])

    def test_api_endpoint_logs_without_sensitive_pipeline_diagnostics(self):
        secret = "private query text"
        with (
            patch.object(rag_service, "run_with_timeout", side_effect=RuntimeError(secret)),
            patch.object(rag_service, "text_collection_exists", side_effect=RuntimeError(secret)),
            self.assertLogs(rag_service.logger, level="WARNING") as diagnostics,
        ):
            response = TestClient(main.app).post("/chat", json={"query": secret})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "error")
        self.assertEqual(len(self.records), 1)
        self.assertNotIn(secret, json.dumps(self.records) + " ".join(diagnostics.output))


if __name__ == "__main__":
    unittest.main()
