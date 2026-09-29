import json
import logging
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import main
from app.services import rag_service
from retrieval.query_enhancement import QueryPlan
from retrieval.query_router import QueryMode, QueryRoute
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

    def test_user_query_emits_correlated_stage_events_without_content(self):
        secret = "private query text"
        document = Document(
            page_content="private document content",
            metadata={
                "document_id": "doc-1", "file_name": "private.pdf", "source_locator": "page 1",
                "chunk_id": "chunk-1", "rerank_score": 1.0,
            },
        )
        with (
            patch.object(rag_service, "answer_llm", FakeListChatModel(responses=["private answer [S1]"])),
            patch.object(rag_service, "get_query_plan", return_value=QueryPlan([secret, "expanded query"])),
            patch.object(rag_service, "text_collection_exists", return_value=True),
            patch.object(rag_service, "get_hybrid_docs", side_effect=[[document], [document]]),
            patch.object(rag_service, "select_final_context_documents", return_value=[document]),
            patch.object(rag_service, "route_retrieval_result", return_value=QueryRoute(QueryMode.RETRIEVE, "match")),
            patch.object(rag_service, "relevant_documents", return_value=[document]),
        ):
            result = rag_service.answer_query(secret, entry_point="api")

        self.assertEqual(result.mode, AnswerMode.GROUNDED)
        stages = [record for record in self.records if record["event"] == "rag_stage_completed"]
        self.assertEqual([record["stage"] for record in stages], [
            "query_planning", "retrieval", "reranking", "generation"
        ])
        query_ids = {record["query_id"] for record in self.records}
        self.assertEqual(len(query_ids), 1)
        self.assertEqual(stages[0]["generated_query_count"], 2)
        self.assertEqual(stages[1]["candidate_count"], 1)
        self.assertEqual(stages[2]["input_count"], 1)
        self.assertEqual(stages[2]["selected_count"], 1)
        self.assertTrue(all(record["duration_ms"] >= 0 for record in stages))
        telemetry = json.dumps(self.records)
        self.assertNotIn(secret, telemetry)
        self.assertNotIn("private answer", telemetry)
        self.assertNotIn("private document content", telemetry)

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
        self.assertEqual([record["event"] for record in self.records], [
            "rag_stage_completed", "query_completed"
        ])
        self.assertNotIn(secret, json.dumps(self.records) + " ".join(diagnostics.output))


if __name__ == "__main__":
    unittest.main()
