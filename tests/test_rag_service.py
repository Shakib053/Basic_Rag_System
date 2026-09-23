import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import main
from app.services import rag_service
from retrieval.result import AnswerMode, AnswerResult


def _run_in_worker_thread(func):
    """Run func the way FastAPI runs a sync endpoint: off the main thread."""
    outcome = {}

    def target():
        try:
            outcome["value"] = func()
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=target)
    worker.start()
    worker.join()
    return outcome


class RagServiceTests(unittest.TestCase):
    def test_ask_question_returns_answer_and_deduplicated_sources(self):
        docs = [
            Document(page_content="a", metadata={"file_name": "cv.pdf", "page": 1, "rerank_score": 7.123456}),
            Document(page_content="b", metadata={"file_name": "cv.pdf", "page": 1, "rerank_score": 3.0}),
            Document(page_content="c", metadata={"file_name": "notes.md", "rerank_score": -1.5}),
        ]
        result = AnswerResult(text="The answer [S1].", mode=AnswerMode.GROUNDED, context_documents=docs)
        with patch.object(rag_service, "answer_query", return_value=result) as answer_query:
            response = rag_service.ask_question("What is it?")

        self.assertEqual(response, {
            "answer": "The answer [S1].",
            "sources": [
                {"document": "cv.pdf", "page": 2, "score": 7.1235},
                {"document": "notes.md", "page": None, "score": -1.5},
            ],
        })
        answer_query.assert_called_once_with("What is it?", [])

    def test_general_answer_has_no_sources(self):
        result = AnswerResult(text="General.", mode=AnswerMode.GENERAL)
        with patch.object(rag_service, "answer_query", return_value=result):
            response = rag_service.ask_question("Capital of France?")

        self.assertEqual(response["sources"], [])

    def test_chat_endpoint_returns_chat_response_shape(self):
        payload = {"answer": "A [S1].", "sources": [{"document": "cv.pdf", "page": 2, "score": 7.1}]}
        with patch.object(main, "ask_question", return_value=payload):
            response = TestClient(main.app).post("/chat", json={"query": "q"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)

    def test_run_with_timeout_works_off_main_thread(self):
        outcome = _run_in_worker_thread(
            lambda: rag_service.run_with_timeout("fast", 5, lambda: "done")
        )

        self.assertEqual(outcome, {"value": "done"})

    def test_run_with_timeout_raises_off_main_thread(self):
        outcome = _run_in_worker_thread(
            lambda: rag_service.run_with_timeout("slow", 0.05, lambda: time.sleep(0.5))
        )

        self.assertIsInstance(outcome.get("error"), TimeoutError)
        self.assertIn("slow timed out", str(outcome["error"]))


if __name__ == "__main__":
    unittest.main()
