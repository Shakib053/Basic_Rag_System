import threading
import time
import unittest
from unittest.mock import patch

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
    def test_ask_question_returns_answer_text(self):
        result = AnswerResult(text="The answer [S1].", mode=AnswerMode.GROUNDED)
        with patch.object(rag_service, "answer_query", return_value=result) as answer_query:
            response = rag_service.ask_question("What is it?")

        self.assertEqual(response, {"answer": "The answer [S1]."})
        answer_query.assert_called_once_with("What is it?", [])

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
