import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import chat
from retrieval.query_enhancement import QueryPlan
from retrieval.result import AnswerMode


class AnswerPipelineTests(unittest.TestCase):
    def test_normal_question_with_apostrophe_is_not_parsed_as_command(self):
        handled, selected_ids = chat._handle_terminal_command(
            "what is kazi's profession",
            None,
        )

        self.assertFalse(handled)
        self.assertIsNone(selected_ids)

    def test_empty_store_uses_explicit_general_fallback(self):
        model = FakeListChatModel(responses=["Paris is the capital of France."])
        with (
            patch.object(chat, "answer_llm", model),
            patch.object(chat, "get_query_plan", return_value=QueryPlan(["Capital of France?"])),
            patch.object(chat, "text_collection_exists", return_value=False),
        ):
            result = chat.answer_query("Capital of France?")

        self.assertEqual(result.mode, AnswerMode.GENERAL)
        self.assertTrue(result.text.startswith("I couldn’t find support for this"))

    def test_retrieval_failure_is_not_reported_as_no_evidence(self):
        with (
            patch.object(chat, "get_query_plan", return_value=QueryPlan(["question"])),
            patch.object(chat, "text_collection_exists", side_effect=ConnectionError("secret")),
        ):
            result = chat.answer_query("question")

        self.assertEqual(result.mode, AnswerMode.ERROR)
        self.assertNotIn("secret", result.text)
        self.assertNotIn("Based on general knowledge", result.text)

    def test_grounded_answer_removes_unknown_citation(self):
        document = Document(
            page_content="The supported fact.",
            metadata={
                "document_id": "doc-1",
                "file_name": "facts.txt",
                "source_locator": "document",
                "chunk_id": "chunk-1",
                "rerank_score": 1.0,
            },
        )
        model = FakeListChatModel(responses=["The supported fact [S99]."])
        with (
            patch.object(chat, "answer_llm", model),
            patch.object(chat, "get_query_plan", return_value=QueryPlan(["supported fact"])),
            patch.object(chat, "text_collection_exists", return_value=True),
            patch.object(chat, "get_hybrid_docs", return_value=[document]),
            patch.object(chat, "select_final_context_documents", return_value=[document]),
        ):
            result = chat.answer_query("supported fact")

        self.assertEqual(result.mode, AnswerMode.GROUNDED)
        self.assertNotIn("S99", result.text)
        self.assertIn("[S1]", result.text)
        self.assertEqual([citation.citation_id for citation in result.citations], ["S1"])


if __name__ == "__main__":
    unittest.main()
