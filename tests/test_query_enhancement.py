import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever

from query_enhancement import (
    build_multi_query_retriever,
    build_rewrite_prompt,
    rewrite_query,
)


class StubRetriever(BaseRetriever):
    """Minimal BaseRetriever that always returns a single hit."""

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content="hit")]


class RewriteQueryTests(unittest.TestCase):
    def test_rewrites_followup_into_standalone_retrieval_query(self):
        # FakeListChatModel returns messages in order; only one call is made.
        fake_llm = FakeListChatModel(
            responses=[
                "What is Kazi Tanjim Shakib's profession; what is Shakib's job"
            ]
        )
        chat_history = [HumanMessage(content="My name is Shakib")]

        result = rewrite_query("what do i do", chat_history, fake_llm)

        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())
        # Must not be a deictic follow-up; must be standalone.
        self.assertNotIn("follow-up", result.lower())
        self.assertNotIn("history", result.lower())

    def test_returns_original_when_model_output_is_empty(self):
        fake_llm = FakeListChatModel(responses=["   "])
        chat_history = [HumanMessage(content="My name is Shakib")]

        result = rewrite_query("what do i do", chat_history, fake_llm)

        self.assertEqual(result, "what do i do")

    def test_handles_empty_history(self):
        fake_llm = FakeListChatModel(responses=["Where has Shakib traveled?"])

        result = rewrite_query("where have i been", [], fake_llm)

        self.assertEqual(result, "Where has Shakib traveled?")

    def test_rewrite_prompt_contains_history_placeholder(self):
        prompt = build_rewrite_prompt()

        self.assertTrue(any(getattr(m, "variable_name", None) == "chat_history" for m in prompt.messages))


class BuildMultiQueryRetrieverTests(unittest.TestCase):
    def test_wraps_retriever_and_includes_original(self):
        fake_llm = FakeListChatModel(responses=["q1", "q2", "q3"])

        retriever = build_multi_query_retriever(StubRetriever(), fake_llm)

        self.assertIs(retriever.include_original, True)
        # A single invoke surfaces the original + the paraphrases (deduped).
        results = retriever.invoke("original question")
        self.assertGreaterEqual(len(results), 1)

    def test_default_include_original_true(self):
        fake_llm = FakeListChatModel(responses=["q1"])

        retriever = build_multi_query_retriever(StubRetriever(), fake_llm)

        self.assertIs(retriever.include_original, True)


if __name__ == "__main__":
    unittest.main()
