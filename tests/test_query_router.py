import unittest

from retrieval.query_router import QueryMode, route_query, route_retrieval_result


class DocumentStub:
    def __init__(self, score):
        self.metadata = {"rerank_score": score}


class QueryRouterTests(unittest.TestCase):
    def test_general_writing_task_is_corpus_first(self):
        route = route_query("Write a short explanation of recursion.")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_math_question_is_corpus_first(self):
        route = route_query("What is 18 * 7?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_document_transformation_is_corpus_first(self):
        route = route_query("Summarize my travel history.")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_document_reference_uses_retrieval(self):
        route = route_query("What does the second chapter of the book say?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_personal_document_reference_uses_retrieval(self):
        route = route_query("What is my job according to my profile?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_uncertain_question_defaults_to_retrieval(self):
        route = route_query("What is the Salah app?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_no_reranked_documents_falls_back_to_direct_llm(self):
        route = route_retrieval_result([])

        self.assertEqual(route.mode, QueryMode.DIRECT)

    def test_irrelevant_reranked_documents_fall_back_to_direct_llm(self):
        route = route_retrieval_result(
            [DocumentStub(-0.4), DocumentStub(-2.1)],
            threshold=0.0,
        )

        self.assertEqual(route.mode, QueryMode.DIRECT)

    def test_relevant_reranked_document_keeps_retrieval_route(self):
        route = route_retrieval_result(
            [DocumentStub(-0.4), DocumentStub(0.2)],
            threshold=0.0,
        )

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_calibrated_threshold_is_applied(self):
        route = route_retrieval_result([DocumentStub(0.2)], threshold=0.5)

        self.assertEqual(route.mode, QueryMode.DIRECT)


if __name__ == "__main__":
    unittest.main()
