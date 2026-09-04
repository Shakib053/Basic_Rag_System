import unittest

from retrieval.query_router import QueryMode, route_query


class QueryRouterTests(unittest.TestCase):
    def test_general_writing_task_uses_direct_llm(self):
        route = route_query("Write a short explanation of recursion.")

        self.assertEqual(route.mode, QueryMode.DIRECT)

    def test_math_question_uses_direct_llm(self):
        route = route_query("What is 18 * 7?")

        self.assertEqual(route.mode, QueryMode.DIRECT)

    def test_document_reference_uses_retrieval(self):
        route = route_query("What does the second chapter of the book say?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_personal_document_reference_uses_retrieval(self):
        route = route_query("What is my job according to my profile?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)

    def test_uncertain_question_defaults_to_retrieval(self):
        route = route_query("What is the Salah app?")

        self.assertEqual(route.mode, QueryMode.RETRIEVE)


if __name__ == "__main__":
    unittest.main()
