import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from retrieval.query_enhancement import QueryPlan, parse_query_plan, plan_queries


class QueryEnhancementTests(unittest.TestCase):
    def test_parser_preserves_original_and_deduplicates_queries(self):
        plan = parse_query_plan(
            '{"queries": ["What is ISO 27001?", "ISO 27001 controls"]}',
            "What is ISO 27001?",
        )

        self.assertEqual(plan.queries, ["What is ISO 27001?", "ISO 27001 controls"])

    def test_parser_rejects_extra_output_fields(self):
        with self.assertRaises(ValueError):
            parse_query_plan('{"queries": ["query"], "answer": "no"}', "original")

    def test_malformed_model_output_falls_back_to_original(self):
        model = FakeListChatModel(responses=["not json"])

        plan = plan_queries("Keep ID AB-123 and date 2026-09-05", [], model)

        self.assertEqual(plan, QueryPlan(["Keep ID AB-123 and date 2026-09-05"]))

    def test_plan_limits_output_to_three_queries_including_original(self):
        model = FakeListChatModel(
            responses=['{"queries": ["one", "two", "three", "four"]}']
        )

        plan = plan_queries("original", [], model)

        self.assertEqual(plan.queries, ["original", "one", "two"])


if __name__ == "__main__":
    unittest.main()
