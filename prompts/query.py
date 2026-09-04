"""Query-rewriting and multi-query prompt templates."""

from langchain_core.prompts import PromptTemplate

REWRITE_SYSTEM_PROMPT = (
    "You rewrite the user's question into a single standalone QUERY that is "
    "optimized for document retrieval (not for answering directly). Rules:\n"
    "- Resolve pronouns and ellipsis ONLY when the conversation history "
    "explicitly identifies the referenced person, organization, document, or "
    "thing.\n"
    "- If a pronoun such as he, she, they, it, his, her, or their has no clear "
    "antecedent in the conversation history, do not invent a name. Preserve "
    "the pronoun or rewrite to neutral topical search terms.\n"
    "- Expand abbreviations and acronyms to their full forms.\n"
    "- Add useful synonyms, related terms, entity names, dates, section titles, "
    "and terminology that may appear in the indexed documents.\n"
    "- If the question has several distinct parts, merge them into one compact search query.\n"
    "- Output ONLY the rewritten query. No prefixes, no quotes, no explanation.\n"
    "Example:\n"
    "History: Human: This document is about Alice. AI: I can answer from it.\n"
    "Follow-up: where did she travel\n"
    "Rewritten: Alice travel destinations trips locations visited\n"
    "Example with no antecedent:\n"
    "History: \n"
    "Follow-up: where did he travel\n"
    "Rewritten: travel history destinations trips locations visited"
)

MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "You are an assistant that helps a RAG retriever find relevant documents. "
    "Generate {n} different standalone questions for the user's question. "
    "Each alternative must: cover a different wording, synonym, or sub-aspect; "
    "be a complete standalone question; and appear on its OWN line. "
    "Output nothing but the questions, one per line.\n\n"
    "Original question: {question}"
)
