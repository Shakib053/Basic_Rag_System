from __future__ import annotations

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# step 1: retrieval-optimized query rewriting

REWRITE_SYSTEM_PROMPT = (
    "You rewrite the user's question into a single standalone QUERY that is "
    "optimized for document retrieval (not for answering directly). Rules:\n"
    "- Resolve pronouns and ellipsis using the conversation history so the "
    "query stands alone.\n"
    "- Expand abbreviations and acronyms to their full forms.\n"
    "- Mention the subject name ('Kazi Tanjim Shakib') naturally ONCE. Do NOT repeat or spam the name multiple times across phrases.\n"
    "- Add relevant domain synonyms and search terms (e.g., travel, destinations, trips, locations, visits, projects, skills).\n"
    "- If the question has several distinct parts, merge them into one compact search query.\n"
    "- Output ONLY the rewritten query. No prefixes, no quotes, no explanation.\n"
    "Example:\n"
    "History: Human: What about travel? AI: I can help with travel history.\n"
    "Follow-up: where did i go\n"
    "Rewritten: Kazi Tanjim Shakib travel history destinations trips locations visited"
)

def build_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def rewrite_query(question: str, chat_history, llm) -> str:
    """Return a standalone, retrieval-optimized query for ``question``.

    Always runs the rewrite LLM: it both de-condenses references from
    ``chat_history`` and expands the wording for better recall. Falls back to
    the raw question if the model returns an empty string.
    """
    chain = build_rewrite_prompt() | llm
    rewritten = chain.invoke(
        {
            "chat_history": chat_history or [],
            "question": question,
        }
    )
    content = getattr(rewritten, "content", rewritten)
    text = str(content).strip()
    return text if text else question

# Stage 2: multi-query expansion

MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "You are an assistant that helps a RAG retriever find relevant documents. "
    "Generate {n} different standalone questions for the user's question. "
    "Each alternative must: cover a different wording, synonym, or sub-aspect; "
    "be a complete standalone question; and appear on its OWN line. "
    "Output nothing but the questions, one per line.\n\n"
    "Original question: {question}"
)

def build_multi_query_retriever(
    retriever,
    llm,
    *,
    num_queries: int = 2,
    include_original: bool = True,
) -> MultiQueryRetriever:
    """Wrap a retriever so it searches for several query paraphrases.

    The wrapped retriever returns the union of results across the generated
    paraphrases (plus the original query when ``include_original``). A later
    cross-encoder rerank narrows the merged pool down to the final context.
    """
    prompt = MULTI_QUERY_PROMPT.partial(n=str(num_queries))
    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=prompt,
        include_original=include_original,
    )