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
    "- Add likely synonyms and domain terms so that both BM25 keyword search "
    "and semantic search match more passages.\n"
    "- If the question has several distinct parts, merge them into one compact "
    "search query that captures all of them (e.g. sub-queries joined by ';').\n"
    "- Output ONLY the rewritten query. No prefixes, no quotes, no explanation.\n"
    "Example:\n"
    "History: Human: My name is Shakib. AI: Nice to meet you, Shakib.\n"
    "Follow-up: what do i do\n"
    "Rewritten: What is Kazi Tanjim Shakib's profession; what is Shakib's job"
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