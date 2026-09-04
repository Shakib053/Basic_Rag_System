from __future__ import annotations

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompts.query import MULTI_QUERY_PROMPT, REWRITE_SYSTEM_PROMPT

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
