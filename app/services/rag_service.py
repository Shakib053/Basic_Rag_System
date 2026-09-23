"""RAG answering pipeline, independent of any CLI or web framework."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from prompts.answer import (
    ANSWER_SYSTEM_PROMPT,
    GENERAL_FALLBACK_PREFIX,
    GENERAL_FALLBACK_SYSTEM_PROMPT,
)
from retrieval.context_formatting import build_cited_context
from embeddings.text_embeddings import get_text_embedding_model
from retrieval.hybrid_retrieval import (
    build_hybrid_retriever,
    select_final_context_documents,
)

from retrieval.query_enhancement import QueryPlan, plan_queries, unique_documents
from retrieval.query_router import QueryMode, relevant_documents, route_query, route_retrieval_result
from retrieval.result import AnswerMode, AnswerResult, cited_ids, remove_invalid_citations
from vectorstore.qdrant_store import (
    load_text_vectorstore,
    retrieve_text_documents,
    text_collection_exists,
)

logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

SHOW_RETRIEVED_DOCS = True
FINAL_CONTEXT_DOCS = 5
QUERY_REWRITE_TIMEOUT_SECONDS = int(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "15"))
RETRIEVAL_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Can also use "google/gemma-4-31b-it:free"       # fast: query rewriting + multi-query expansion

ANSWER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # heavy: final answer generation
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

embedding_model = None
vectorstore = None
if LLM_PROVIDER == "ollama":
    retrieval_llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=128,
        disable_streaming=True,
        sync_client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
    )
    answer_llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        num_predict=512,
        disable_streaming=True,
        sync_client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
    )
else:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter.")
    retrieval_llm = ChatOpenAI(
        model=RETRIEVAL_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=256,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=1,
    )
    answer_llm = ChatOpenAI(
        model=ANSWER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=512,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=1,
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

direct_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERAL_FALLBACK_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def print_retrieved_docs(docs):
    if not SHOW_RETRIEVED_DOCS:
        return

    lines = ["Retrieved documents used for this answer:"]
    for index, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "unknown file")
        chunk_index = doc.metadata.get("chunk_index", "unknown chunk")
        page = doc.metadata.get("page")
        rerank_score = doc.metadata.get("rerank_score")
        
        preview = " ".join(doc.page_content.split())

        if len(preview) > 300:
            preview = preview[:300] + "..."

        score_text = ""
        if rerank_score is not None:
            score_text = f" | rerank score: {rerank_score:.4f}"

        page_text = f" | page {page + 1}" if isinstance(page, int) else ""
        lines.append(f"{index}. {file_name}{page_text} | chunk {chunk_index}{score_text}")
        lines.append(f"   {preview}")

    logger.info("\n".join(lines))

hybrid_retriever = None
# Concurrent API requests must not load the embedding model and store twice.
_retrieval_components_lock = threading.Lock()


def _ensure_retrieval_components():
    global embedding_model, vectorstore, hybrid_retriever

    with _retrieval_components_lock:
        if hybrid_retriever is not None:
            return

        embedding_model = get_text_embedding_model()
        vectorstore = load_text_vectorstore(embedding_model)
        hybrid_retriever = build_hybrid_retriever(vectorstore)


def reset_retrieval_components():
    global embedding_model, vectorstore, hybrid_retriever
    with _retrieval_components_lock:
        embedding_model = None
        vectorstore = None
        hybrid_retriever = None


def get_hybrid_retriever():
    _ensure_retrieval_components()
    return hybrid_retriever

def get_hybrid_docs(query, document_ids=None):
    _ensure_retrieval_components()
    return retrieve_text_documents(
        vectorstore,
        query,
        k=20,
        document_ids=document_ids,
    )

# Thread-based rather than signal-based: SIGALRM only works on the main thread,
# but FastAPI runs sync endpoints in worker threads. A timed-out call is abandoned,
# not killed; it finishes in the background.
_timeout_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-timeout")


def run_with_timeout(label, timeout_seconds, func):
    if timeout_seconds <= 0:
        return func()

    future = _timeout_executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        raise TimeoutError(f"{label} timed out after {timeout_seconds}s") from None

def get_query_plan(question, chat_history):
    try:
        return run_with_timeout(
            "query planning",
            QUERY_REWRITE_TIMEOUT_SECONDS,
            lambda: plan_queries(question, chat_history, retrieval_llm),
        )
    except Exception as exc:
        logger.warning(f"Query planning skipped: {exc}")
        return QueryPlan(queries=[question.strip()])


def _general_answer(question, chat_history, queries, reason) -> AnswerResult:
    try:
        response = (direct_prompt | answer_llm).invoke({
            "question": question,
            "chat_history": chat_history,
        })
        text = str(response.content).strip()
        if not text.startswith(GENERAL_FALLBACK_PREFIX):
            text = f"{GENERAL_FALLBACK_PREFIX}\n\n{text}"
        return AnswerResult(
            text=text,
            mode=AnswerMode.GENERAL,
            retrieval_queries=list(queries),
            reason=reason,
        )
    except Exception as exc:
        logger.warning(f"Answer generation failed: {exc}")
        return AnswerResult(
            text="I couldn't generate an answer because the language model is unavailable.",
            mode=AnswerMode.ERROR,
            retrieval_queries=list(queries),
            reason="answer model unavailable",
        )


def answer_query(question, chat_history=None, *, document_ids=None) -> AnswerResult:
    """Return a structured grounded, general, or error response."""
    chat_history = chat_history or []
    route = route_query(question)
    logger.info(f"Query route: {route.mode.value} ({route.reason})")

    plan = get_query_plan(question, chat_history)
    logger.info(f"RAG retrieval queries: {plan.queries}")

    try:
        if not text_collection_exists():
            return _general_answer(question, chat_history, plan.queries, "document store is empty")
        logger.info("Retrieving documents...")
        result_groups = [
            get_hybrid_docs(query, document_ids=document_ids)
            for query in plan.queries
        ]
        candidate_docs = unique_documents(result_groups)
    except Exception as exc:
        logger.warning(f"Retrieval failed: {exc}")
        return AnswerResult(
            text="I couldn't search your uploaded files because the retrieval system is unavailable.",
            mode=AnswerMode.ERROR,
            retrieval_queries=plan.queries,
            reason="retrieval system unavailable",
        )

    logger.info("Selecting context...")
    rerank_query = plan.queries[-1]
    try:
        docs = select_final_context_documents(
            rerank_query,
            candidate_docs,
            rerank_top_k=FINAL_CONTEXT_DOCS,
        )
    except Exception as exc:
        logger.warning(f"Reranking failed: {exc}")
        return AnswerResult(
            text="I couldn't search your uploaded files because the retrieval system is unavailable.",
            mode=AnswerMode.ERROR,
            retrieval_queries=plan.queries,
            reason="reranker unavailable",
        )
    retrieval_route = route_retrieval_result(docs)
    if retrieval_route.mode == QueryMode.DIRECT:
        logger.info(f"Query route: {retrieval_route.mode.value} ({retrieval_route.reason})")
        return _general_answer(question, chat_history, plan.queries, retrieval_route.reason)

    docs = relevant_documents(docs)
    print_retrieved_docs(docs)
    context, available_citations = build_cited_context(docs)

    logger.info("Generating answer...")
    try:
        response = (prompt | answer_llm).invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })
    except Exception as exc:
        logger.warning(f"Grounded answer generation failed: {exc}")
        return AnswerResult(
            text="I found relevant sources but couldn't generate an answer because the language model is unavailable.",
            mode=AnswerMode.ERROR,
            retrieval_queries=plan.queries,
            reason="answer model unavailable",
        )

    answer = remove_invalid_citations(str(response.content), available_citations).strip()
    used_ids = cited_ids(answer)
    if not used_ids and available_citations:
        source_markers = " ".join(f"[{citation.citation_id}]" for citation in available_citations)
        answer = f"{answer}\n\nSources: {source_markers}"
        used_ids = cited_ids(answer)
    citations = [citation for citation in available_citations if citation.citation_id in used_ids]
    return AnswerResult(
        text=answer,
        mode=AnswerMode.GROUNDED,
        citations=citations,
        retrieval_queries=plan.queries,
        reason=retrieval_route.reason,
    )


def get_rag_response(question, chat_history):
    """Backward-compatible response text wrapper."""
    return answer_query(question, chat_history).text


def ask_question(query: str) -> dict:
    """Answer one question with the full RAG pipeline (no chat history)."""
    result = answer_query(query, [])
    return {"answer": result.text}
