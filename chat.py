import os
import shlex
import signal
from langchain_core.messages import AIMessage, HumanMessage
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
from ingestion.document_loader import DocumentLoadError
from ingestion.text_pipeline import ingest_file
from vectorstore.qdrant_store import (
    delete_document,
    list_document_records,
    load_text_vectorstore,
    retrieve_text_documents,
    text_collection_exists,
)

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

    print("\nRetrieved documents used for this answer:")
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
        print(f"{index}. {file_name}{page_text} | chunk {chunk_index}{score_text}")
        print(f"   {preview}")
    
    print()

hybrid_retriever = None


def _ensure_retrieval_components():
    global embedding_model, vectorstore, hybrid_retriever

    if hybrid_retriever is not None:
        return

    embedding_model = get_text_embedding_model()
    vectorstore = load_text_vectorstore(embedding_model)
    hybrid_retriever = build_hybrid_retriever(vectorstore)


def reset_retrieval_components():
    global embedding_model, vectorstore, hybrid_retriever
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

def run_with_timeout(label, timeout_seconds, func):
    if timeout_seconds <= 0:
        return func()

    def handle_timeout(signum, frame):
        raise TimeoutError(f"{label} timed out after {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

def get_query_plan(question, chat_history):
    try:
        return run_with_timeout(
            "query planning",
            QUERY_REWRITE_TIMEOUT_SECONDS,
            lambda: plan_queries(question, chat_history, retrieval_llm),
        )
    except Exception as exc:
        print(f"Query planning skipped: {exc}")
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
        print(f"Answer generation failed: {exc}")
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
    print(f"Query route: {route.mode.value} ({route.reason})")

    plan = get_query_plan(question, chat_history)
    print("RAG retrieval queries:", plan.queries)

    try:
        if not text_collection_exists():
            return _general_answer(question, chat_history, plan.queries, "document store is empty")
        print("Retrieving documents...")
        result_groups = [
            get_hybrid_docs(query, document_ids=document_ids)
            for query in plan.queries
        ]
        candidate_docs = unique_documents(result_groups)
    except Exception as exc:
        print(f"Retrieval failed: {exc}")
        return AnswerResult(
            text="I couldn't search your uploaded files because the retrieval system is unavailable.",
            mode=AnswerMode.ERROR,
            retrieval_queries=plan.queries,
            reason="retrieval system unavailable",
        )

    print("Selecting context...")
    rerank_query = plan.queries[-1]
    try:
        docs = select_final_context_documents(
            rerank_query,
            candidate_docs,
            rerank_top_k=FINAL_CONTEXT_DOCS,
        )
    except Exception as exc:
        print(f"Reranking failed: {exc}")
        return AnswerResult(
            text="I couldn't search your uploaded files because the retrieval system is unavailable.",
            mode=AnswerMode.ERROR,
            retrieval_queries=plan.queries,
            reason="reranker unavailable",
        )
    retrieval_route = route_retrieval_result(docs)
    if retrieval_route.mode == QueryMode.DIRECT:
        print(f"Query route: {retrieval_route.mode.value} ({retrieval_route.reason})")
        return _general_answer(question, chat_history, plan.queries, retrieval_route.reason)

    docs = relevant_documents(docs)
    print_retrieved_docs(docs)
    context, available_citations = build_cited_context(docs)

    print("Generating answer...")
    try:
        response = (prompt | answer_llm).invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })
    except Exception as exc:
        print(f"Grounded answer generation failed: {exc}")
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


def _handle_terminal_command(command: str, selected_document_ids: list[str] | None):
    if not command.lstrip().startswith("/"):
        return False, selected_document_ids
    parts = shlex.split(command)
    name = parts[0].casefold() if parts else ""
    if name == "/upload":
        if len(parts) != 2:
            print("Usage: /upload <path>")
            return True, selected_document_ids
        try:
            result = ingest_file(parts[1])
            reset_retrieval_components()
            print(f"{result.status.value}: {result.file_name} ({result.chunk_count} chunks, id {result.document_id})")
            for warning in result.warnings:
                print(f"Warning: {warning}")
        except (DocumentLoadError, OSError, RuntimeError, ValueError) as exc:
            print(f"Upload failed: {exc}")
        return True, selected_document_ids
    if name == "/documents":
        try:
            records = list_document_records()
            if not records:
                print("No documents are indexed.")
            for record in records:
                print(f"{record.document_id} | {record.file_name} | {record.file_type} | {record.chunk_count} chunks")
        except Exception as exc:
            print(f"Could not list documents: {exc}")
        return True, selected_document_ids
    if name == "/delete":
        if len(parts) != 2:
            print("Usage: /delete <document_id>")
            return True, selected_document_ids
        try:
            deleted = delete_document(parts[1])
            reset_retrieval_components()
            print("Document deleted." if deleted else "Document was not found.")
        except Exception as exc:
            print(f"Delete failed: {exc}")
        return True, selected_document_ids
    if name == "/use":
        if len(parts) == 2 and parts[1].casefold() == "all":
            print("Searching all documents.")
            return True, None
        if len(parts) < 2:
            print("Usage: /use all OR /use <document_id> [document_id ...]")
            return True, selected_document_ids
        print(f"Searching {len(parts) - 1} selected document(s).")
        return True, parts[1:]
    return False, selected_document_ids

if __name__ == "__main__":
    print("\nLocal RAG Chat (type 'exit' to quit; /upload, /documents, /delete, /use)\n")
    chat_history = []
    selected_document_ids = None

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        try:
            handled, selected_document_ids = _handle_terminal_command(query, selected_document_ids)
        except ValueError as exc:
            print(f"Invalid command: {exc}")
            continue
        if handled:
            continue

        result = answer_query(query, chat_history, document_ids=selected_document_ids)
        print(f"\nAI: {result.text}\n")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result.text))
