import os
import re
import signal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from context_formatting import build_combined_context
from embeddings.text_embeddings import get_text_embedding_model
from hybrid_retrieval import (
    _load_documents_from_vectorstore,
    build_hybrid_retriever,
    select_final_context_documents,
)

from image_retrieval import (
    format_image_references,
    get_image_docs_with_scores,
    load_image_vectorstore,
)

from query_enhancement import build_multi_query_retriever, rewrite_query
from ingestion.text_vectorstore import load_text_vectorstore

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

SHOW_RETRIEVED_DOCS = True
FINAL_CONTEXT_DOCS = 5
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "false").strip().lower() == "true"
MULTI_QUERY_COUNT = 3
QUERY_REWRITE_TIMEOUT_SECONDS = int(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "15"))
RETRIEVAL_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Can also use "google/gemma-4-31b-it:free"       # fast: query rewriting + multi-query expansion

ANSWER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # heavy: final answer generation
AMBIGUOUS_PRONOUN_PATTERN = re.compile(
    r"\b(he|she|they|it|him|her|them|his|hers|their|theirs)\b",
    re.IGNORECASE,
)
FORBIDDEN_AMBIGUOUS_ANSWER_PATTERN = re.compile(
    r"\b(he|she|him|her|his|hers|the person)\b",
    re.IGNORECASE,
)
DESTINATION_PATTERN = re.compile(r"^#+\s*Destination:\s*(.+)$", re.IGNORECASE)
TRAVEL_QUERY_PATTERN = re.compile(
    r"\b(travel|traveled|travelled|trip|trips|visited|visit|destination|destinations|went)\b",
    re.IGNORECASE,
)

if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

embedding_model = get_text_embedding_model()
vectorstore = load_text_vectorstore(embedding_model)

image_vectorstore = load_image_vectorstore()

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

ANSWER_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions over the user's indexed local documents (notes, profiles, books, PDFs, and Word documents).
If the user asks about your identity, your capabilities, or what you do, explain that you are an AI assistant that searches and answers questions about the contents of the user's documents.
If the context below does not contain the answer, say so plainly.
Otherwise, use ONLY the context below to answer.
If the user's question contains an unclear pronoun such as he, she, they, it, his, her, or their, do not guess the person or entity. Answer from the retrieved evidence and group or label the answer by source document when that helps avoid ambiguity.
Do not infer that first-person text in one document belongs to a named person from another document unless the same retrieved context clearly links them.
Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
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

def print_retrieved_images(image_results):
    if not SHOW_RETRIEVED_DOCS or not image_results:
        return

    print("\nRetrieved image references:")
    for index, reference in enumerate(format_image_references(image_results), start=1):
        source = reference["source"]
        page = reference["page"]
        image_index = reference["image_index"]
        image_path = reference["image_path"]
        score = reference["score"]

        page_text = f" | page {page}" if page is not None else ""
        image_text = f" | image {image_index}" if image_index is not None else ""
        score_text = f" | distance: {score:.4f}" if score is not None else ""
        print(f"{index}. {source}{page_text}{image_text}{score_text}")
        print(f"   {image_path}")

    print()

hybrid_retriever = build_hybrid_retriever(vectorstore)
if ENABLE_MULTI_QUERY:
    hybrid_retriever = build_multi_query_retriever(
        hybrid_retriever,
        retrieval_llm,
        num_queries=MULTI_QUERY_COUNT,
    )

def get_hybrid_docs(query):
    return hybrid_retriever.invoke(query)

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

def get_retrieval_query(question, chat_history):
    try:
        return run_with_timeout(
            "query rewrite",
            QUERY_REWRITE_TIMEOUT_SECONDS,
            lambda: rewrite_query(question, chat_history, retrieval_llm),
        )
    except Exception as exc:
        print(f"Query rewrite skipped: {exc}")
        return question

def is_ambiguous_pronoun_question(question, chat_history) -> bool:
    return not chat_history and bool(AMBIGUOUS_PRONOUN_PATTERN.search(question))

def build_answer_question(question, chat_history):
    if not is_ambiguous_pronoun_question(question, chat_history):
        return question

    topical_query = AMBIGUOUS_PRONOUN_PATTERN.sub("", question)
    topical_query = re.sub(r"\s+", " ", topical_query).strip()

    return (
        "Answer this ambiguous-pronoun document query by source document only.\n"
        f"Topical query without the ambiguous pronoun: {topical_query}\n"
        "Do not use he, she, his, her, the person, or similar wording in the "
        "answer. Do not name a subject unless the same retrieved source chunk "
        "explicitly names that subject. Start each part with the source file "
        "name, for example: 'In Travel_history.txt, ...'."
    )

def _document_source_name(doc):
    return doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown source"

def _document_source_key(doc):
    return doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown source"

def expand_docs_from_selected_sources(selected_docs):
    selected_sources = {_document_source_key(doc) for doc in selected_docs}
    if not selected_sources:
        return selected_docs

    try:
        all_docs = _load_documents_from_vectorstore(vectorstore)
    except Exception as exc:
        print(f"Source expansion skipped: {exc}")
        return selected_docs

    expanded_docs = [
        doc for doc in all_docs
        if _document_source_key(doc) in selected_sources
    ]
    return expanded_docs or selected_docs

def _extract_destinations(doc):
    destinations = []
    normalized_content = re.sub(r"\s+(?=#+\s*Destination:)", "\n", doc.page_content)
    for line in normalized_content.splitlines():
        match = DESTINATION_PATTERN.search(line.strip())
        if match:
            destinations.append(match.group(1).strip())
    return destinations

def build_source_fallback_answer(docs):
    if not docs:
        return "I could not find relevant context in the indexed documents."

    destinations_by_source = {}
    snippets_by_source = {}

    for doc in docs:
        source = _document_source_name(doc)
        destinations_by_source.setdefault(source, [])
        snippets_by_source.setdefault(source, [])

        destinations_by_source[source].extend(_extract_destinations(doc))
        snippet = " ".join(doc.page_content.split())
        if snippet:
            snippets_by_source[source].append(snippet[:300])

    lines = []
    for source, destinations in destinations_by_source.items():
        unique_destinations = list(dict.fromkeys(destinations))
        if unique_destinations:
            lines.append(
                f"In {source}, the retrieved chunks mention these destinations: "
                f"{', '.join(unique_destinations)}."
            )

    if lines:
        return "\n".join(lines)

    for source, snippets in snippets_by_source.items():
        if snippets:
            lines.append(f"In {source}, relevant retrieved text says: {snippets[0]}")

    return "\n".join(lines) if lines else "I could not find relevant context in the indexed documents."

def should_use_source_fallback(answer, question, chat_history):
    if not is_ambiguous_pronoun_question(question, chat_history):
        return False

    text = str(answer or "").strip()
    return not text or bool(FORBIDDEN_AMBIGUOUS_ANSWER_PATTERN.search(text))

def should_answer_from_sources(question, chat_history) -> bool:
    return (
        is_ambiguous_pronoun_question(question, chat_history)
        and bool(TRAVEL_QUERY_PATTERN.search(question))
    )

def get_rag_response(question, chat_history):
    print("Rewriting query...")
    standalone = get_retrieval_query(question, chat_history)

    print("RAG retrieval query:", standalone)

    print("Retrieving documents...")
    candidate_docs = get_hybrid_docs(standalone)

    print("Selecting context...")
    image_results = get_image_docs_with_scores(standalone, image_vectorstore)
    docs = select_final_context_documents(
        standalone,
        candidate_docs,
        rerank_top_k=FINAL_CONTEXT_DOCS,
    )
    print_retrieved_docs(docs)
    print_retrieved_images(image_results)
    context = build_combined_context(docs, image_results)

    if should_answer_from_sources(question, chat_history):
        return build_source_fallback_answer(expand_docs_from_selected_sources(docs))

    print("Generating answer...")
    rag_chain = prompt | answer_llm
    response = rag_chain.invoke({
        "context": context,
        "question": build_answer_question(question, chat_history),
        "chat_history": chat_history
    })
    answer = response.content
    if should_use_source_fallback(answer, question, chat_history):
        return build_source_fallback_answer(docs)

    return answer

if __name__ == "__main__":
    print("\nLocal RAG Chat (type 'exit' to quit)\n")
    chat_history = []

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        answer = get_rag_response(query, chat_history)
        print(f"\nAI: {answer}\n")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
