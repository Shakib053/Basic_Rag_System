import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from context_formatting import build_combined_context
from embeddings.text_embeddings import get_text_embedding_model
from hybrid_retrieval import build_hybrid_retriever, select_final_context_documents

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
ENABLE_MULTI_QUERY = True
MULTI_QUERY_COUNT = 3
RETRIEVAL_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Can also use "google/gemma-4-31b-it:free"       # fast: query rewriting + multi-query expansion

ANSWER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # heavy: final answer generation

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
        client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
    )
    answer_llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
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
    ("system", """You are a helpful AI assistant that answers questions over the user's indexed local documents (notes, profiles, books, and PDFs).
If the user asks about your identity, your capabilities, or what you do, explain that you are an AI assistant that searches and answers questions about the contents of the user's documents.
If the context below does not contain the answer, say so plainly.
Otherwise, use ONLY the context below to answer.
Context:
{context}"""),
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

def get_rag_response(question, chat_history):
    standalone = rewrite_query(question, chat_history, retrieval_llm)

    print("RAG retrieval query:", standalone)

    candidate_docs = get_hybrid_docs(standalone)

    image_results = get_image_docs_with_scores(standalone, image_vectorstore)
    docs = select_final_context_documents(
        standalone,
        candidate_docs,
        rerank_top_k=FINAL_CONTEXT_DOCS,
    )
    print_retrieved_docs(docs)
    print_retrieved_images(image_results)
    context = build_combined_context(docs, image_results)

    rag_chain = prompt | answer_llm
    response = rag_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })
    return response.content

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
