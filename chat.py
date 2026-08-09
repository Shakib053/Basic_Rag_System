import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from hybrid_retrieval import build_hybrid_retriever, rerank_documents
from query_enhancement import build_multi_query_retriever, rewrite_query

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SHOW_RETRIEVED_DOCS = True
FINAL_CONTEXT_DOCS = 5
ENABLE_MULTI_QUERY = True
MULTI_QUERY_COUNT = 3

if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",   # or any supported model
    openai_api_key=HF_TOKEN or None,
    openai_api_base="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=512,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Kazi Tanjim Shakib's helpful personal AI assistant.
If the user asks about your identity, your capabilities, or what you do, explain that you are an AI assistant designed to help search and answer questions about Kazi Tanjim Shakib's professional background, projects, skills, and travels.
Otherwise, use ONLY the context below to answer. If the answer is not in context, say 'I don't know'.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

hybrid_retriever = build_hybrid_retriever(vectorstore)
if ENABLE_MULTI_QUERY:
    hybrid_retriever = build_multi_query_retriever(
        hybrid_retriever,
        llm,
        num_queries=MULTI_QUERY_COUNT,
    )

def get_hybrid_docs(query):
    return hybrid_retriever.invoke(query)

def get_rag_response(question, chat_history):
    standalone = rewrite_query(question, chat_history, llm)

    print("RAG retrieval query:", standalone)

    candidate_docs = get_hybrid_docs(standalone)
    docs = rerank_documents(standalone, candidate_docs, top_k = FINAL_CONTEXT_DOCS)
    print_retrieved_docs(docs)
    context = format_docs(docs)

    rag_chain = prompt | llm
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