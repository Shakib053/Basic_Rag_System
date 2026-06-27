from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from hybrid_retrieval import BM25Index, build_bm25_from_documents, reciprocal_rank_fusion

load_dotenv()                                                      
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load vector DB
bm25_path = Path("./bm25_corpus.json")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

if bm25_path.exists():
    bm25_index = BM25Index.load(bm25_path)
else:
    stored = vectorstore.get()
    documents = []
    for page_content, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        documents.append(Document(page_content=page_content, metadata=metadata or {}))
    bm25_index = build_bm25_from_documents(documents)
    bm25_index.save(bm25_path)

# llm = ChatOllama(model="qwen3:1.7b", temperature=0) runs locally on own mac

# 3. HuggingFace LLM — ChatHuggingFace wrapper
"""
This is a way of calling HuggingFace using Endpoint

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",  # ← swap here
    task="conversational",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)
# llm = ChatHuggingFace(llm=endpoint)  // Need this if we want to create llm

"""

# 3. OpenAI Chat Interface , much easier, don't need to worry much
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",   # or any supported model
    openai_api_key=HF_TOKEN,
    openai_api_base="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=512,
)

# 4. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Use ONLY the context below to answer.
If the answer is not in context, say 'I don't know'.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 5. Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_hybrid_docs(query):
    semantic_docs = vector_retriever.invoke(query)
    keyword_docs = [item.document for item in bm25_index.search(query, top_k=4)]
    return reciprocal_rank_fusion([semantic_docs, keyword_docs])

# 6. RAG chain with history-aware retrieval
def get_rag_response(question, chat_history):
    if chat_history:
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the conversation history and a follow-up question, "
             "rewrite the follow-up as a standalone QUESTION (not an answer, not a statement). "
             "Example:\n"
             "History: Human: My name is Shakib. AI: Nice to meet you, Shakib.\n"
             "Follow-up: what do i do\n"
             "Standalone: What is Shakib's profession?\n\n"
             "Now rewrite the follow-up below:"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        condense_chain = condense_prompt | llm
        standalone = condense_chain.invoke({
            "chat_history": chat_history,
            "question": question
        }).content
    else:
        standalone = question

    print("RAG retrieval query:", standalone)

    docs = get_hybrid_docs(standalone)
    context = format_docs(docs)

    rag_chain = prompt | llm
    response = rag_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })
    return response.content

# 7. Chat loop
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
