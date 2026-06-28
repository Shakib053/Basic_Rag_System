import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from hybrid_retrieval import build_hybrid_retriever

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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
    ("system", """You are a helpful assistant.
Use ONLY the context below to answer.
If the answer is not in context, say 'I don't know'.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

hybrid_retriever = build_hybrid_retriever(vectorstore)

def get_hybrid_docs(query):
    return hybrid_retriever.invoke(query)

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
