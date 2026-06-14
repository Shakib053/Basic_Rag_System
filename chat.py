from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# 1. Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load vector DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. LLM
llm = ChatOllama(model="qwen3:1.7b", temperature=0)

# 4. Prompt — now includes chat_history slot
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Use ONLY the context below to answer.
If the answer is not in context, say 'I don't know'.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),  # ← history goes here
    ("human", "{question}"),
])

# 5. Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. RAG chain with history-aware retrieval
def get_rag_response(question, chat_history):
    # Condense question + history into a standalone question for retrieval
    if chat_history:
        condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the conversation history and a follow-up question, "
           "rewrite the follow-up as a standalone QUESTION (not an answer, not a statement). "
           "Example:\n"
           "History: Human: My name is Shakib. AI: Nice to meet you, Shakib.\n"
           "Follow-up: what do i do\n"
           "Standalone: What is Shakib's profession?\n\n"
           "Now rewrite the follow-up below:"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        # print("condense prompt", condense_prompt)
        condense_chain = condense_prompt | llm
        standalone = condense_chain.invoke({
            "chat_history": chat_history,
            "question": question
        }).content
    else:
        standalone = question  # first question needs no rewriting

    print("refined question is: ", standalone)
    # Retrieve using the standalone question
    docs = retriever.invoke(standalone)
    context = format_docs(docs)

    # Generate answer
    rag_chain = prompt | llm
    response = rag_chain.invoke({
        "context": context,
        "question": question,        # original question shown to user
        "chat_history": chat_history
    })
    return response.content

# 7. Chat loop
print("\nLocal RAG Chat (type 'exit' to quit)\n")

chat_history = []  # grows with each turn

while True:
    query = input("You: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        break

    answer = get_rag_response(query, chat_history)
    print(f"\nAI: {answer}\n")

    # Append this turn to history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))