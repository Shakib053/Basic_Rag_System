from ollama import chat

response = chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in simple terms."
        }
    ]
)

print(response.message.content)
