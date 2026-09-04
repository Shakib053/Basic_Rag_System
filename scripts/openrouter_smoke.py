
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= os.getenv("OPENROUTER_API_KEY")
)

"""
response = client.chat.completions.create(
    model="google/gemma-4-31b-it:free",
    messages=[
        {
            "role": "user",
            "content": "Explain what Agentic RAG is in simple terms. Also find the difference between Agentic RAG and RAG. Provide a simple example of Agentic RAG."
        }
    ],
)
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="google/gemma-4-31b-it:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=512,
)

response = llm.invoke(
    "Explain what Agentic RAG is in simple terms."
)

print(response.content)
