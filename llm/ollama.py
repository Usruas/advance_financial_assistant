
import os

from langchain_ollama import ChatOllama
from pydantic import BaseModel


def get_ollama_client():
    """
    Create local Ollama LLM client.
    """

    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:
    """
    Generate structured output using local Ollama.
    """

    llm = ChatOllama(
        model=model or os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )

    # Convert LLM into a structured-output generator
    structured_llm = llm.with_structured_output(
        response_model
    )

    # Generate and validate structured response
    result = structured_llm.invoke(prompt)

    return result
