import logging
import shlex
from langchain_core.messages import AIMessage, HumanMessage

from app.services.rag_service import answer_query, reset_retrieval_components
from ingestion.document_loader import DocumentLoadError
from ingestion.text_pipeline import ingest_file
from vectorstore.qdrant_store import delete_document, list_document_records


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
    logging.basicConfig(format="%(message)s")
    logging.getLogger("app.services.rag_service").setLevel(logging.INFO)
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
