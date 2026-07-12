from fastapi import UploadFile, File, APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from services import llm_service, whisper, rag_service, search_service
from services import chat_service, auth_service
from services.providers import get_available_providers
from Data.models import User, Chat as ChatModel
from Data.database import get_db
import os
import json

router = APIRouter()


# ---------------------------------------------------------------------------
# Unauthenticated utility endpoints
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models():
    """List available models from the current provider."""
    return llm_service.get_available_models()


@router.get("/default-model")
async def default_model():
    """Return the current default model (from settings.json)."""
    model = llm_service.get_default_model()
    return {"default_model": model}


@router.get("/settings")
async def get_settings():
    """Return full settings (without secrets)."""
    return llm_service.get_settings()


@router.get("/prompts")
async def suggested_prompts(category: str = None):
    """Get suggested prompts for the user."""
    return llm_service.get_suggested_prompts(category)


# ---------------------------------------------------------------------------
# Provider management (authenticated)
# ---------------------------------------------------------------------------


@router.get("/providers")
async def list_providers():
    """List available providers with their config status."""
    return get_available_providers()


@router.post("/providers/set")
async def set_provider(
    provider: str = Query(..., description="Provider name: ollama, openai, alibaba, gemini"),
    user: User = Depends(auth_service.get_current_user),
):
    """Switch the active LLM provider."""
    from services.providers import REGISTRY
    if provider not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'. Available: {list(REGISTRY)}")
    settings = llm_service.set_active_provider(provider)
    return {"message": f"Switched to provider '{provider}'", "settings": settings}


@router.post("/providers/config")
async def configure_provider(
    provider: str = Query(...),
    api_key: str = Query(""),
    base_url: str = Query(""),
    user: User = Depends(auth_service.get_current_user),
):
    """Configure a provider's API key and/or base URL."""
    from services.providers import REGISTRY
    if provider not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")
    config = {}
    if api_key:
        config["api_key"] = api_key
    if base_url:
        config["base_url"] = base_url
    llm_service.save_provider_config(provider, config)
    return {"message": f"Configuration updated for '{provider}'"}


# ---------------------------------------------------------------------------
# Context endpoints (authenticated)
# ---------------------------------------------------------------------------


@router.post("/set_context")
async def set_context(
    file: UploadFile = File(...),
    user: User = Depends(auth_service.get_current_user),
):
    content = await file.read()
    mime_type = file.content_type or ""
    if mime_type.startswith("text/"):
        text_context = content.decode("utf-8")
    elif mime_type.startswith("audio/"):
        try:
            text_context = await whisper.transcript.transcribe_audio(content)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Transcription failed: {str(e)}"
            )
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    llm_service.session.extra_context = text_context
    return {"status": "Context loaded"}


@router.post("/set_context_folder")
async def set_folder_context(
    path: str,
    user: User = Depends(auth_service.get_current_user),
):
    """Legacy endpoint - simple text concatenation"""
    if os.path.exists(path):
        llm_service.session.extra_context = llm_service.read_context_from_folder(path)
        return {"status": f"Context loaded from {path}"}
    raise HTTPException(status_code=400, detail="Path does not exist")


@router.post("/models")
async def set_model(
    name: str,
    user: User = Depends(auth_service.get_current_user),
):
    """Set the default model. Use 'auto' to reset to auto-detection."""
    if name != "auto":
        from services.providers import get_current_provider

        try:
            supported = get_current_provider().is_model_supported(name)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot reach provider: {e}",
            )
        if not supported:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{name}' is not available on the current provider.",
            )
    llm_service.save_default_model(name)
    resolved = llm_service.get_default_model()
    llm_service.session.model = resolved or ""
    return {
        "message": f"Default model set to '{name}'",
        "resolved_model": resolved,
    }


# ---------------------------------------------------------------------------
# Chat management endpoints (authenticated)
# ---------------------------------------------------------------------------


@router.post("/chats")
async def create_chat(
    title: str = "New Chat",
    model: str | None = None,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new chat session. Model defaults to settings.json value if omitted."""
    return chat_service.create_chat(db, user.id, title, model)


@router.get("/chats")
async def list_chats(
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """List all chats for the current user."""
    return chat_service.list_chats(db, user.id)


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Get a chat with all its messages."""
    chat = chat_service.get_chat(db, chat_id, user.id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat."""
    if not chat_service.delete_chat(db, chat_id, user.id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "Chat deleted"}


@router.patch("/chats/{chat_id}")
async def rename_chat(
    chat_id: str,
    title: str = Query(...),
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Rename a chat."""
    chat = chat_service.rename_chat(db, chat_id, user.id, title)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.get("/chats/search/")
async def search_chats(
    q: str = Query(..., min_length=1),
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Search through chat messages."""
    return chat_service.search_chats(db, user.id, q)


@router.post("/chats/{chat_id}/export")
async def export_chat(
    chat_id: str,
    fmt: str = Query("json", pattern="^(json|md)$"),
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Export a chat to a file. Formats: json, md."""
    filepath = chat_service.export_chat(db, chat_id, user.id, fmt)
    if not filepath:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "Exported", "path": filepath}


# ---------------------------------------------------------------------------
# Chat / LLM streaming endpoint (authenticated)
# ---------------------------------------------------------------------------


@router.get("/chat")
async def chat(
    prompt: str,
    chat_id: str = None,
    model: str = None,
    use_rag: bool = False,
    use_web_search: bool = False,
    think: bool = False,
    top_k: int = 3,
    search_results: int = 5,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """
    Chat with LLM, optionally using RAG context and/or web search.

    - chat_id: If provided, continues an existing chat. Otherwise creates a new one.
    - model: Override the model for this request (falls back to chat/default).
    - use_rag: Enrich prompt with local document context from ChromaDB.
    - use_web_search: Search the web and inject results as context.
    - think: Enable extended thinking (model must support it).
    """

    # Resolve or create chat
    if chat_id:
        existing = chat_service.get_chat(db, chat_id, user.id)
        if not existing:
            raise HTTPException(status_code=404, detail="Chat not found")
    else:
        new_chat = chat_service.create_chat(db, user.id)
        chat_id = new_chat["id"]

    # Get the chat's model
    chat_record = db.query(ChatModel).filter(ChatModel.id == chat_id).first()
    resolved_model = llm_service.resolve_model(
        model or (chat_record.model if chat_record else None)
    )
    # Keep the chat pinned to the model actually used so the picker stays honest.
    if chat_record and chat_record.model != resolved_model:
        chat_record.model = resolved_model
        db.commit()
    model = resolved_model

    sources = []
    context_parts = []

    # 1. RAG context
    if use_rag:
        if rag_service.current_index is None:
            raise HTTPException(
                status_code=400,
                detail="No RAG index loaded. Please load or create a collection first.",
            )
        try:
            rag_context, rag_sources = rag_service.get_context_for_llm(prompt, top_k)
            context_parts.append(f"=== Document Context ===\n{rag_context}")
            sources.extend([{**s, "type": "rag"} for s in rag_sources])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    # 2. Web search context
    if use_web_search:
        try:
            search_context = search_service.format_search_results_as_context(
                prompt, max_results=search_results
            )
            context_parts.append(f"=== Web Search Results ===\n{search_context}")
            web_results = search_service.web_search(prompt, max_results=search_results)
            for i, result in enumerate(web_results, 1):
                if "error" not in result:
                    sources.append(
                        {
                            "id": f"web-{i}",
                            "type": "web",
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "text": result.get("body", "")[:200] + "...",
                        }
                    )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Web search error: {str(e)}")

    # Build a per-request LLM session with chat history from DB
    llm_session = llm_service.LLMSession(model=model)

    # Load conversation history from database
    db_messages = chat_service.get_chat_messages(db, chat_id)
    llm_session.conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in db_messages
        if m["role"] in ("user", "assistant")
    ]

    # Set context
    if context_parts:
        llm_session.extra_context = "\n\n".join(context_parts)

    # Save user message to DB
    chat_service.add_message(db, chat_id, "user", prompt)

    # Stream response
    async def stream_with_metadata():
        yield json.dumps({"type": "chat_id", "data": chat_id}) + "\n"

        if sources:
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

        full_reply = ""
        full_thinking = ""

        try:
            async for chunk_json in llm_session.stream(prompt, think=think):
                parsed = json.loads(chunk_json)
                if parsed["type"] == "content":
                    full_reply += parsed["content"]
                elif parsed["type"] == "thinking":
                    full_thinking += parsed["content"]
                elif parsed["type"] == "done":
                    chat_service.add_message(
                        db,
                        chat_id,
                        "assistant",
                        full_reply,
                        thinking=full_thinking if full_thinking else None,
                    )
                yield chunk_json
        except Exception as e:
            # Surface the real reason to the client instead of dropping the
            # connection (which the UI can only report as a network error).
            if full_reply:
                chat_service.add_message(db, chat_id, "assistant", full_reply)
            yield json.dumps({
                "type": "error",
                "message": llm_service.friendly_error(e),
                "model": llm_session.model,
            }) + "\n"

    return StreamingResponse(stream_with_metadata(), media_type="text/event-stream")


@router.post("/clear")
async def clear_conversation(
    user: User = Depends(auth_service.get_current_user),
):
    """Reset the global conversation history (legacy)."""
    llm_service.session.clear_history()
    return {"status": "Conversation history cleared"}
