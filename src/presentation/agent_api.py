from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json

from services import auth_service, chat_service, llm_service, agent_service
from Data.models import User, Chat as ChatModel
from Data.database import get_db

router = APIRouter()


@router.get("/agent/chat")
async def agent_chat(
    prompt: str,
    chat_id: str = None,
    model: str = None,
    think: bool = False,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """
    Agentic chat with tool access (Gmail, Calendar, Google Tasks, local tasks).

    The LLM can autonomously use tools to fulfill requests involving email,
    calendar events, and tasks.

    - chat_id: If provided, continues an existing chat. Otherwise creates a new one.
    - model: Override the model for this request (falls back to chat/default).
    - think: Enable extended thinking (if model supports it).
    """

    if chat_id:
        existing = chat_service.get_chat(db, chat_id, user.id)
        if not existing:
            raise HTTPException(status_code=404, detail="Chat not found")
    else:
        new_chat = chat_service.create_chat(db, user.id, title="Agent Chat")
        chat_id = new_chat["id"]

    chat_record = db.query(ChatModel).filter(ChatModel.id == chat_id).first()
    resolved_model = llm_service.resolve_model(
        model or (chat_record.model if chat_record else None)
    )
    # Keep the chat pinned to the model actually used so the picker stays honest.
    if chat_record and chat_record.model != resolved_model:
        chat_record.model = resolved_model
        db.commit()
    model = resolved_model

    db_messages = chat_service.get_chat_messages(db, chat_id)

    chat_service.add_message(db, chat_id, "user", prompt)

    async def stream_with_metadata():
        yield json.dumps({"type": "chat_id", "data": chat_id}) + "\n"

        full_reply = ""

        try:
            async for event_json in agent_service.stream_agent_response(
                prompt=prompt,
                messages=db_messages,
                model=model,
                user=user,
                db=db,
            ):
                parsed = json.loads(event_json)
                if parsed.get("type") == "content":
                    full_reply += parsed["content"]
                elif parsed.get("type") == "done":
                    chat_service.add_message(db, chat_id, "assistant", full_reply)
                yield event_json
        except Exception as e:
            # Surface the real reason to the client instead of dropping the
            # connection (which the UI can only report as a network error).
            if full_reply:
                chat_service.add_message(db, chat_id, "assistant", full_reply)
            yield json.dumps({
                "type": "error",
                "message": llm_service.friendly_error(e),
                "model": model,
            }) + "\n"

    return StreamingResponse(stream_with_metadata(), media_type="text/event-stream")
