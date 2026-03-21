"""
Chat endpoints - AI chat assistant with SSE streaming.

Provides a conversational interface for vehicle diagnostics:
- SSE streaming for real-time AI responses
- Vehicle context-aware conversations
- Diagnosis context integration
- Follow-up suggestions in Hungarian
"""

import asyncio
import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.v1.endpoints.auth import get_optional_current_user
from app.api.v1.schemas.chat import ChatRequest
from app.core.logging import get_logger
from app.db.postgres.models import User
from app.services.chat_service import get_chat_service

router = APIRouter()
logger = get_logger(__name__)

# Streaming endpoint protection constants
STREAM_TIMEOUT_SECONDS = 300  # 5-minute max duration per stream
MAX_CONCURRENT_STREAMS = 10  # Limit concurrent chat streams
_stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)


def _format_sse_event(event: dict) -> str:
    """Format a chat event dict as SSE format."""
    event_type = event.get("event_type", "token")
    return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post(
    "/message",
    summary="Chat with AI assistant (SSE streaming)",
    description="""
**Chat endpoint** - Send a message and receive a streaming AI response via Server-Sent Events.

Event types in the stream:
1. **start**: Chat session started
2. **token**: Each text chunk as it's generated
3. **source**: Source references used for the answer
4. **suggestion**: Follow-up question suggestions (in Hungarian)
5. **complete**: Response generation finished
6. **error**: Error occurred

**Event Format (SSE):**
```
event: token
data: {"event_type": "token", "data": {"content": "A P0300..."}, "conversation_id": "..."}

```

**Usage with JavaScript:**
```javascript
const response = await fetch('/api/v1/chat/message', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: "Mi a P0300 hibakod?"})
});
const reader = response.body.getReader();
```
    """,
)
async def chat_message(
    request: ChatRequest,
    http_request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
) -> Any:
    """
    SSE streaming chat endpoint.

    Accepts a ChatRequest and returns a StreamingResponse with SSE events.
    Protected by concurrent stream semaphore and per-stream timeout.

    Args:
        request: Chat request with message and optional context.
        http_request: Raw HTTP request for disconnect detection.
        current_user: Optional authenticated user.

    Returns:
        StreamingResponse with text/event-stream media type.
    """
    conversation_id = request.conversation_id or ""
    chat_service = get_chat_service()

    # Build vehicle context dict from schema if provided
    vehicle_context = None
    if request.vehicle_context:
        vehicle_context = {
            "make": request.vehicle_context.make,
            "model": request.vehicle_context.model,
            "year": request.vehicle_context.year,
            "dtc_codes": request.vehicle_context.dtc_codes or [],
        }

    async def _is_connected() -> bool:
        """Check if the client is still connected."""
        return not await http_request.is_disconnected()

    async def generate_events():
        """Generate SSE events with semaphore and error handling."""
        acquired = False
        try:
            # Acquire semaphore with timeout
            try:
                await asyncio.wait_for(_stream_semaphore.acquire(), timeout=10.0)
                acquired = True
            except asyncio.TimeoutError:
                yield _format_sse_event(
                    {
                        "event_type": "error",
                        "data": {
                            "error_type": "capacity",
                            "message": "A szerver jelenleg tul van terhelve. "
                            "Kerem, probalkozzon kesobb.",
                        },
                        "conversation_id": conversation_id,
                        "timestamp": "",
                    }
                )
                return

            # Stream events from chat service
            async for event in chat_service.process_message(
                message=request.message,
                conversation_id=request.conversation_id,
                vehicle_context=vehicle_context,
                diagnosis_id=request.diagnosis_id,
            ):
                if not await _is_connected():
                    logger.info(
                        "Client disconnected during chat stream",
                        extra={"conversation_id": conversation_id},
                    )
                    return
                yield _format_sse_event(event)

        except asyncio.CancelledError:
            logger.info(
                "Chat stream cancelled",
                extra={"conversation_id": conversation_id},
            )
            return

        except Exception as e:
            logger.exception(
                "Chat streaming error",
                extra={
                    "conversation_id": conversation_id,
                    "error_type": type(e).__name__,
                },
            )
            try:
                yield _format_sse_event(
                    {
                        "event_type": "error",
                        "data": {
                            "error_type": "internal",
                            "message": "Varatlan hiba tortent a chat soran.",
                        },
                        "conversation_id": conversation_id,
                        "timestamp": "",
                    }
                )
            except Exception:
                logger.error(
                    "Failed to send error event for chat stream",
                    extra={"conversation_id": conversation_id},
                )

        finally:
            if acquired:
                _stream_semaphore.release()

    async def _timeout_wrapper():
        """Enforce stream timeout to prevent Slowloris attacks."""
        deadline = asyncio.get_event_loop().time() + STREAM_TIMEOUT_SECONDS
        gen = generate_events()
        try:
            async for event in gen:
                if asyncio.get_event_loop().time() > deadline:
                    logger.warning(
                        "Chat stream timeout reached",
                        extra={"conversation_id": conversation_id},
                    )
                    yield _format_sse_event(
                        {
                            "event_type": "error",
                            "data": {
                                "error_type": "timeout",
                                "message": "A chat tullepte a maximalis idokeretet.",
                            },
                            "conversation_id": conversation_id,
                            "timestamp": "",
                        }
                    )
                    await gen.aclose()
                    return
                yield event
        except GeneratorExit:
            await gen.aclose()

    return StreamingResponse(
        _timeout_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
