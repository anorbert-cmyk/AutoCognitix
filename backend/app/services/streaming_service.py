"""SSE streaming service for diagnosis results."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any


async def stream_diagnosis_chunks(
    analysis_gen: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap LLM chunks into SSE format."""
    async for chunk in analysis_gen:
        data = json.dumps({"chunk": chunk, "done": False})
        yield f"data: {data}\n\n"
    yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"


async def stream_result_as_chunks(
    result: dict[str, Any],
    chunk_size: int = 50,
) -> AsyncGenerator[str, None]:
    """
    Fallback: stream a completed diagnosis result as SSE chunks.

    Splits the analysis text into chunks and yields each as an SSE event.
    The final event includes the full result payload.

    Args:
        result: Completed diagnosis result dict with at least an "analysis" key.
        chunk_size: Number of characters per text chunk.

    Yields:
        SSE-formatted strings ready to write to a StreamingResponse.
    """
    text = result.get("analysis", "")
    for i in range(0, max(len(text), 1), chunk_size):
        chunk = text[i : i + chunk_size]
        data = json.dumps({"chunk": chunk, "done": False})
        yield f"data: {data}\n\n"
        await asyncio.sleep(0.01)
    final_data = json.dumps({"chunk": "", "done": True, "full_result": result})
    yield f"data: {final_data}\n\n"
