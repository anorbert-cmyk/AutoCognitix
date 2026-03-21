"""
Chat Service - AI chat assistant for vehicle diagnostics.

Provides conversational AI interface with:
- Hungarian language car mechanic expertise
- Vehicle context-aware responses
- RAG-based knowledge retrieval
- SSE streaming response generation
- Follow-up suggestion generation
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.logging import get_logger
from app.services.llm_provider import LLMConfig, get_llm_provider, is_llm_available

logger = get_logger(__name__)

# System prompt for the chat assistant
CHAT_SYSTEM_PROMPT = (
    "Te egy tapasztalt magyar autoszorelo AI asszisztens vagy az AutoCognitix platformon. "
    "Feladatod, hogy segitsd a felhasznalokat jarmu diagnosztikai kerdesekben.\n\n"
    "Szabalyok:\n"
    "- Mindig magyarul valaszolj, kezelheto, erthetoen.\n"
    "- Hivatkozz konkret DTC kodokra es alkatreszekre, ha relevan.\n"
    "- Ha nem vagy biztos a valaszban, jelezd egyertelmuen.\n"
    "- Ajanlj szakszervizes ellenorzest, ha a problema biztonsagi kockazatot jelent.\n"
    "- Legy udvarias es segitokesz.\n"
    "- Ne adj orvosi, jogi vagy penzugyi tanacsot.\n"
    "- Az AI altal generalt valaszok tajekoztatojeleguek, NEM helyettesitik a szakkepzett "
    "szerelo velemenyet.\n"
)

# Default follow-up suggestions in Hungarian
DEFAULT_SUGGESTIONS: List[str] = [
    "Mi okozhatja meg ezt a hibat?",
    "Mennyibe kerulhet a javitas?",
    "Biztonsagos-e igy vezetni?",
]

# LLM config for chat (slightly more creative than diagnosis)
CHAT_LLM_CONFIG = LLMConfig(
    temperature=0.5,
    max_tokens=2048,
)


MAX_CONVERSATIONS = 500  # Prevent unbounded memory growth

# Blocklist for prompt injection detection
_INJECTION_BLOCKLIST = [
    "SYSTEM:",
    "IGNORE",
    "OVERRIDE",
    "FORGET ALL",
    "IGNORE PREVIOUS",
    "DISREGARD",
    "NEW INSTRUCTIONS",
    "JAILBREAK",
    "DAN:",
    "PROMPT:",
]


def _sanitize_input(text: str) -> str:
    """Strip dangerous markers from user input."""
    sanitized = text
    for marker in _INJECTION_BLOCKLIST:
        # Case-insensitive removal
        idx = sanitized.upper().find(marker.upper())
        while idx != -1:
            sanitized = sanitized[:idx] + sanitized[idx + len(marker) :]
            idx = sanitized.upper().find(marker.upper())
    return sanitized.strip()


class ChatService:
    """AI Chat assistant service with SSE streaming support."""

    def __init__(self) -> None:
        self._conversations: Dict[str, List[Dict[str, str]]] = {}

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        vehicle_context: Optional[Dict[str, Any]] = None,
        diagnosis_id: Optional[str] = None,
    ):
        """
        Process a chat message and yield SSE stream events.

        Args:
            message: User message text.
            conversation_id: Optional existing conversation ID.
            vehicle_context: Optional vehicle context dict.
            diagnosis_id: Optional diagnosis ID for additional context.

        Yields:
            Dict representing ChatStreamEvent data for SSE serialization.
        """
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid4())

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Yield start event
        yield {
            "event_type": "start",
            "data": {
                "message": "Chat inditasa...",
                "conversation_id": conversation_id,
            },
            "conversation_id": conversation_id,
            "timestamp": timestamp,
        }

        try:
            # Sanitize user input against prompt injection
            safe_message = _sanitize_input(message)

            # Build context-enriched prompt
            system_prompt = CHAT_SYSTEM_PROMPT
            user_prompt = await self._build_user_prompt(safe_message, vehicle_context, diagnosis_id)

            # Evict oldest conversations if at capacity
            if (
                conversation_id not in self._conversations
                and len(self._conversations) >= MAX_CONVERSATIONS
            ):
                oldest_key = next(iter(self._conversations))
                del self._conversations[oldest_key]

            # Store conversation history
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []

            self._conversations[conversation_id].append({"role": "user", "content": user_prompt})

            # Include recent conversation history in prompt
            history_prompt = self._build_history_prompt(conversation_id)
            if history_prompt:
                user_prompt = f"{history_prompt}\n\nFelhasznalo: {user_prompt}"

            # Stream LLM response
            if not is_llm_available():
                logger.warning("No LLM provider available, using fallback response")
                fallback_content = (
                    "Sajnos jelenleg nem erheto el az AI szolgaltatas. "
                    "Kerem, probalkozzon kesobb ujra, vagy hasznaljon "
                    "konkret DTC kodot a diagnosztika oldalon."
                )
                yield {
                    "event_type": "token",
                    "data": {"content": fallback_content},
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                # Store fallback response in conversation history
                self._conversations[conversation_id].append(
                    {"role": "assistant", "content": fallback_content}
                )
            else:
                llm = get_llm_provider()
                full_response = ""

                async for chunk in llm.generate_stream_with_system(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    config=CHAT_LLM_CONFIG,
                ):
                    if chunk.content:
                        full_response += chunk.content
                        yield {
                            "event_type": "token",
                            "data": {"content": chunk.content},
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z"),
                        }

                # Store assistant response in conversation history
                self._conversations[conversation_id].append(
                    {"role": "assistant", "content": full_response}
                )

                # Trim conversation history to last 10 messages
                if len(self._conversations[conversation_id]) > 10:
                    self._conversations[conversation_id] = self._conversations[conversation_id][
                        -10:
                    ]

            # Yield source events
            sources = await self._get_sources(vehicle_context)
            for source in sources:
                yield {
                    "event_type": "source",
                    "data": source,
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }

            # Yield suggestion events (3 follow-up questions)
            suggestions = self._generate_suggestions(message, vehicle_context)
            yield {
                "event_type": "suggestion",
                "data": {"suggestions": suggestions},
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            # Yield complete event
            yield {
                "event_type": "complete",
                "data": {
                    "message": "Valasz kesz.",
                    "conversation_id": conversation_id,
                },
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

        except Exception as e:
            logger.exception(
                "Chat processing error",
                extra={
                    "conversation_id": conversation_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            yield {
                "event_type": "error",
                "data": {
                    "error_type": "processing_error",
                    "message": "Hiba tortent a valasz generalasa soran. Kerem, probalkozzon ujra.",
                },
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

    async def _build_user_prompt(
        self,
        message: str,
        vehicle_context: Optional[Dict[str, Any]],
        diagnosis_id: Optional[str],
    ) -> str:
        """Build the user prompt enriched with vehicle and diagnosis context."""
        context_parts: List[str] = []

        # Add vehicle context if provided
        if vehicle_context:
            make = vehicle_context.get("make", "")
            model = vehicle_context.get("model", "")
            year = vehicle_context.get("year", "")
            dtc_codes = vehicle_context.get("dtc_codes", [])

            vehicle_info = f"Jarmu: {make} {model} ({year})"
            if dtc_codes:
                vehicle_info += f"\nDTC kodok: {', '.join(dtc_codes)}"
                rag_context = await self._fetch_rag_context(dtc_codes)
                if rag_context:
                    context_parts.append(rag_context)
            context_parts.append(vehicle_info)

        # Fetch existing diagnosis context if diagnosis_id provided
        if diagnosis_id:
            diag_context = await self._fetch_diagnosis_context(diagnosis_id)
            if diag_context:
                context_parts.append(diag_context)

        if context_parts:
            context_block = "\n\n".join(context_parts)
            return f"Kontextus:\n{context_block}\n\nFelhasznalo kerdese: {message}"
        return message

    async def _fetch_rag_context(self, dtc_codes: List[str]) -> Optional[str]:
        """Fetch DTC context from RAG service for the given codes."""
        try:
            from app.services.rag_service import get_rag_service

            rag_service = get_rag_service()
            context_lines: List[str] = []

            for code in dtc_codes[:5]:  # Limit to 5 codes
                results = await rag_service.retrieve_from_qdrant(
                    query=code, collection="dtc_codes", top_k=3
                )
                if results:
                    summaries = [
                        str(r.content.get("description", r.content)) for r in results if r.content
                    ]
                    context_lines.append(f"DTC {code}: {'; '.join(summaries)}")

            if context_lines:
                return "Diagnosztikai kontextus:\n" + "\n".join(context_lines)
        except Exception as e:
            logger.warning(
                "Failed to fetch RAG context for chat",
                extra={"dtc_codes": dtc_codes, "error": str(e)},
            )
        return None

    async def _fetch_diagnosis_context(self, diagnosis_id: str) -> Optional[str]:
        """Fetch existing diagnosis context by ID."""
        try:
            from app.db.postgres.session import async_session_maker

            async with async_session_maker() as db:
                from app.db.postgres.repositories import DiagnosisSessionRepository

                repo = DiagnosisSessionRepository(db)
                from uuid import UUID

                session = await repo.get(UUID(diagnosis_id))
                if session:
                    parts: List[str] = [
                        f"Korabbi diagnozis: {session.vehicle_make} "
                        f"{session.vehicle_model} ({session.vehicle_year})",
                    ]
                    if session.dtc_codes:
                        parts.append(f"DTC kodok: {', '.join(session.dtc_codes)}")
                    if session.symptoms_text:
                        parts.append(f"Tunetek: {session.symptoms_text}")
                    return "\n".join(parts)
        except Exception as e:
            logger.warning(
                "Failed to fetch diagnosis context for chat",
                extra={"diagnosis_id": diagnosis_id, "error": str(e)},
            )
        return None

    async def _get_sources(self, vehicle_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate source references based on context."""
        sources: List[Dict[str, Any]] = [
            {
                "title": "OBD-II DTC Adatbazis",
                "type": "database",
                "relevance_score": 0.9,
            },
        ]
        if vehicle_context and vehicle_context.get("dtc_codes"):
            sources.append(
                {
                    "title": "NHTSA Visszahivas Adatbazis",
                    "type": "recall",
                    "relevance_score": 0.7,
                }
            )
        return sources

    def _generate_suggestions(
        self,
        message: str,
        vehicle_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate 3 follow-up question suggestions in Hungarian."""
        suggestions = list(DEFAULT_SUGGESTIONS)

        # Add context-specific suggestions if vehicle info is available
        if vehicle_context:
            dtc_codes = vehicle_context.get("dtc_codes", [])
            if dtc_codes:
                suggestions[0] = f"Mi okozhatja meg a {dtc_codes[0]} hibakodot?"

        return suggestions[:3]

    def _build_history_prompt(self, conversation_id: str) -> Optional[str]:
        """Build conversation history prompt from stored messages."""
        history = self._conversations.get(conversation_id, [])
        # Skip the last message (current one) and use up to 6 previous messages
        previous = history[:-1][-6:] if len(history) > 1 else []
        if not previous:
            return None

        lines: List[str] = ["Korabbi beszelgetes:"]
        for msg in previous:
            role_label = "Felhasznalo" if msg["role"] == "user" else "Asszisztens"
            # Truncate long messages in history
            content = _sanitize_input(msg["content"][:300])
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)


# =============================================================================
# Singleton
# =============================================================================

_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get the singleton ChatService instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
