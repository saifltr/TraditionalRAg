"""
Claude inference for the RAG application.

Stateless with respect to conversation history: the caller (a Session) owns the
chat history and passes prior messages in on each call. One ClaudeStreamer
instance per model is reused across all sessions — it only holds the API client
and model configuration, no per-conversation state.
"""

import logging
from typing import Any, Dict, Generator, List, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer the question using the context provided. Keep answers brief and direct."


class ClaudeStreamer:
    """Thin wrapper around the Anthropic streaming API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 500,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        if not api_key:
            raise ValueError("Anthropic API key required")

        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"ClaudeStreamer initialized: model={model}")

    def stream(
        self,
        context: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        usage_out: Optional[Dict[str, int]] = None,
    ) -> Generator[str, None, None]:
        """Stream a response. ``usage_out`` (if given) is filled with token usage.

        ``history`` is a list of prior {role, content} messages; the current
        question is appended (with the retrieved context) as the final user turn.
        """
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages: List[Dict[str, str]] = list(history or [])
        messages.append({"role": "user", "content": user_message})

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

                if usage_out is not None:
                    final = stream.get_final_message()
                    if final and hasattr(final, "usage"):
                        usage_out["input_tokens"] = final.usage.input_tokens
                        usage_out["output_tokens"] = final.usage.output_tokens
                        usage_out["total_tokens"] = (
                            final.usage.input_tokens + final.usage.output_tokens
                        )
        except Exception as e:
            logger.error(f"Error in Claude streaming: {e}")
            yield f"Error: {str(e)}"
