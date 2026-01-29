"""
LLM Inference Module for RAG Application
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInference:
    """Handles Claude API inference."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 500,
        history_file: Optional[str] = None
    ):
        """Initialize LLM inference."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        self.history_file = Path(history_file) if history_file else Path("qa_history.json")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.history = self._load_history()
        self.chat_messages: List[Dict[str, str]] = []
        
        logger.info(f"LLM Inference initialized: model={model}")
    
    def _get_system_prompt(self) -> str:
        """Get simple system prompt."""
        return """Answer the question using the context provided. Keep answers brief and direct."""

    def _load_history(self) -> Dict[str, Any]:
        """Load history from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "queries": [],
            "cumulative_token_usage": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0
            }
        }
    
    def _save_history(self):
        """Save history to JSON file."""
        try:
            self.history["last_updated"] = datetime.now().isoformat()
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def _add_to_chat_history(self, role: str, content: str):
        """Add message to chat history (keep last 4 messages)."""
        self.chat_messages.append({"role": role, "content": content})
        if len(self.chat_messages) > 4:
            self.chat_messages = self.chat_messages[-4:]
    
    def _add_to_history(self, question: str, answer: str, token_usage: Dict[str, int], sources: List[str]):
        """Add Q&A to persistent history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "token_usage": token_usage,
            "sources_used": sources
        }
        
        self.history["queries"].append(entry)
        self.history["cumulative_token_usage"]["total_input_tokens"] += token_usage.get("input_tokens", 0)
        self.history["cumulative_token_usage"]["total_output_tokens"] += token_usage.get("output_tokens", 0)
        self.history["cumulative_token_usage"]["total_tokens"] += token_usage.get("total_tokens", 0)
        
        self._save_history()
    
    def generate_response_stream(
        self,
        question: str,
        context: str,
        sources: List[Dict[str, Any]],
        use_history: bool = True
    ) -> Generator[str, None, None]:
        """Generate streaming response."""
        user_message = f"""Context:
{context}

Question: {question}"""

        full_response = ""
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        source_files = list(set(s.get("filename", "unknown") for s in sources))
        
        try:
            messages = []
            if use_history and self.chat_messages:
                messages.extend(self.chat_messages)
            messages.append({"role": "user", "content": user_message})
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._get_system_prompt(),
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield text
                
                final_message = stream.get_final_message()
                if final_message and hasattr(final_message, "usage"):
                    token_usage = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens
                    }
            
            self._add_to_chat_history("user", question)
            self._add_to_chat_history("assistant", full_response)
            self._add_to_history(question, full_response, token_usage, source_files)
            
            self._last_result = {
                "question": question,
                "answer": full_response,
                "token_usage": token_usage,
                "sources": source_files
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
            self._last_result = {
                "question": question,
                "answer": f"Error: {str(e)}",
                "token_usage": token_usage,
                "sources": []
            }
    
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get last streaming result."""
        return getattr(self, "_last_result", None)
    
    def get_token_summary(self) -> Dict[str, Any]:
        """Get cumulative token usage."""
        return {
            "session_id": self.history.get("session_id"),
            "total_queries": len(self.history.get("queries", [])),
            "cumulative_token_usage": self.history.get("cumulative_token_usage", {}),
            "history_file": str(self.history_file)
        }
    
    def clear_history(self):
        """Clear all history."""
        self.history = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(),
            "queries": [],
            "cumulative_token_usage": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0
            }
        }
        self.chat_messages = []
        self._save_history()
        logger.info("History cleared")