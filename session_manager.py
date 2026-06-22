"""
Session Manager for the RAG Application.

Each session owns a directory on local disk:

    data/sessions/{session_id}/
    ├── docs/                 # the original uploaded / downloaded files
    ├── index/
    │   ├── faiss.index       # FAISS index (binary)
    │   └── store.json        # chunks + document metadata
    ├── chat_history.json     # single shared history; each message tagged with model
    └── session_state.json    # session metadata

Sessions are held in memory for speed and lazily rehydrated from disk on a miss
(e.g. after a server restart). There is NO background flushing / blob sync — the
local volume is the single source of truth, persisted synchronously on write.
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vector_store import VectorStore

logger = logging.getLogger(__name__)

# Root for all session data (lives on the mounted Docker volume).
SESSIONS_BASE_DIR = Path("data/sessions")
CHAT_HISTORY_FILE = "chat_history.json"
SESSION_STATE_FILE = "session_state.json"
DOCS_DIRNAME = "docs"
INDEX_DIRNAME = "index"


class Session:
    """In-memory state for a single session."""

    def __init__(self, session_id: str, vector_store: VectorStore):
        self.session_id = session_id
        self.vector_store = vector_store
        # Each entry: {"role", "content", "model", "timestamp"}
        self.chat_history: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self._lock = threading.Lock()

    # --- path helpers -----------------------------------------------------
    @property
    def dir(self) -> Path:
        return SESSIONS_BASE_DIR / self.session_id

    @property
    def docs_dir(self) -> Path:
        return self.dir / DOCS_DIRNAME

    @property
    def index_dir(self) -> Path:
        return self.dir / INDEX_DIRNAME

    # --- chat history -----------------------------------------------------
    def history_messages(self, limit: int = 8) -> List[Dict[str, str]]:
        """Return the last ``limit`` turns as plain {role, content} messages.

        Model-agnostic: history written by any model is replayed to whichever
        model is chatting now, so the user can switch models mid-conversation.
        """
        with self._lock:
            recent = self.chat_history[-limit:]
            return [{"role": m["role"], "content": m["content"]} for m in recent]

    def add_message(self, role: str, content: str, model: str) -> None:
        with self._lock:
            self.chat_history.append(
                {
                    "role": role,
                    "content": content,
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def clear_history(self) -> None:
        with self._lock:
            self.chat_history = []
        self.save_chat_history()

    # --- persistence ------------------------------------------------------
    def save_chat_history(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            history = list(self.chat_history)
        try:
            with open(self.dir / CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save chat history for {self.session_id}: {e}")

    def save_state(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat(),
            "documents": list(self.vector_store.documents.values()),
        }
        try:
            with open(self.dir / SESSION_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save session state for {self.session_id}: {e}")

    def save_all(self) -> None:
        """Persist vector index, chat history and metadata together."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(str(self.index_dir))
        self.save_chat_history()
        self.save_state()

    def info(self) -> Dict[str, Any]:
        stats = self.vector_store.get_stats()
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "chat_history_length": len(self.chat_history),
            "documents": stats["documents"],
        }


class SessionManager:
    """Thread-safe registry of sessions with lazy disk recovery."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, top_k: int = 3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        SESSIONS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("SessionManager initialized")

    def _new_vector_store(self) -> VectorStore:
        return VectorStore(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=self.top_k,
        )

    def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create a brand-new session and its directory."""
        session_id = session_id or f"session_{uuid.uuid4().hex}"
        session = Session(session_id, self._new_vector_store())
        session.docs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return a session, lazily loading it from disk if needed."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            return session
        return self._load_from_disk(session_id)

    def _load_from_disk(self, session_id: str) -> Optional[Session]:
        session_dir = SESSIONS_BASE_DIR / session_id
        if not session_dir.exists():
            return None

        vector_store = self._new_vector_store()
        vector_store.load(str(session_dir / INDEX_DIRNAME))  # ok if empty

        session = Session(session_id, vector_store)

        state_path = session_dir / SESSION_STATE_FILE
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                session.created_at = state.get("created_at", session.created_at)
            except Exception as e:
                logger.warning(f"Could not read session state for {session_id}: {e}")

        chat_path = session_dir / CHAT_HISTORY_FILE
        if chat_path.exists():
            try:
                with open(chat_path, "r", encoding="utf-8") as f:
                    session.chat_history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read chat history for {session_id}: {e}")

        with self._lock:
            self._sessions[session_id] = session
        logger.info(f"Session recovered from disk: {session_id}")
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List sessions known on disk (the durable source of truth)."""
        sessions = []
        if not SESSIONS_BASE_DIR.exists():
            return sessions
        for item in sorted(SESSIONS_BASE_DIR.iterdir()):
            if item.is_dir():
                session = self.get_session(item.name)
                if session:
                    sessions.append(session.info())
        return sessions

    def delete_session(self, session_id: str) -> bool:
        import shutil

        with self._lock:
            self._sessions.pop(session_id, None)
        session_dir = SESSIONS_BASE_DIR / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
