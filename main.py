"""
FastAPI Backend for RAG Application — session-based.

Each upload creates (or reuses) a session whose documents, vector index and chat
history are persisted to a local directory under data/sessions/{session_id}/.
Chatting requires a session_id and works across three models — Claude Sonnet,
Claude Haiku, and Vexoo SRA — which all share one chat history so the user can
switch models mid-conversation.
"""

import os
import json
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# httpx for Vexoo SRA streaming
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from session_manager import SessionManager
from llm_inference import ClaudeStreamer, SYSTEM_PROMPT
from google_ingest import ingest_google_urls, parse_urls

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Document Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls", ".pptx", ".ppt"
]


class ModelType(str, Enum):
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_HAIKU = "claude-haiku"
    VEXOO_SRA = "vexoo-sra"


# Claude model IDs (Sonnet 4.6 / Haiku 4.5 — current generation)
CLAUDE_MODEL_IDS = {
    ModelType.CLAUDE_SONNET: "claude-sonnet-4-6",
    ModelType.CLAUDE_HAIKU: "claude-haiku-4-5",
}

# Session registry (vector store + chat history per session, persisted to disk)
session_manager = SessionManager(chunk_size=1000, chunk_overlap=100, top_k=3)

# One Claude streamer per model, shared across sessions (no per-conversation state)
claude_streamers = {}


def get_claude_streamer(model: ModelType) -> ClaudeStreamer:
    """Get or create the shared Claude streamer for a model."""
    if model not in claude_streamers:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found")
        try:
            claude_streamers[model] = ClaudeStreamer(
                api_key=api_key,
                model=CLAUDE_MODEL_IDS[model],
                max_tokens=500,
            )
        except Exception as e:
            logger.error(f"Error initializing {model.value}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return claude_streamers[model]


def get_vexoo_sra_config() -> dict:
    """Get Vexoo SRA configuration from environment."""
    base_url = os.getenv("VEXOO_SRA_BASE_URL", "http://20.106.211.17:8001")
    base_url = base_url.rstrip("/")
    return {
        "base_url": base_url,
        "max_tokens": int(os.getenv("VEXOO_SRA_MAX_TOKENS", "16384")),
        "temperature": float(os.getenv("VEXOO_SRA_TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("VEXOO_SRA_TOP_P", "0.9")),
        "repetition_penalty": float(os.getenv("VEXOO_SRA_REPETITION_PENALTY", "1.1")),
        "timeout": float(os.getenv("VEXOO_SRA_TIMEOUT", "180")),
    }


def _ingest_local_file(session, upload: UploadFile) -> dict:
    """Save an uploaded file into the session docs dir and index it."""
    file_ext = Path(upload.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = upload.file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    try:
        # Keep a copy of the original document inside the session
        dest = session.docs_dir / Path(upload.filename).name
        dest.write_bytes(content)
        result = session.vector_store.add_document(tmp_path)
        result["filename"] = upload.filename
        return result
    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    return {"name": "RAG Document Assistant API", "version": "2.0.0"}


@app.post("/upload-documents")
async def upload_documents(
    files: Optional[List[UploadFile]] = File(None),
    google_knowledge_base: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """Upload documents and/or ingest from a Google knowledge base.

    Creates a new session (returning its ``session_id``) unless an existing
    ``session_id`` is supplied, in which case documents are added to it.
    Provide ``files`` (multipart uploads) and/or ``google_knowledge_base``
    (a Google Drive folder/file URL, a direct/zip URL, or several
    comma/newline-separated). At least one source is required.
    """
    has_files = bool(files)
    has_google = bool(google_knowledge_base and google_knowledge_base.strip())
    if not has_files and not has_google:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: files or google_knowledge_base",
        )

    # Resume an existing session or create a new one
    if session_id:
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    else:
        session = session_manager.create_session()

    processed_docs = []
    errors = []

    # 1) Local file uploads
    for upload in files or []:
        try:
            result = _ingest_local_file(session, upload)
            processed_docs.append({
                "filename": result["filename"],
                "document_id": result["document_id"],
                "chunk_count": result["chunk_count"],
                "total_characters": result["total_characters"],
                "source": "upload",
            })
        except Exception as e:
            logger.error(f"Error processing {upload.filename}: {e}")
            errors.append({"filename": upload.filename, "error": str(e)})

    # 2) Google knowledge base URLs
    if has_google:
        urls = parse_urls(google_knowledge_base)
        try:
            saved_files = await ingest_google_urls(urls, str(session.docs_dir))
            for sf in saved_files:
                try:
                    result = session.vector_store.add_document(str(session.docs_dir / sf["filename"]))
                    processed_docs.append({
                        "filename": sf["filename"],
                        "document_id": result["document_id"],
                        "chunk_count": result["chunk_count"],
                        "total_characters": result["total_characters"],
                        "source": sf.get("source", "google_knowledge_base"),
                    })
                except Exception as e:
                    logger.error(f"Error indexing {sf['filename']}: {e}")
                    errors.append({"filename": sf["filename"], "error": str(e)})
        except Exception as e:
            logger.error(f"Google knowledge base ingestion failed: {e}")
            errors.append({"filename": "google_knowledge_base", "error": str(e)})

    if not processed_docs:
        # Nothing indexed — clean up a freshly-created empty session
        if not session_id:
            session_manager.delete_session(session.session_id)
        raise HTTPException(status_code=400, detail=f"Failed to process documents: {errors}")

    # Persist the session (index + history + metadata) to disk
    session.save_all()

    return {
        "success": True,
        "session_id": session.session_id,
        "message": f"Processed {len(processed_docs)} document(s)",
        "documents_processed": len(processed_docs),
        "documents": processed_docs,
        "errors": errors,
    }


@app.post("/chat")
async def chat_stream(
    session_id: str = Query(...),
    query: str = Query(...),
    model: ModelType = Query(...),
    use_history: bool = Query(True),
):
    """Chat over a session's documents with streaming. Requires a session_id."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    stats = session.vector_store.get_stats()
    if stats["total_documents"] == 0:
        raise HTTPException(status_code=400, detail="No documents in this session")

    context, _sources = session.vector_store.get_context_for_llm(query)
    if not context:
        raise HTTPException(status_code=404, detail="No relevant context found")

    history = session.history_messages() if use_history else []

    async def generate_stream():
        full_response = ""
        try:
            await asyncio.sleep(2)  # small delay before streaming starts

            if model == ModelType.VEXOO_SRA:
                if not HTTPX_AVAILABLE:
                    yield "Error: httpx package not installed. Run: pip install httpx"
                    return

                config = get_vexoo_sra_config()
                user_message = f"Context:\n{context}\n\nQuestion: {query}"
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(history)
                messages.append({"role": "user", "content": user_message})

                payload = {
                    "messages": messages,
                    "max_tokens": config["max_tokens"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "repetition_penalty": config["repetition_penalty"],
                    "stream": True,
                }
                logger.info(f"[VexooSRA-RAG] Streaming request to {config['base_url']}/v1/chat")

                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config["timeout"], connect=30.0)
                ) as client:
                    async with client.stream(
                        "POST",
                        f"{config['base_url']}/v1/chat",
                        json=payload,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    ) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_msg = (
                                f"Vexoo SRA returned status {response.status_code}: "
                                f"{error_body.decode()[:500]}"
                            )
                            logger.error(error_msg)
                            yield f"Error: {error_msg}"
                            return

                        buffer = ""
                        async for chunk in response.aiter_text():
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line or not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                if not data_str:
                                    continue
                                try:
                                    data = json.loads(data_str)
                                    if "token" in data:
                                        full_response += data["token"]
                                        yield data["token"]
                                    elif "done" in data and data["done"]:
                                        break
                                    elif "error" in data:
                                        logger.error(f"Vexoo SRA stream error: {data['error']}")
                                        yield f"\n\nError from model: {data['error']}"
                                        return
                                except json.JSONDecodeError:
                                    continue

            else:
                # Claude streaming (Sonnet / Haiku)
                streamer = get_claude_streamer(model)
                for text in streamer.stream(context, query, history):
                    full_response += text
                    yield text

            # Persist the turn to the shared session history (raw question stored)
            session.add_message("user", query, model.value)
            session.add_message("assistant", full_response, model.value)
            session.save_chat_history()

        except httpx.ConnectError as e:
            logger.error(f"Could not connect to Vexoo SRA: {e}")
            yield "Error: Could not connect to Vexoo SRA server. Is it running?"
        except httpx.TimeoutException as e:
            logger.error(f"Vexoo SRA request timed out: {e}")
            yield "Error: Request to Vexoo SRA timed out."
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/sessions")
async def list_sessions():
    """List all sessions known on disk."""
    return {"sessions": session_manager.list_sessions()}


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.info()


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"session_id": session_id, "chat_history": session.chat_history}


@app.delete("/sessions/{session_id}/history")
async def clear_session_history(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    session.clear_history()
    return {"success": True, "message": f"Chat history cleared for session {session_id}"}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"success": True, "message": f"Session {session_id} deleted"}


@app.get("/health")
async def health_check():
    """Health check."""
    health = {
        "status": "healthy",
        "total_sessions": len(session_manager.list_sessions()),
    }
    try:
        config = get_vexoo_sra_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config['base_url']}/health")
            health["vexoo_sra_status"] = "reachable" if resp.status_code == 200 else "error"
    except Exception:
        health["vexoo_sra_status"] = "unreachable"
    return health


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
