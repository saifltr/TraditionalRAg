"""
Google Knowledge Base ingestion.

Downloads documents from a Google Drive folder, a Google Drive file, or a direct
URL (including .zip archives) into a session's docs directory. Mirrors the
document-URL handling of the Knowledge Distillation RAG: gdown handles Drive's
large-file confirmation pages and folder traversal; ZIPs are extracted.

Public, link-shared Drive resources ("Anyone with the link") work without auth.
"""

import asyncio
import io
import logging
import os
import re
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
    ".csv", ".xlsx", ".xls", ".pptx", ".ppt",
}


def _safe_name(name: str, fallback_ext: str = "") -> str:
    base, ext = os.path.splitext(os.path.basename(name))
    ext = (ext or fallback_ext).lower()
    safe = re.sub(r"[^\w\-]", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")[:100] or "document"
    return f"{safe}_{uuid.uuid4().hex[:8]}{ext}"


def _is_gdrive(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "drive.google.com" in host or "docs.google.com" in host


def _is_gdrive_folder(url: str) -> bool:
    return "/folders/" in url


def _extract_gdrive_file_id(url: str) -> str:
    # .../file/d/<id>/...  or  ...?id=<id>
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    qs = parse_qs(urlparse(url).query)
    if "id" in qs:
        return qs["id"][0]
    m = re.search(r"/d/([^/]+)", url)
    return m.group(1) if m else ""


def _save_supported_file(content: bytes, filename: str, docs_dir: Path) -> List[Dict]:
    """Save one file, extracting it first if it is a ZIP archive."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".zip" or content[:4] == b"PK\x03\x04":
        return _extract_zip(content, docs_dir)

    if ext not in SUPPORTED_EXTENSIONS:
        logger.info(f"Skipping unsupported file: {filename}")
        return []

    safe = _safe_name(filename)
    (docs_dir / safe).write_bytes(content)
    return [{"filename": safe, "source": filename, "size_bytes": len(content)}]


def _extract_zip(content: bytes, docs_dir: Path) -> List[Dict]:
    saved: List[Dict] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                base = os.path.basename(member)
                if not base or base.startswith(".") or base.startswith("~$"):
                    continue
                if os.path.splitext(base)[1].lower() not in SUPPORTED_EXTENSIONS:
                    continue
                data = zf.read(member)
                safe = _safe_name(base)
                (docs_dir / safe).write_bytes(data)
                saved.append({"filename": safe, "source": member, "size_bytes": len(data)})
    except zipfile.BadZipFile as e:
        raise ValueError(f"Invalid ZIP archive: {e}")
    return saved


def _download_gdrive_folder(url: str, docs_dir: Path) -> List[Dict]:
    import gdown
    import tempfile

    saved: List[Dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            paths = gdown.download_folder(url=url, output=tmp, quiet=True, use_cookies=False)
        except Exception as e:
            raise ValueError(
                f"gdown failed to download folder. Ensure it is shared "
                f"('Anyone with the link'). Error: {e}"
            )
        for p in paths or []:
            if os.path.isfile(p):
                saved.extend(_save_supported_file(Path(p).read_bytes(), os.path.basename(p), docs_dir))
    return saved


def _download_gdrive_file(url: str, docs_dir: Path) -> List[Dict]:
    import gdown
    import tempfile

    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError(f"Could not extract a Google Drive file id from: {url}")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            out = gdown.download(id=file_id, output=tmp + os.sep, quiet=True, use_cookies=False)
        except Exception as e:
            raise ValueError(f"Google Drive file download failed: {e}")
        if not out or not os.path.isfile(out):
            raise ValueError(
                "Google Drive file download failed: ensure the file is shared "
                "('Anyone with the link')."
            )
        return _save_supported_file(Path(out).read_bytes(), os.path.basename(out), docs_dir)


def _download_direct_url(url: str, docs_dir: Path) -> List[Dict]:
    import httpx

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        content = resp.content

    # Derive a filename from the URL path; fall back to a generated name.
    name = os.path.basename(urlparse(url).path) or "document"
    return _save_supported_file(content, name, docs_dir)


def _ingest_one(url: str, docs_dir: Path) -> List[Dict]:
    url = url.strip()
    if not url:
        return []
    if _is_gdrive(url):
        if _is_gdrive_folder(url):
            return _download_gdrive_folder(url, docs_dir)
        return _download_gdrive_file(url, docs_dir)
    return _download_direct_url(url, docs_dir)


def parse_urls(raw: str) -> List[str]:
    """Split a comma/newline-separated string of URLs into a clean list."""
    parts = re.split(r"[,\n]+", raw or "")
    return [p.strip() for p in parts if p.strip()]


async def ingest_google_urls(urls: List[str], docs_dir: str) -> List[Dict]:
    """Download all ``urls`` into ``docs_dir``. Returns saved-file descriptors.

    Runs the synchronous gdown / httpx work in a thread so the event loop is
    not blocked.
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()
    saved: List[Dict] = []
    for url in urls:
        files = await loop.run_in_executor(None, _ingest_one, url, docs_path)
        saved.extend(files)
        logger.info(f"Ingested {len(files)} file(s) from {url}")
    return saved
