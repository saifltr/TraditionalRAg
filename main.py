"""
FastAPI Backend for RAG Application
"""

import os
import json
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Mistral imports
try:
    from mistralai_azure import MistralAzure
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# httpx for Vexoo SRA streaming
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from vector_store import VectorStore
from llm_inference import LLMInference

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Document Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelType(str, Enum):
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_HAIKU = "claude-haiku"
    MISTRAL = "mistral"
    VEXOO_SRA = "vexoo-sra"

# Initialize vector store with top_k=3
vector_store = VectorStore(
    chunk_size=1000,
    chunk_overlap=100,
    top_k=3
)

# Store LLM instances
llm_instances = {}
mistral_client = None

def get_llm_instance(model: ModelType):
    """Get or create LLM instance."""
    model_key = model.value
    
    if model_key not in llm_instances:
        try:
            if model == ModelType.CLAUDE_SONNET:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found")
                
                llm_instances[model_key] = LLMInference(
                    api_key=api_key,
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=500,
                    history_file=f"qa_history_sonnet.json"
                )
                
            elif model == ModelType.CLAUDE_HAIKU:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found")
                
                llm_instances[model_key] = LLMInference(
                    api_key=api_key,
                    model="claude-3-5-haiku-20241022",
                    max_tokens=500,
                    history_file=f"qa_history_haiku.json"
                )
            
        except Exception as e:
            logger.error(f"Error initializing {model_key}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return llm_instances[model_key]


def get_mistral_client():
    """Get or create Mistral client."""
    global mistral_client
    
    if not MISTRAL_AVAILABLE:
        raise HTTPException(status_code=501, detail="Mistral not installed")
    
    if not mistral_client:
        endpoint = os.getenv("AZURE_AI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY")
        
        if not endpoint or not api_key:
            raise HTTPException(status_code=500, detail="Azure credentials not found")
        
        mistral_client = MistralAzure(
            azure_endpoint=endpoint,
            azure_api_key=api_key
        )
    
    return mistral_client


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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RAG Document Assistant API",
        "version": "1.0.0"
    }


@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...)
):
    """Upload multiple documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_docs = []
    errors = []
    
    for uploaded_file in files:
        try:
            file_ext = Path(uploaded_file.filename).suffix.lower()
            supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls", ".pptx", ".ppt"]
            
            if file_ext not in supported_extensions:
                errors.append({
                    "filename": uploaded_file.filename,
                    "error": f"Unsupported file type: {file_ext}"
                })
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await uploaded_file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                result = vector_store.add_document(tmp_path)
                processed_docs.append({
                    "filename": uploaded_file.filename,
                    "document_id": result["document_id"],
                    "chunk_count": result["chunk_count"],
                    "total_characters": result["total_characters"]
                })
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.filename}: {e}")
            errors.append({
                "filename": uploaded_file.filename,
                "error": str(e)
            })
    
    if not processed_docs and errors:
        raise HTTPException(status_code=400, detail=f"Failed to process documents: {errors}")
    
    return {
        "success": True,
        "message": f"Processed {len(processed_docs)} document(s)",
        "documents_processed": len(processed_docs),
        "documents": processed_docs,
        "errors": errors
    }


@app.post("/chat")
async def chat_stream(
    query: str = Query(...),
    model: ModelType = Query(...),
    use_history: bool = Query(True)
):
    """Chat endpoint with streaming."""
    stats = vector_store.get_stats()
    if stats["total_documents"] == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    
    context, sources = vector_store.get_context_for_llm(query)
    
    if not context:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    async def generate_stream():
        """Generate SSE stream."""
        try:
            # Small delay before streaming starts
            await asyncio.sleep(2)
            
            if model == ModelType.MISTRAL:
                # Mistral streaming
                client = get_mistral_client()
                
                system_prompt = "Answer the question using the context provided. Keep answers brief and direct."
                user_message = f"Context:\n{context}\n\nQuestion: {query}"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                stream = client.chat.stream(
                    messages=messages,
                    model="azureai"
                )
                
                if stream:
                    for event in stream:
                        try:
                            if event.data.choices and len(event.data.choices) > 0:
                                delta = event.data.choices[0].delta
                                if delta.content:
                                    yield delta.content
                        except:
                            continue
            
            elif model == ModelType.VEXOO_SRA:
                # Vexoo SRA streaming via /v1/chat SSE endpoint
                if not HTTPX_AVAILABLE:
                    yield "Error: httpx package not installed. Run: pip install httpx"
                    return
                
                config = get_vexoo_sra_config()
                
                system_prompt = "Answer the question using the context provided. Keep answers brief and direct."
                user_message = f"Context:\n{context}\n\nQuestion: {query}"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                payload = {
                    "messages": messages,
                    "max_tokens": config["max_tokens"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "repetition_penalty": config["repetition_penalty"],
                    "stream": True
                }
                
                logger.info(f"[VexooSRA-RAG] Streaming request to {config['base_url']}/v1/chat")
                
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config["timeout"], connect=30.0)
                ) as client:
                    async with client.stream(
                        "POST",
                        f"{config['base_url']}/v1/chat",
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                    ) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_msg = f"Vexoo SRA returned status {response.status_code}: {error_body.decode()[:500]}"
                            logger.error(error_msg)
                            yield f"Error: {error_msg}"
                            return
                        
                        # Parse SSE stream: data: {"token": "word "}
                        buffer = ""
                        async for chunk in response.aiter_text():
                            buffer += chunk
                            
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                
                                if not line:
                                    continue
                                
                                if line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    if not data_str:
                                        continue
                                    
                                    try:
                                        data = json.loads(data_str)
                                        
                                        if "token" in data:
                                            yield data["token"]
                                        elif "done" in data and data["done"]:
                                            return
                                        elif "error" in data:
                                            logger.error(f"Vexoo SRA stream error: {data['error']}")
                                            yield f"\n\nError from model: {data['error']}"
                                            return
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Process remaining buffer
                        if buffer.strip():
                            line = buffer.strip()
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:].strip())
                                    if "token" in data:
                                        yield data["token"]
                                except json.JSONDecodeError:
                                    pass
            
            else:
                # Claude streaming (Sonnet / Haiku)
                llm = get_llm_instance(model)
                
                for chunk in llm.generate_response_stream(query, context, sources, use_history):
                    yield chunk
            
        except httpx.ConnectError as e:
            logger.error(f"Could not connect to Vexoo SRA: {e}")
            yield f"Error: Could not connect to Vexoo SRA server. Is it running?"
        except httpx.TimeoutException as e:
            logger.error(f"Vexoo SRA request timed out: {e}")
            yield f"Error: Request to Vexoo SRA timed out."
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"Error: {str(e)}"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics."""
    return vector_store.get_stats()


@app.get("/token-usage/{model}")
async def get_token_usage(model: ModelType):
    """Get token usage for model."""
    try:
        if model in (ModelType.MISTRAL, ModelType.VEXOO_SRA):
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0
            }
        
        llm = get_llm_instance(model)
        summary = llm.get_token_summary()
        cumulative = summary.get("cumulative_token_usage", {})
        
        return {
            "total_input_tokens": cumulative.get("total_input_tokens", 0),
            "total_output_tokens": cumulative.get("total_output_tokens", 0),
            "total_tokens": cumulative.get("total_tokens", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vector-store/clear")
async def clear_vector_store():
    """Clear vector store."""
    try:
        vector_store.clear()
        return {"success": True, "message": "Vector store cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/clear/{model}")
async def clear_chat_history(model: ModelType):
    """Clear chat history."""
    try:
        if model not in (ModelType.MISTRAL, ModelType.VEXOO_SRA):
            llm = get_llm_instance(model)
            llm.clear_history()
        return {"success": True, "message": f"Chat history cleared for {model.value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check."""
    health = {
        "status": "healthy",
        "vector_store_documents": vector_store.get_stats()["total_documents"],
        "vector_store_chunks": vector_store.get_stats()["total_chunks"]
    }
    
    # Optionally check Vexoo SRA connectivity
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