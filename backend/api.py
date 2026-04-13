from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import logging
import os
import shutil

from contextlib import asynccontextmanager
from backend.agents.agent import graph
from backend.memory.persistent import (
    get_all_episodes,
    get_recent_episodes,
    delete_episodes_by_session,
    init_db
)
from backend.storage.lancedb_store import init_table, store_chunks
from backend.ingestion.chunker import chunk_and_embed

# Configuration du logging
logger = logging.getLogger(__name__)

# Historique des sessions en mémoire
session_histories: dict[str, list] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événements de cycle de vie de l'application."""
    logger.info("Initialisation des bases de données...")
    try:
        init_db()
        init_table()
        logger.info("Bases de données initialisées avec succès.")
    except Exception as e:
        logger.error(f"Échec de l'initialisation : {e}")
    yield
    logger.info("Arrêt de l'API...")

app = FastAPI(
    title="Assistant Médical RAG API",
    description="API pour l'Assistant de Recherche Médicale ",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent_path: list[str]

@app.get("/")
async def root():
    return {"message": "Assistant Médical RAG API v2.0 "}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint principal de chat avec le système multi-agents."""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Préparer l'état initial
        initial_state = {
            "messages": [],
            "question": request.message,
            "retrieved_docs": "",
            "external_results": "",
            "final_response": "",
            "agent_path": [],
            "routing_decision": ""
        }

        # Exécuter le graphe LangGraph
        result = graph.invoke(initial_state)

        final_response = result.get("final_response", "Désolé, je n'ai pas pu générer de réponse.")
        agent_path = result.get("agent_path", [])

        logger.info(f"[Chat] Session {session_id} - Chemin : {' → '.join(agent_path)}")

        return ChatResponse(
            session_id=session_id,
            response=final_response,
            agent_path=agent_path
        )

    except Exception as e:
        logger.error(f"Erreur Chat : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    """Ingère un fichier PDF, TXT ou MD dans LanceDB ET Neo4j."""
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
 
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
 
        # Extraction du contenu
        content = ""
        if file.filename.lower().endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(file_path)
                content = "\n".join(page.get_text() for page in doc)
                doc.close()
            except ImportError:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                content = "\n".join(page.extract_text() for page in reader.pages)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
 
        if not content.strip():
            raise HTTPException(status_code=400, detail="Fichier vide ou non lisible")
 
        # Découpage et embedding
        chunks = chunk_and_embed(content)
 
        metadata = {
            "source": file.filename,
            "title": file.filename,
            "domain": "medical"
        }
 
        # ── Stockage LanceDB ──
        count = store_chunks(chunks, metadata)
 
        # ── Stockage Neo4j ──
        try:
            from backend.storage.neo4j_store import store_document_chunks
            store_document_chunks(
                chunks=chunks,
                source=file.filename,
                title=file.filename
            )
            logger.info(f"[Neo4j] {file.filename} indexé dans le graphe")
        except Exception as e:
            logger.warning(f"[Neo4j] Indexation ignorée : {e}")
 
        # Nettoyage
        #os.remove(file_path)
 
        logger.info(f"[Ingest] {file.filename} → {count} chunks indexés")
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": count,
            "message": f"{count} fragments indexés (LanceDB + Neo4j)"
        }
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur d'ingestion : {e}")
        raise HTTPException(status_code=500, detail=str(e))
 

@app.get("/sessions")
async def list_sessions():
    """Liste toutes les sessions avec leurs épisodes."""
    try:
        episodes = get_all_episodes(limit=100)
        sessions = {}
        for ep in episodes:
            sid = ep.session_id
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "created_at": ep.created_at.isoformat() if ep.created_at else None,
                    "message_count": 0,
                    "last_query": ""
                }
            sessions[sid]["message_count"] += 1
            if not sessions[sid]["last_query"]:
                sessions[sid]["last_query"] = ep.user_query[:100]
        return list(sessions.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/messages")
async def get_session_messages_endpoint(session_id: str):
    """Récupère tous les messages d'une session."""
    try:
        episodes = get_recent_episodes(session_id, limit=50)
        messages = []
        for ep in reversed(episodes):
            messages.append({
                "id": str(ep.id),
                "user_query": ep.user_query,
                "agent_response": ep.agent_response,
                "agent_path": ep.agent_path,
                "created_at": ep.created_at.isoformat() if ep.created_at else None
            })
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Supprime une session et son historique."""
    session_histories.pop(session_id, None)
    delete_episodes_by_session(session_id)
    return {"message": f"Session {session_id} supprimée"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0", "framework": "LangGraph"}