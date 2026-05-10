from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core.brain_system import BrainSystem

logger = logging.getLogger(__name__)

_brain_system: Optional[BrainSystem] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入消息")
    source: str = Field(default="user", description="消息来源标识")
    context: Optional[Dict[str, str]] = Field(default=None, description="附加上下文")
    emotional_feedback: float = Field(default=0.0, ge=-1.0, le=1.0, description="情感反馈")


class ChatResponse(BaseModel):
    response: str
    memory_trace_id: str
    reconstruction_id: str
    importance: float
    novelty: float
    emotional_valence: float
    memory_confidence: float
    distortion_score: float
    neuromodulatory_state: Dict[str, Any]


class RecallRequest(BaseModel):
    query: str = Field(..., description="回忆查询")
    source: str = Field(default="user", description="查询来源")


class GoalsRequest(BaseModel):
    goals: List[str] = Field(..., description="目标列表")


class ConfigRequest(BaseModel):
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_type: Optional[str] = None
    system_identity: Optional[str] = None
    consolidation_interval: Optional[int] = None


def create_app(
    llm_base_url: str = "http://localhost:11434",
    llm_model: str = "qwen2.5:7b",
    llm_api_type: str = "ollama",
    data_dir: str = "./neurocortex_data",
    system_identity: str = "",
    consolidation_interval: int = 10,
) -> FastAPI:
    """
    Create the FastAPI application for the NeuroCortex brain system.

    This API provides the interface for a robot to interact with
    the brain-inspired memory system. All endpoints follow REST
    conventions and return JSON responses.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _brain_system
        _brain_system = BrainSystem(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_type=llm_api_type,
            data_dir=data_dir,
            system_identity=system_identity,
            consolidation_interval=consolidation_interval,
        )
        await _brain_system.load_state()
        logger.info("NeuroCortex brain system initialized")
        yield
        await _brain_system.shutdown()
        logger.info("NeuroCortex brain system shut down")

    app = FastAPI(
        title="NeuroCortex",
        description=(
            "Brain-Inspired Episodic Memory System for Embodied LLM Agents. "
            "Implements multi-region brain architecture with reconstructive recall, "
            "replay-based consolidation, and neuromodulatory gating."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Main interaction endpoint. Send a message and receive a
        brain-memory-informed response.

        The system will:
        1. Detect novelty of the input
        2. Assess importance via amygdala
        3. Encode into hippocampal episodic memory
        4. Perform reconstructive recall
        5. Generate response with memory context
        6. Update neuromodulatory state
        """
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        result = await _brain_system.process_input(
            user_message=request.message,
            source=request.source,
            context=request.context,
            emotional_feedback=request.emotional_feedback,
        )

        return ChatResponse(**result)

    @app.post("/api/memory/recall")
    async def recall_memory(request: RecallRequest):
        """
        Explicitly recall a memory without generating a response.
        Useful for testing memory properties and debugging.
        """
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        result = await _brain_system.recall_memory(
            query=request.query,
            source=request.source,
        )
        return result

    @app.post("/api/memory/consolidate")
    async def trigger_consolidation():
        """
        Manually trigger a memory consolidation round.
        During consolidation, episodic traces are replayed and
        semantic schemas are extracted into the neocortex.
        """
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        result = await _brain_system.force_consolidation()
        return result

    @app.get("/api/memory/status")
    async def memory_status():
        """
        Get comprehensive status of all brain regions including
        hippocampal trace count, neocortical schema count,
        neuromodulatory levels, and LLM health.
        """
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        return await _brain_system.get_memory_status()

    @app.get("/api/memory/traces")
    async def list_traces(limit: int = 20, min_strength: float = 0.05):
        """List recent episodic memory traces."""
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        traces = _brain_system.hippocampus.get_active_traces(min_strength)
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return {
            "count": len(traces),
            "traces": [t.to_dict() for t in traces[:limit]],
        }

    @app.get("/api/memory/schemas")
    async def list_schemas(limit: int = 20):
        """List semantic schemas in the neocortex."""
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        schemas = _brain_system.neocortex.get_all_schemas()
        schemas.sort(key=lambda s: s.updated, reverse=True)
        return {
            "count": len(schemas),
            "schemas": [s.to_dict() for s in schemas[:limit]],
        }

    @app.post("/api/goals")
    async def set_goals(request: GoalsRequest):
        """Set or update the system's current goals."""
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        _brain_system.set_goals(request.goals)
        return {"status": "ok", "goals": request.goals}

    @app.get("/api/health")
    async def health_check():
        """Check system health including LLM backend connectivity."""
        if _brain_system is None:
            return {"status": "initializing"}

        llm_health = await _brain_system.llm_engine.check_health()
        return {
            "status": "ok" if llm_health.get("status") == "healthy" else "degraded",
            "llm": llm_health,
            "interaction_count": _brain_system._interaction_count,
        }

    @app.post("/api/state/save")
    async def save_state():
        """Manually save the brain state to disk."""
        if _brain_system is None:
            raise HTTPException(status_code=503, detail="Brain system not initialized")

        await _brain_system.save_state()
        return {"status": "saved"}

    return app
