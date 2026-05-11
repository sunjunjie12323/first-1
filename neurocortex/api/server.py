from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neurocortex.config import Config
from neurocortex.core.brain_system import BrainSystem
from neurocortex.core.memory_trace import ContextTag
from neurocortex.core.theory import PatternSeparationTheory, ReconstructiveDistortionTheory

logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroCortex", version="0.1.0")

brain: Optional[BrainSystem] = None


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    social_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "user"


class ChatResponse(BaseModel):
    response: str
    trace_id: Optional[str] = None
    importance: float = 0.0
    novelty_score: float = 0.0
    encoding_gate: float = 0.0
    recall_confidence: float = 0.0
    distortion_score: float = 0.0
    neuromodulatory_state: Dict[str, Any] = {}


class RecallRequest(BaseModel):
    query: str


class RecallResponse(BaseModel):
    reconstruction_id: str
    query: str
    reconstructed_narrative: str
    source_traces: List[str] = []
    source_schemas: List[str] = []
    confidence: float = 0.0
    distortion_score: float = 0.0
    emotional_tone: float = 0.0


class StatusResponse(BaseModel):
    total_traces: int
    total_schemas: int
    phase_distribution: Dict[str, int]
    avg_memory_strength: float
    avg_importance: float
    working_memory_load: int
    working_memory_capacity: int
    current_goal: Optional[str]
    neuromodulatory_state: Dict[str, Any]
    hippocampus_epsilon: float


class TheoryEpsilonResponse(BaseModel):
    optimal_epsilon: float
    diagnostics: Dict[str, Any]


class TheoryDistortionResponse(BaseModel):
    distortion_score: float
    schacter_sins: Dict[str, float]


class TheoryBridgeResponse(BaseModel):
    results: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    llm_available: bool


class ConsolidateResponse(BaseModel):
    consolidated: int
    forgotten: int
    schemas_created: int


@app.on_event("startup")
async def startup():
    global brain
    config = Config()
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))

    brain = BrainSystem(
        llm_base_url=config.LLM_BASE_URL,
        llm_model=config.LLM_MODEL,
        embedding_model=config.EMBEDDING_MODEL,
        api_type=config.API_TYPE,
        api_key=config.API_KEY,
        epsilon=config.EPSILON,
        working_memory_capacity=config.WORKING_MEMORY_CAPACITY,
    )
    logger.info("NeuroCortex brain system initialized")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    context = None
    if request.context:
        context = ContextTag(
            spatial=request.context.get("spatial"),
            temporal_period=request.context.get("temporal_period"),
            interlocutor=request.context.get("interlocutor"),
            activity=request.context.get("activity"),
            modality=request.context.get("modality"),
        )

    result = await brain.process_input(
        content=request.message,
        context=context,
        source=request.source,
        emotional_valence=request.emotional_valence,
        social_relevance=request.social_relevance,
        goal_relevance=request.goal_relevance,
    )

    return ChatResponse(**result)


@app.post("/api/memory/recall", response_model=RecallResponse)
async def recall(request: RecallRequest):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    result = await brain.recall_memory(request.query)
    return RecallResponse(
        reconstruction_id=result.reconstruction_id,
        query=result.query,
        reconstructed_narrative=result.reconstructed_narrative,
        source_traces=result.source_traces,
        source_schemas=result.source_schemas,
        confidence=result.confidence,
        distortion_score=result.distortion_score,
        emotional_tone=result.emotional_tone,
    )


@app.post("/api/memory/consolidate", response_model=ConsolidateResponse)
async def consolidate():
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    result = await brain.force_consolidation()
    return ConsolidateResponse(**result)


@app.get("/api/memory/status", response_model=StatusResponse)
async def status():
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    return StatusResponse(**brain.get_memory_status())


@app.get("/api/memory/traces")
async def list_traces(limit: int = 50, offset: int = 0):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    traces = list(brain.hippocampus.traces.values())
    traces.sort(key=lambda t: t.timestamp, reverse=True)
    paginated = traces[offset:offset + limit]
    return {"traces": [t.to_dict() for t in paginated], "total": len(traces)}


@app.get("/api/memory/schemas")
async def list_schemas(limit: int = 50, offset: int = 0):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    schemas = list(brain.neocortex.schemas.values())
    schemas.sort(key=lambda s: s.confidence, reverse=True)
    paginated = schemas[offset:offset + limit]
    return {"schemas": [s.to_dict() for s in paginated], "total": len(schemas)}


@app.get("/api/theory/optimal-epsilon", response_model=TheoryEpsilonResponse)
async def optimal_epsilon(n_trials: int = 3):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    result = brain.compute_optimal_epsilon(n_trials=n_trials)
    return TheoryEpsilonResponse(**result)


@app.get("/api/theory/distortion-analysis", response_model=TheoryDistortionResponse)
async def distortion_analysis():
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    result = brain.analyze_distortion()
    return TheoryDistortionResponse(**result)


@app.get("/api/theory/encoding-recall-bridge", response_model=TheoryBridgeResponse)
async def encoding_recall_bridge(n_trials: int = 3):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    result = brain.measure_encoding_recall_bridge(n_trials=n_trials)
    return TheoryBridgeResponse(**result)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    if brain is None:
        return HealthResponse(status="not_initialized", llm_available=False)

    llm_ok = await brain.llm_engine.health_check()
    return HealthResponse(status="ok" if llm_ok else "degraded", llm_available=llm_ok)
