from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from finshield.serving.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreResponse,
    TransactionEventInput,
)
from finshield.serving.service import FraudInferenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

service_holder: dict[str, FraudInferenceService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_holder["service"] = FraudInferenceService(auto_train_if_missing=True)
    yield
    service_holder.clear()


app = FastAPI(
    title="FIN-SHIELD ML Risk & Fraud Scoring API",
    description=(
        "Drift-Resilient Transaction Fraud and Relational Risk Engine with Causal Graph Features, "
        "Conformal/Calibrated Scoring, and Cost-Aware Decisioning."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_service() -> FraudInferenceService:
    svc = service_holder.get("service")
    if svc is None:
        svc = FraudInferenceService(auto_train_if_missing=True)
        service_holder["service"] = svc
    return svc


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return get_service().health()


@app.post("/v1/score", response_model=ScoreResponse)
def score_transaction(event: TransactionEventInput) -> ScoreResponse:
    try:
        return get_service().score_single(event)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/score/batch", response_model=BatchScoreResponse)
def score_batch_transactions(request: BatchScoreRequest) -> BatchScoreResponse:
    try:
        return get_service().score_batch(request.events)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
