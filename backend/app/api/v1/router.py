"""
Master API v1 router consolidating all sub-endpoints.
"""
from fastapi import APIRouter
from app.api.v1.endpoints import health, evidence

api_router = APIRouter()
api_router.include_router(health.router, tags=["System & Health"])
api_router.include_router(evidence.router, tags=["Digital Evidence"])
