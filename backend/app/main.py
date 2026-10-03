"""
TRUSTTRACE Backend Application.
AI-Assisted Digital Evidence Verification & Tamper Detection Platform.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import TrustTraceException
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Production-grade, academic cybersecurity platform for digital evidence verification, "
        "forensic feature extraction, tamper detection, and explainable multi-criteria assessment."
    ),
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(TrustTraceException)
async def trusttrace_exception_handler(request: Request, exc: TrustTraceException):
    """Custom handler ensuring consistent, structured error envelopes."""
    logger.warning(f"Domain exception on {request.url.path}: {exc.message} [{exc.code}]")
    status_code = status.HTTP_400_BAD_REQUEST
    if exc.code == "EVIDENCE_NOT_FOUND":
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code == "INTERNAL_ERROR":
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "path": request.url.path,
            },
            "detail": exc.message,
        },
    )


# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", summary="Root Health Check")
def root_redirect():
    """Root redirect with basic service information."""
    return {
        "service": settings.PROJECT_NAME,
        "description": settings.PROJECT_DESCRIPTION,
        "version": settings.VERSION,
        "documentation": "/docs",
        "api_v1_prefix": settings.API_V1_PREFIX,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
