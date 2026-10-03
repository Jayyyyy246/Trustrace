"""
Evidence endpoints for secure upload, ingestion, analysis execution, and reporting.
"""
from typing import Union, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from fastapi.responses import Response

from app.schemas.evidence import EvidenceUploadResponse
from app.schemas.analysis import AnalysisResultResponse
from app.services.evidence_service import evidence_service
from app.services.report_service import report_service
from app.core.exceptions import EvidenceValidationError, EvidenceNotFoundError

router = APIRouter()


@router.get(
    "/evidence/dashboard/stats",
    summary="Retrieve Real Operational Dashboard Metrics",
)
def get_dashboard_stats() -> Dict[str, Any]:
    """Returns authentic operational metrics calculated strictly from actual backend data."""
    return evidence_service.get_dashboard_metrics()


@router.get(
    "/evidence",
    response_model=List[EvidenceUploadResponse],
    summary="List All Registered Evidence Records",
)
def list_evidence() -> List[EvidenceUploadResponse]:
    """Lists all active and quarantined digital evidence files in the system."""
    return evidence_service.list_all_evidence()


@router.post(
    "/evidence/upload",
    response_model=Union[AnalysisResultResponse, EvidenceUploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Quarantined Intake of Digital Evidence",
)
async def upload_evidence(
    file: UploadFile = File(..., description="Binary evidence file (JPEG, PNG, WebP, TIFF, BMP)"),
    auto_analyze: bool = Query(
        default=True,
        description="When true, immediately executes forensic analysis and returns full assessment."
    ),
):
    """
    Ingests digital evidence into a secure quarantine store:
    1. Validates file size and magic-byte signatures
    2. Computes cryptographic SHA-256 and MD5 hashes
    3. Assigns an immutable Evidence ID
    4. Automatically executes full deterministic analysis if auto_analyze=True
    """
    try:
        content = await file.read()
        intake_record = evidence_service.ingest_evidence(
            file_bytes=content,
            original_filename=file.filename or "evidence",
        )

        if auto_analyze:
            analysis_result = evidence_service.run_analysis(intake_record.evidence_id)
            return analysis_result

        return intake_record

    except EvidenceValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence processing failed: {str(e)}",
        )


@router.get(
    "/evidence/{evidence_id}/file",
    summary="Stream Quarantined Evidence File for Forensic Inspection",
)
def get_evidence_file(evidence_id: str):
    """Securely streams quarantined evidence bytes for browser preview."""
    target_id = evidence_id
    if not evidence_service.storage.exists(target_id):
        for cached in evidence_service._analysis_cache.values():
            if cached.analysis_id == evidence_id and cached.evidence.evidence_id:
                if evidence_service.storage.exists(cached.evidence.evidence_id):
                    target_id = cached.evidence.evidence_id
                    break

    if not evidence_service.storage.exists(target_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence ID '{evidence_id}' was not found in storage.",
        )
    data = evidence_service.storage.read_bytes(target_id)
    record = evidence_service.get_evidence_record(target_id)
    mime = record.mime_type if record else "application/octet-stream"
    return Response(content=data, media_type=mime)


@router.get(
    "/evidence/{evidence_id}",
    response_model=EvidenceUploadResponse,
    summary="Retrieve Stored Evidence Intake Record",
)
def get_evidence(evidence_id: str):
    """Retrieves metadata and cryptographic hash for a previously uploaded evidence file."""
    record = evidence_service.get_evidence_record(evidence_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence ID '{evidence_id}' was not found in storage.",
        )
    return record


@router.get(
    "/evidence/{evidence_id}/analysis",
    response_model=AnalysisResultResponse,
    summary="Retrieve or Execute Comprehensive Forensic Analysis",
)
def get_evidence_analysis(evidence_id: str):
    """
    Returns the structured forensic analysis result:
    - Metadata and container analysis
    - Computer vision & ELA metrics
    - Typography / OCR metrics
    - Screenshot geometry validation
    - Machine learning status
    - Itemized findings
    - Calibrated verdict and operational limitations
    """
    try:
        return evidence_service.run_analysis(evidence_id)
    except EvidenceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence ID '{evidence_id}' was not found.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}",
        )


@router.get(
    "/evidence/{evidence_id}/report",
    summary="Generate Forensic Audit Digest Report",
)
def get_evidence_report(
    evidence_id: str,
    format: str = Query(default="json", description="Report format: 'json' or 'pdf'"),
):
    """Returns a structured, exportable forensic audit digest with chain-of-custody data, or binary PDF."""
    try:
        analysis = evidence_service.run_analysis(evidence_id)
        if format.lower() == "pdf":
            pdf_bytes = report_service.generate_pdf_report(analysis)
            filename = f"TRUSTTRACE_REPORT_{evidence_id[:12].upper()}.pdf"
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(pdf_bytes)),
                },
            )
        return report_service.generate_summary_digest(analysis)
    except EvidenceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence ID '{evidence_id}' was not found.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation error: {str(e)}",
        )


@router.get(
    "/evidence/{evidence_id}/report/pdf",
    summary="Download Official PDF Investigation Report",
)
def download_evidence_report_pdf(evidence_id: str):
    """Generates and streams an official vectorized A4 digital evidence examination PDF report."""
    try:
        analysis = evidence_service.run_analysis(evidence_id)
        pdf_bytes = report_service.generate_pdf_report(analysis)
        filename = f"TRUSTTRACE_REPORT_{evidence_id[:12].upper()}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except EvidenceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence ID '{evidence_id}' was not found.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report PDF generation error: {str(e)}",
        )

