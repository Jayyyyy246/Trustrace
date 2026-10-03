# TRUSTTRACE REST API Specification (v1)

**Document Version:** 1.0.0  
**Base URL:** `/api/v1`  
**OpenAPI Specification:** Available at `http://localhost:8000/docs` (Swagger UI) or `/redoc` (ReDoc)  
**Content-Type:** `application/json` (except binary evidence upload and PDF download)  

---

## 1. Overview & Architectural Principles

The TRUSTTRACE REST API provides programmatic access to evidence intake, cryptographic verification, multi-analyzer forensic processing, and report generation. 

### Key Characteristics:
- **Stateless Operation:** Requests contain all necessary query parameters or multipart payloads.
- **Strict Data Contracts:** All input and output schemas are validated against Pydantic models.
- **Honest Metrics:** Operational statistics reflect actual storage records without simulated numbers.
- **Chain of Custody:** Cryptographic SHA-256 and MD5 hashes accompany every evidence record.

---

## 2. Global Error Envelope

When an error occurs, the API returns a structured error object along with standard HTTP status codes:

```json
{
  "error": {
    "code": "EVIDENCE_NOT_FOUND",
    "message": "Evidence ID 'ev_abc123' was not found in storage.",
    "path": "/api/v1/evidence/ev_abc123"
  },
  "detail": "Evidence ID 'ev_abc123' was not found in storage."
}
```

| HTTP Status | Meaning | Common Scenarios |
|---|---|---|
| `200 OK` | Request succeeded | Retrieval of analysis, report, or status |
| `201 Created` | Resource created | Successful evidence ingestion and upload |
| `400 Bad Request` | Validation failure | Unsupported MIME type, corrupted image bytes, missing file |
| `404 Not Found` | Resource not found | Invalid or non-existent `evidence_id` |
| `500 Server Error` | Unhandled failure | System error (logged securely in backend logs) |

---

## 3. Endpoints

### 3.1. System Health Check
`GET /api/v1/health`

Returns operational readiness of the backend and all forensic sub-modules.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "storage_healthy": true,
  "quarantine_count": 14,
  "analyzers": {
    "metadata": "available",
    "image_forensics": "available",
    "screenshot_geometry": "available",
    "ocr": "available",
    "ml_inference": "standby"
  },
  "timestamp": "2026-10-03T07:45:00Z"
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

---

### 3.2. Dashboard Metrics
`GET /api/v1/evidence/dashboard/stats`

Calculates authentic operational statistics strictly from actual files in quarantine storage.

**Response (200 OK):**
```json
{
  "total_evidence_analyzed": 14,
  "verdict_distribution": {
    "REAL": 5,
    "EDITED": 4,
    "AI_GENERATED": 2,
    "SCREENSHOT_MANIPULATED": 1,
    "UNKNOWN": 2
  },
  "average_confidence": 0.842,
  "tamper_detection_rate": 0.50,
  "storage_enclave_status": "LOCKED_READ_ONLY"
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence/dashboard/stats"
```

---

### 3.3. List Registered Evidence
`GET /api/v1/evidence`

Returns all active and quarantined evidence records currently cataloged in the system.

**Response (200 OK):**
```json
[
  {
    "evidence_id": "ev_7b5f10a8c2d4",
    "filename": "suspect_contract_scan.png",
    "size": 412850,
    "mime_type": "image/png",
    "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
    "md5": "d41d8cd98f00b204e9800998ecf8427e",
    "dimensions": { "width": 1920, "height": 1080 },
    "created_at": "2026-10-03T07:12:00Z"
  }
]
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence"
```

---

### 3.4. Upload Evidence File
`POST /api/v1/evidence/upload`

Uploads an image file to the quarantine enclave. If `auto_analyze=True` (default), immediately runs the full multi-modal forensic pipeline and returns the complete analysis.

**Query Parameters:**
- `auto_analyze` (boolean, optional, default: `true`): When true, returns `AnalysisResultResponse`; when false, returns `EvidenceUploadResponse`.

**Request:** `multipart/form-data` with form field `file`.

**Response (201 Created):** Returns full `AnalysisResultResponse` (see section 3.5).

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/evidence/upload?auto_analyze=true" \
  -H "Accept: application/json" \
  -F "file=@/path/to/suspect_image.jpg"
```

---

### 3.5. Retrieve Forensic Analysis
`GET /api/v1/evidence/{evidence_id}/analysis`

Retrieves or executes a comprehensive forensic examination for the given evidence asset.

**Response (200 OK):**
```json
{
  "analysis_id": "an_91a0b3c5e7",
  "pipeline_version": "1.0.0",
  "created_at": "2026-10-03T07:12:05Z",
  "evidence": {
    "evidence_id": "ev_7b5f10a8c2d4",
    "filename": "suspect_image.jpg",
    "size": 412850,
    "mime_type": "image/jpeg",
    "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
    "dimensions": { "width": 1920, "height": 1080 }
  },
  "analyzers": {
    "metadata": {
      "status": "COMPLETED",
      "exif_present": true,
      "camera_make": "Nikon",
      "camera_model": "D850",
      "editing_software_detected": true,
      "software": "Adobe Photoshop 2026"
    },
    "image": {
      "status": "COMPLETED",
      "ela_variance": 142.5,
      "estimated_jpeg_quality": 85,
      "laplacian_variance": 42.1,
      "is_blurry": false
    },
    "screenshot": {
      "status": "COMPLETED",
      "is_likely_screenshot": false,
      "matched_profile": null
    },
    "ocr": {
      "status": "COMPLETED",
      "engine": "tesseract",
      "word_count": 0
    },
    "ml": {
      "model_status": "NOT_AVAILABLE",
      "predicted_label": "UNKNOWN",
      "confidence": null,
      "uncertainty": null
    }
  },
  "findings": [
    {
      "analyzer": "metadata_analyzer",
      "code": "META-SOFTWARE-EDITED",
      "title": "Editing Software Metadata Signature",
      "description": "Container metadata specifies editing tool: 'Adobe Photoshop 2026'.",
      "severity": "HIGH",
      "category": "METADATA_TAMPERING",
      "interpretation": "Container preserves evidence of post-processing software."
    }
  ],
  "final_verdict": {
    "label": "EDITED",
    "confidence": 0.88,
    "uncertainty": 0.12,
    "risk_score": 0.85,
    "justification": "Forensic evidence indicates post-capture editing: metadata records 'Adobe Photoshop 2026' and ELA variance (142.5) indicates localized compression disparity.",
    "conflict_detected": false,
    "supporting_findings": [
      {
        "source": "metadata_analyzer",
        "metric": "Software Tag",
        "direction": "EDITED",
        "reliability": 0.90,
        "explanation": "Software tag 'Adobe Photoshop 2026' indicates post-processing."
      }
    ],
    "contradictory_findings": [],
    "limitations": [
      "Container metadata may be retained from unedited donor capture."
    ]
  },
  "limitations": [
    "Source image provenance cannot be cryptographically verified without original camera hardware signature."
  ]
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence/ev_7b5f10a8c2d4/analysis"
```

---

### 3.6. Stream Evidence File
`GET /api/v1/evidence/{evidence_id}/file`

Securely streams raw quarantined evidence bytes for browser preview.

**Response (200 OK):** Binary image stream (`image/jpeg`, `image/png`, etc.).

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence/ev_7b5f10a8c2d4/file" \
  --output preview.jpg
```

---

### 3.7. Generate Forensic Report (JSON or PDF)
`GET /api/v1/evidence/{evidence_id}/report`

Generates an official 17-section forensic evidence audit report.

**Query Parameters:**
- `format` (string, optional, default: `"json"`): Either `"json"` or `"pdf"`.

**Response (200 OK):**
- When `format=json`: Structured JSON containing all 17 numbered report sections.
- When `format=pdf`: Binary stream with `Content-Type: application/pdf` and `Content-Disposition: attachment; filename="TRUSTTRACE_REPORT_...pdf"`.

**cURL Example (JSON):**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence/ev_7b5f10a8c2d4/report?format=json"
```

**cURL Example (Download PDF):**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence/ev_7b5f10a8c2d4/report?format=pdf" \
  --output forensic_report.pdf
```

---

### 3.8. Dedicated PDF Report Download Endpoint
`GET /api/v1/evidence/{evidence_id}/report/pdf`

Dedicated endpoint for streaming official vectorized A4 forensic investigation PDF reports.

**Response (200 OK):** Binary PDF data (`application/pdf`).

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/evidence/ev_7b5f10a8c2d4/report/pdf" \
  --output forensic_report.pdf
```
