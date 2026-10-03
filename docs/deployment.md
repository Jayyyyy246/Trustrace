# TRUSTTRACE Production Deployment & Operational Engineering Guide

**Document Version:** 1.0.0  
**Target Environments:** Linux (Ubuntu 22.04 LTS / Debian 12), Windows Server 2022 / Windows 11, Containerized (Docker / Kubernetes)  
**Security Standard:** CIS Benchmarks, NIST SP 800-123 (Guide to General Server Security)  

---

## 1. System Requirements & Hardware Sizing

TRUSTTRACE performs compute-intensive digital signal processing (8×8 DCT grid recompression, discrete quantization table parsing, SIFT/ORB feature descriptor matching, and neural network inference).

| Component | Minimum Specification | Recommended Production Specification |
|---|---|---|
| **Processor (CPU)** | 4 Cores (x86_64 or ARM64) | 8+ Cores (e.g., AMD EPYC / Intel Xeon / Apple Silicon) |
| **System Memory (RAM)** | 8 GB RAM | 16 GB - 32 GB RAM (accommodates high-res image decodes) |
| **Storage (Disk)** | 20 GB SSD | 250+ GB NVMe SSD (fast I/O for quarantined evidence vault) |
| **Graphics (GPU)** | Not required (CPU inference supported) | NVIDIA CUDA GPU (8+ GB VRAM, Turing architecture or newer) |
| **Operating System** | Linux (Ubuntu 22.04+), Windows 10/11, macOS 13+ | Linux (Ubuntu 22.04 LTS or Debian 12) |
| **Runtimes** | Python 3.10 - 3.14, Node.js 18+ | Python 3.11+, Node.js 20 LTS |

---

## 2. Environment Variables & Configuration Matrix

All configuration parameters are managed via environment variables or a `.env` file located in the project root:

| Variable | Type | Default | Description |
|---|---|---|---|
| `PORT` | integer | `8000` | Port for the FastAPI backend service |
| `HOST` | string | `0.0.0.0` | Host binding interface for FastAPI |
| `TRUSTTRACE_STORAGE_DIR` | path | `data/quarantine` | Absolute or relative path to the quarantined evidence vault |
| `TRUSTTRACE_MODEL_PATH` | path | `null` | Path to trained PyTorch model checkpoint (`.pt` or `.pth`) |
| `TRUSTTRACE_MODEL_METADATA` | path | `null` | Path to JSON metadata manifest containing verified model checksum |
| `CORS_ORIGINS` | list | `["*"]` | Allowed CORS origins (must be locked to specific origins in prod) |
| `LOG_LEVEL` | string | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MAX_UPLOAD_SIZE_BYTES` | integer | `52428800` | Maximum single evidence payload size (50 MB) |

---

## 3. Standalone Native Deployment

### Step 1: Backend Setup
```bash
# 1. Clone repository
git clone https://github.com/Jayyyyy246/Trustrace.git
cd Trustrace

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 3. Install backend dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Optional: Install system OCR engine
# On Ubuntu/Debian: sudo apt-get install -y tesseract-ocr
# On Windows: Install via official installer and add to PATH

# 5. Launch FastAPI backend with production ASGI server
export PYTHONPATH="backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 2: Frontend Production Build
```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Build production assets
npm run build

# Assets are emitted to frontend/dist/ ready for web-server delivery.
```

---

## 4. Production Reverse Proxy Configuration (Nginx)

In production, place FastAPI behind Nginx to handle SSL termination, client payload limits, and high-performance static asset caching.

```nginx
server {
    listen 80;
    server_name forensics.trusttrace.internal;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name forensics.trusttrace.internal;

    ssl_certificate /etc/ssl/certs/trusttrace.crt;
    ssl_certificate_key /etc/ssl/private/trusttrace.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Enforce upload size bound matching application limit (50 MB)
    client_max_body_size 50M;

    # Frontend Static Distribution
    location / {
        root /var/www/trusttrace/frontend/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API Proxy to Uvicorn Backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Swagger / OpenAPI Documentation
    location ~ ^/(docs|redoc|openapi.json) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
```

---

## 5. Containerized Deployment (Docker Compose)

### Dockerfile (`Dockerfile.production`)
```dockerfile
# Multi-stage build for production containerization
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS production
WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY model/ ./model/
COPY --from=frontend-builder /app/frontend/dist /app/static

ENV PYTHONPATH="/app/backend"
ENV TRUSTTRACE_STORAGE_DIR="/app/data/quarantine"
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 6. Health Monitoring & Operational Verification

### Automated Health Polling
Configure monitoring systems (e.g., Datadog, Prometheus Blackbox, CloudWatch) to probe:
```bash
curl -f -s http://localhost:8000/api/v1/health | grep '"status":"healthy"'
```

### Log Rotation & Audit Custody
Configure `logrotate` to prevent storage exhaustion while preserving audit history:
```
/var/log/trusttrace/*.log {
    daily
    missingok
    rotate 90
    compress
    delaycompress
    notifempty
    create 0640 trusttrace trusttrace
}
```
