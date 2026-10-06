"""TRUSTTRACE System Diagnostics CLI command.
Reports authentic runtime state of OCR, ML, hardware devices, and model parameters.
"""
import sys
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.services.ocr_service import ocr_service
from app.services.ml_inference_service import ml_inference_service


def run_diagnostics():
    ocr_avail = ocr_service.is_engine_available()
    ocr_engine = ocr_service.get_engine_name()
    ml_diag = ml_inference_service.get_diagnostics()

    print("=" * 60)
    print("TRUSTTRACE SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print(f"OCR:              {'AVAILABLE' if ocr_avail else 'NOT_AVAILABLE'}")
    print(f"OCR Engine:       {ocr_engine}")
    print(f"ML:               {ml_diag['status']}")
    print(f"Checkpoint:       {ml_diag['checkpoint_path'] or 'None (Awaiting training dataset)'}")
    print(f"Checkpoint Hash:  {ml_diag['checkpoint_sha256'] or 'N/A'}")
    print(f"Model:            {ml_diag['model_name']}")
    print(f"Model Version:    {ml_diag['model_version'] or 'N/A'}")
    print(f"Device:           {ml_diag['device']}")
    print(f"Calibration:      {ml_diag['calibration']}")
    print(f"OOD:              {ml_diag['ood_detection']}")
    print(f"MC Dropout:       {ml_diag['mc_dropout']}")
    if ml_diag.get("reason"):
        print(f"ML Diagnostic:    {ml_diag['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
