"""TRUSTTRACE Model Inference CLI Script.

Executes digital evidence visual artifact inference and returns structured JSON conforming
strictly to the TRUSTTRACE specification.

Usage:
    python predict.py --image path/to/evidence.jpg --model-path model/checkpoints/trusttrace-v1.0.0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model.predictor import (
    ChecksumMismatchError,
    CorruptImageError,
    ForensicPredictor,
    ModelNotFoundError,
    UnsupportedModelError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Forensic Model Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to evidence image file")
    parser.add_argument("--model-path", type=str, default="model/checkpoints/trusttrace-v1.0.0.pt", help="Path to model weights checkpoint")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to model metadata JSON")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Override minimum confidence threshold for UNKNOWN")
    parser.add_argument("--entropy-threshold", type=float, default=None, help="Override maximum normalized entropy threshold for UNKNOWN")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--output", type=str, default=None, help="Optional output path to write JSON result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model_path)

    try:
        predictor = ForensicPredictor(
            model_path=model_path,
            metadata_path=args.metadata_path,
            device=args.device,
            verify_checksum=True,
        )

        result = predictor.predict(
            image_input=image_path,
            confidence_threshold=args.confidence_threshold,
            entropy_threshold=args.entropy_threshold,
        )

        # Output to stdout
        json_output = json.dumps(result, indent=2)
        print(json_output)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(json_output)

        return 0

    except ModelNotFoundError as e:
        err_res = {
            "model_version": "unknown",
            "prediction": "UNKNOWN",
            "probabilities": {},
            "uncertainty": 1.0,
            "status": "error",
            "error_detail": f"ModelNotFoundError: {e}",
        }
        print(json.dumps(err_res, indent=2), file=sys.stderr)
        return 1

    except ChecksumMismatchError as e:
        err_res = {
            "model_version": "unknown",
            "prediction": "UNKNOWN",
            "probabilities": {},
            "uncertainty": 1.0,
            "status": "error",
            "error_detail": f"ChecksumMismatchError: {e}",
        }
        print(json.dumps(err_res, indent=2), file=sys.stderr)
        return 2

    except CorruptImageError as e:
        err_res = {
            "model_version": getattr(predictor, "architecture", "unknown") if "predictor" in locals() else "unknown",
            "prediction": "UNKNOWN",
            "probabilities": {},
            "uncertainty": 1.0,
            "status": "error",
            "error_detail": f"CorruptImageError: {e}",
        }
        print(json.dumps(err_res, indent=2), file=sys.stderr)
        return 3

    except UnsupportedModelError as e:
        err_res = {
            "model_version": "unknown",
            "prediction": "UNKNOWN",
            "probabilities": {},
            "uncertainty": 1.0,
            "status": "error",
            "error_detail": f"UnsupportedModelError: {e}",
        }
        print(json.dumps(err_res, indent=2), file=sys.stderr)
        return 4

    except Exception as e:
        err_res = {
            "model_version": "unknown",
            "prediction": "UNKNOWN",
            "probabilities": {},
            "uncertainty": 1.0,
            "status": "error",
            "error_detail": str(e),
        }
        print(json.dumps(err_res, indent=2), file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
