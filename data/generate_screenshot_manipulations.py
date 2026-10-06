"""TRUSTTRACE Controlled Screenshot Manipulation Dataset Generator.

Generates rigorous, documented forensic manipulation samples from authentic base screenshots.
Each sample records full cryptographic and transformation provenance:
- source_image_id
- source_image_sha256
- transformation_type
- transformation_parameters
- generated_image_sha256

All samples are explicitly stamped as synthetic training data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.deduplication import compute_sha256


TRANSFORMATION_TYPES = [
    "numeric_alteration",
    "text_alteration",
    "timestamp_modification",
    "ui_element_splicing",
    "ui_region_compositing",
]


def apply_numeric_alteration(
    img: Image.Image,
    target_box: Optional[Tuple[int, int, int, int]] = None,
    new_text: Optional[str] = None,
    variation_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Alters a numeric transaction amount region."""
    tampered = img.copy()
    draw = ImageDraw.Draw(tampered)
    w, h = tampered.size

    amounts = [
        "$4,850.00", "$12,450.00", "$780.50", "$3,210.99", "$95,000.00",
        "$1,890.25", "$6,400.00", "$24,500.00", "$520.00", "$8,750.30"
    ]
    if new_text is None:
        new_text = amounts[variation_seed % len(amounts)]

    # Default to an upper-middle region if not specified, slightly offset by seed
    if target_box is None:
        bx = int(w * 0.25)
        by = int(h * (0.32 + (variation_seed % 4) * 0.04))
        bw = int(w * 0.50)
        bh = int(h * 0.08)
    else:
        bx, by, bw, bh = target_box

    # Sample background color from edge of region
    bg_color = tampered.getpixel((bx + 2, by + 2))
    draw.rectangle([bx, by, bx + bw, by + bh], fill=bg_color)

    # Render altered numeric text
    draw.text((bx + 8, by + int(bh * 0.2)), new_text, fill=(20, 20, 20))

    params = {
        "transformation": "numeric_alteration",
        "bounding_box": [bx, by, bw, bh],
        "injected_text": new_text,
        "fill_color": bg_color if isinstance(bg_color, (list, tuple)) else [bg_color],
        "variation_seed": variation_seed,
    }
    return tampered, params


def apply_text_alteration(
    img: Image.Image,
    target_box: Optional[Tuple[int, int, int, int]] = None,
    replacement_text: Optional[str] = None,
    variation_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Replaces text with synthetic overlay introducing baseline/kerning drift."""
    tampered = img.copy()
    draw = ImageDraw.Draw(tampered)
    w, h = tampered.size

    texts = [
        "Payment Approved - Verified Account",
        "Wire Transfer Completed - Cleared",
        "Authorization Code Verified: AUTH-9921",
        "Direct Deposit Confirmed by System",
        "Funds Released to Recipient Account",
        "Merchant Transaction Approved #8812",
    ]
    if replacement_text is None:
        replacement_text = texts[variation_seed % len(texts)]

    if target_box is None:
        bx = int(w * 0.15)
        by = int(h * (0.42 + (variation_seed % 4) * 0.04))
        bw = int(w * 0.70)
        bh = int(h * 0.06)
    else:
        bx, by, bw, bh = target_box

    bg_color = tampered.getpixel((bx + 1, by + 1))
    draw.rectangle([bx, by, bx + bw, by + bh], fill=bg_color)

    # Introduce subtle vertical baseline displacement
    offset_y = by + int(bh * 0.15) + (variation_seed % 3 + 1)
    draw.text((bx + 5, offset_y), replacement_text, fill=(30, 30, 30))

    params = {
        "transformation": "text_alteration",
        "bounding_box": [bx, by, bw, bh],
        "replacement_text": replacement_text,
        "baseline_displacement_px": (variation_seed % 3 + 1),
        "variation_seed": variation_seed,
    }
    return tampered, params


def apply_timestamp_modification(
    img: Image.Image,
    target_box: Optional[Tuple[int, int, int, int]] = None,
    new_timestamp: Optional[str] = None,
    variation_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Modifies timestamp or status bar time."""
    tampered = img.copy()
    draw = ImageDraw.Draw(tampered)
    w, h = tampered.size

    timestamps = [
        "Today at 09:41 AM",
        "Oct 03, 2026 14:22:10 UTC",
        "Yesterday at 11:58 PM",
        "Sep 30, 2026 08:15 AM",
        "Just now (0.2s ago)",
        "09:14 AM - Complete",
    ]
    if new_timestamp is None:
        new_timestamp = timestamps[variation_seed % len(timestamps)]

    if target_box is None:
        bx = int(w * 0.10)
        by = int(h * (0.04 + (variation_seed % 3) * 0.02))
        bw = int(w * 0.40)
        bh = int(h * 0.04)
    else:
        bx, by, bw, bh = target_box

    bg_color = tampered.getpixel((bx + 1, by + 1))
    draw.rectangle([bx, by, bx + bw, by + bh], fill=bg_color)
    draw.text((bx + 2, by + int(bh * 0.1)), new_timestamp, fill=(90, 90, 95))

    params = {
        "transformation": "timestamp_modification",
        "bounding_box": [bx, by, bw, bh],
        "new_timestamp": new_timestamp,
        "variation_seed": variation_seed,
    }
    return tampered, params


def apply_ui_element_splicing(
    img: Image.Image,
    source_box: Optional[Tuple[int, int, int, int]] = None,
    dest_pos: Optional[Tuple[int, int]] = None,
    variation_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Copies a UI badge/element from one location and splices it elsewhere."""
    tampered = img.copy()
    w, h = tampered.size

    badges = ["VERIFIED", "APPROVED", "SECURE", "OFFICIAL", "PRIORITY"]
    badge_text = badges[variation_seed % len(badges)]

    if source_box is None:
        sx = int(w * 0.10)
        sy = int(h * 0.10)
        sw = int(w * 0.20)
        sh = int(h * 0.10)
    else:
        sx, sy, sw, sh = source_box

    if dest_pos is None:
        dx = int(w * (0.50 + (variation_seed % 3) * 0.10))
        dy = int(h * (0.65 + (variation_seed % 4) * 0.05))
    else:
        dx, dy = dest_pos

    # Crop source patch and composite a spliced UI badge
    patch = tampered.crop((sx, sy, sx + sw, sy + sh))
    draw_patch = ImageDraw.Draw(patch)
    draw_patch.rectangle([0, 0, sw - 1, sh - 1], fill=(225, 238, 255), outline=(40, 110, 220), width=2)
    draw_patch.text((4, 4), badge_text, fill=(20, 80, 180))
    tampered.paste(patch, (dx, dy))

    params = {
        "transformation": "ui_element_splicing",
        "source_crop": [sx, sy, sw, sh],
        "destination_pos": [dx, dy],
        "spliced_element": f"{badge_text.lower()}_badge",
        "variation_seed": variation_seed,
    }
    return tampered, params


def apply_ui_region_compositing(
    img: Image.Image,
    target_box: Optional[Tuple[int, int, int, int]] = None,
    variation_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Blanks out a credential or confirmation card and re-renders altered details."""
    tampered = img.copy()
    draw = ImageDraw.Draw(tampered)
    w, h = tampered.size

    ref_num = f"TR-{990000 + variation_seed * 1337 % 9999}"
    auth_code = f"{400 + variation_seed * 17 % 500}-{100 + variation_seed * 23 % 800}-{800 + variation_seed * 31 % 199}"

    if target_box is None:
        bx = int(w * 0.10)
        by = int(h * (0.55 + (variation_seed % 3) * 0.04))
        bw = int(w * 0.80)
        bh = int(h * 0.25)
    else:
        bx, by, bw, bh = target_box

    # Draw rounded rectangle composite card
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=10, fill=(245, 247, 250), outline=(210, 215, 225), width=2)
    draw.text((bx + 16, by + 16), "TRANSACTION COMPLETED", fill=(10, 120, 60))
    draw.text((bx + 16, by + 48), f"Reference: {ref_num}", fill=(60, 60, 60))
    draw.text((bx + 16, by + 76), f"Auth Code: {auth_code}", fill=(100, 100, 100))

    params = {
        "transformation": "ui_region_compositing",
        "bounding_box": [bx, by, bw, bh],
        "composited_elements": ["status_header", "reference_number", "auth_code"],
        "reference_number": ref_num,
        "auth_code": auth_code,
        "variation_seed": variation_seed,
    }
    return tampered, params


def generate_manipulation(
    image_path: Path,
    output_dir: Path,
    transformation: str,
    output_prefix: str = "synth_tamper",
    variation_idx: int = 0,
) -> Dict[str, Any]:
    """Applies a specified transformation and saves the resulting image with provenance metadata."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image '{image_path}' not found.")

    with Image.open(image_path) as src:
        src = src.convert("RGB")
        src_sha256 = compute_sha256(image_path)

        if transformation == "numeric_alteration":
            tampered, params = apply_numeric_alteration(src, variation_seed=variation_idx)
        elif transformation == "text_alteration":
            tampered, params = apply_text_alteration(src, variation_seed=variation_idx)
        elif transformation == "timestamp_modification":
            tampered, params = apply_timestamp_modification(src, variation_seed=variation_idx)
        elif transformation == "ui_element_splicing":
            tampered, params = apply_ui_element_splicing(src, variation_seed=variation_idx)
        elif transformation == "ui_region_compositing":
            tampered, params = apply_ui_region_compositing(src, variation_seed=variation_idx)
        else:
            raise ValueError(f"Unsupported transformation: '{transformation}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{output_prefix}_{transformation}_v{variation_idx}_{src_sha256[:8]}_{int(datetime.now().timestamp())}_{variation_idx}.png"
    out_path = output_dir / out_filename
    tampered.save(out_path, format="PNG")

    gen_sha256 = compute_sha256(out_path)

    provenance = {
        "generated_image_id": out_filename,
        "generated_image_path": str(out_path.resolve()),
        "generated_image_sha256": gen_sha256,
        "source_image_id": image_path.name,
        "source_image_path": str(image_path.resolve()),
        "source_image_sha256": src_sha256,
        "target_class": "SCREENSHOT-MANIPULATED",
        "transformation_type": transformation,
        "transformation_parameters": params,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_tag": "SYNTHETIC_FORENSIC_TAMPERING_CONTROLLED",
        "disclaimer": "This sample was synthetically generated under controlled forensic testing protocols and must not be conflated with organic captures.",
    }
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Screenshot Manipulation Dataset Generator")
    parser.add_argument("--source-image", type=str, required=True, help="Path to base authentic screenshot")
    parser.add_argument("--output-dir", type=str, default="data/dataset/SCREENSHOT-MANIPULATED", help="Output directory")
    parser.add_argument("--transformation", type=str, choices=TRANSFORMATION_TYPES + ["all"], default="all", help="Transformation type")
    parser.add_argument("--provenance-log", type=str, default="data/manifests/screenshot_manipulation_provenance.json", help="Path to provenance log")
    args = parser.parse_args()

    src_path = Path(args.source_image)
    out_dir = Path(args.output_dir)
    prov_path = Path(args.provenance_log)
    prov_path.parent.mkdir(parents=True, exist_ok=True)

    tf_list = TRANSFORMATION_TYPES if args.transformation == "all" else [args.transformation]
    print(f"Generating {len(tf_list)} controlled screenshot manipulation(s) from '{src_path}'...")

    existing_logs: List[Dict[str, Any]] = []
    if prov_path.is_file():
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        except Exception:
            existing_logs = []

    for tf in tf_list:
        record = generate_manipulation(src_path, out_dir, tf)
        existing_logs.append(record)
        print(f"  [+] {tf}: {record['generated_image_id']} (SHA256: {record['generated_image_sha256'][:12]}...)")

    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, indent=2)

    print(f"Provenance recorded to: {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
