#!/usr/bin/env python3
"""
MediaPipe Landmark Extraction Feasibility Test for ASL-HG (v2 — actual layout).
Samples images from each class, runs HandLandmarker, reports per-class success
and per-participant success (useful for later signer-independent splits).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

RAW_ROOT = Path("tools/datasets/asl_hg/asl_dataset")
MODEL_PATH = Path("tools/hand_landmarker.task")
OUTPUT_REPORT = Path("tools/artifacts/landmark_feasibility.json")
IMG_EXTS = (".jpg", ".jpeg", ".png")
PARTICIPANT_RE = re.compile(r"^P(\d+)_")


def extract_participant(filename: str):
    m = PARTICIPANT_RE.match(filename)
    return f"P{m.group(1)}" if m else "unknown"


def create_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path: Path, landmarker):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "read_failed"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None, "no_hand_detected"
    lm = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    return coords, "success"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Optional class whitelist, e.g. A B L V Y")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not RAW_ROOT.exists():
        print(f"[ERROR] Dataset not found: {RAW_ROOT}")
        sys.exit(1)

    landmarker = create_landmarker()
    all_class_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()])
    class_dirs = all_class_dirs
    if args.classes:
        wanted = {c.upper() for c in args.classes}
        class_dirs = [d for d in all_class_dirs if d.name.upper() in wanted]
        if not class_dirs:
            print(f"[ERROR] No matching classes for {args.classes}")
            sys.exit(1)

    results = {}
    total_attempted = 0
    total_success = 0

    for cls_dir in tqdm(class_dirs, desc="Classes"):
        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        if not images:
            continue
        sampled = images[: args.samples_per_class]
        per_participant = defaultdict(lambda: {"attempted": 0, "success": 0})
        fail_reasons = defaultdict(int)
        cls_success = 0

        for img_path in sampled:
            total_attempted += 1
            p = extract_participant(img_path.name)
            per_participant[p]["attempted"] += 1
            coords, status = extract_landmarks(img_path, landmarker)
            if coords is not None:
                cls_success += 1
                total_success += 1
                per_participant[p]["success"] += 1
            else:
                fail_reasons[status] += 1

        results[cls_dir.name] = {
            "sampled": len(sampled),
            "success": cls_success,
            "rate": cls_success / len(sampled),
            "fail_reasons": dict(fail_reasons),
            "per_participant": {k: dict(v) for k, v in per_participant.items()},
        }

    landmarker.close()

    overall = (total_success / total_attempted) if total_attempted else 0.0
    print(f"\n=== Landmark Extraction Feasibility ===")
    print(f"Attempted: {total_attempted} | Success: {total_success} | "
          f"Rate: {overall*100:.1f}%")

    low = {k: v for k, v in results.items() if v["rate"] < 0.8}
    if low:
        print("\n[WARN] Classes below 80% success:")
        for k, v in low.items():
            print(f"  {k}: {v['success']}/{v['sampled']} "
                  f"({v['rate']*100:.0f}%) — {v['fail_reasons']}")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(
            {"overall_success_rate": overall, "classes": results},
            f, indent=2,
        )
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()