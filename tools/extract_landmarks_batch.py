#!/usr/bin/env python3
"""
SignaBridge — Stage 2 batch landmark extraction.

Reads tools/artifacts/split_manifest.json, runs MediaPipe HandLandmarker on
every image, converts to the canonical 63-D feature vector via
feature_extractor.extract_features, and writes per-split .npz archives.

Output layout per split:
    landmarks  : float32 (N, 21, 3)
    features   : float32 (N, 63)
    labels     : <U1     (N,)
    participants : <U3   (N,)
    paths      : object  (N,)

Failures are logged, not silently dropped. A split is flagged if success
rate falls below SUCCESS_WARN_THRESHOLD (0.95).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from feature_extractor import extract_features, FeatureExtractionError

MODEL_PATH = Path("tools/hand_landmarker.task")
MANIFEST_PATH = Path("tools/artifacts/split_manifest.json")
ARTIFACTS = Path("tools/artifacts")
SUCCESS_WARN_THRESHOLD = 0.95


def create_landmarker() -> vision.HandLandmarker:
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


def process_image(landmarker, img_path: str):
    """Return (landmarks(21,3) | None, features(63,) | None, status)."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, "read_failed"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_img)
    except Exception as exc:  # noqa: BLE001 — we want to record every failure mode
        return None, None, f"detector_error: {exc}"

    if not result.hand_landmarks:
        return None, None, "no_hand_detected"

    lms = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
    try:
        features = extract_features(coords)
    except FeatureExtractionError as exc:
        return coords, None, f"feature_error: {exc}"
    return coords, features, "ok"


def process_split(landmarker, split_name: str, rows: list[dict]) -> dict:
    n = len(rows)
    landmarks = np.zeros((n, 21, 3), dtype=np.float32)
    features = np.zeros((n, 63), dtype=np.float32)
    labels = np.empty(n, dtype="<U1")
    participants = np.empty(n, dtype="<U3")
    paths = np.empty(n, dtype=object)
    ok_mask = np.zeros(n, dtype=bool)
    failures: list[dict] = []

    t_start = time.time()
    for i, row in enumerate(tqdm(rows, desc=split_name, unit="img")):
        lms, feat, status = process_image(landmarker, row["path"])
        if status == "ok":
            landmarks[i] = lms
            features[i] = feat
            labels[i] = row["class"]
            participants[i] = row["participant"]
            paths[i] = row["path"]
            ok_mask[i] = True
        else:
            failures.append({"path": row["path"], "reason": status})
    elapsed = time.time() - t_start

    return {
        "landmarks": landmarks[ok_mask],
        "features": features[ok_mask],
        "labels": labels[ok_mask],
        "participants": participants[ok_mask],
        "paths": paths[ok_mask],
        "failures": failures,
        "n_attempted": n,
        "n_ok": int(ok_mask.sum()),
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"[ERROR] Model not found: {MODEL_PATH}")
    if not MANIFEST_PATH.exists():
        sys.exit(
            f"[ERROR] Manifest not found: {MANIFEST_PATH}. "
            "Run tools/prepare_dataset.py first."
        )

    manifest = json.load(open(MANIFEST_PATH))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("SignaBridge — Batch Landmark Extraction")
    print("=" * 60)
    print(f"Vocabulary: {manifest['vocabulary']}")
    print(f"Split policy: {manifest['split_policy']}")
    print()

    landmarker = create_landmarker()
    report: dict = {}
    try:
        for split in ("train", "validation", "test"):
            rows = manifest["splits"][split]
            print(f"Processing {split}: {len(rows)} images")
            result = process_split(landmarker, split, rows)

            out_path = ARTIFACTS / f"features_{split}.npz"
            np.savez_compressed(
                out_path,
                landmarks=result["landmarks"],
                features=result["features"],
                labels=result["labels"],
                participants=result["participants"],
                paths=result["paths"],
            )

            rate = result["n_ok"] / result["n_attempted"] if result["n_attempted"] else 0.0
            print(
                f"  → {result['n_ok']}/{result['n_attempted']} ok "
                f"({rate * 100:.1f}%) in {result['elapsed_sec']}s"
            )
            print(f"  → {out_path}")
            if rate < SUCCESS_WARN_THRESHOLD:
                print(
                    f"[WARN] {split}: success rate {rate * 100:.1f}% "
                    f"below {SUCCESS_WARN_THRESHOLD * 100:.0f}%"
                )

            report[split] = {
                "attempted": result["n_attempted"],
                "ok": result["n_ok"],
                "success_rate": round(rate, 4),
                "elapsed_sec": result["elapsed_sec"],
                "failure_count": len(result["failures"]),
                "failure_examples": result["failures"][:5],
            }
    finally:
        landmarker.close()

    report_path = ARTIFACTS / "landmark_extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    total_ok = sum(r["ok"] for r in report.values())
    total_attempted = sum(r["attempted"] for r in report.values())
    print(f"Overall: {total_ok}/{total_attempted} ok "
          f"({100.0 * total_ok / total_attempted:.2f}%)")


if __name__ == "__main__":
    main()