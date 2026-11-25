#!/usr/bin/env python3
"""
SignaBridge — Stage 2 dataset preparation.

Reads the raw ASL-HG images from tools/datasets/asl_hg/asl_dataset/
and produces a participant-based, signer-independent split manifest for the
V1 vocabulary (A, B, L, V, Y).

Split policy (from the Stage 1 feasibility report):
    TRAIN      : P1, P2, P3, P4, P5, P6   → 600 images / class → 3,000 total
    VALIDATION : P7, P8                   → 200 images / class → 1,000 total
    TEST       : P9, P10                  → 200 images / class → 1,000 total

The pre-existing asl_processed/ split is deliberately ignored: the Stage 1
inspection proved it leaks signers across train/test.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path("tools/datasets/asl_hg/asl_dataset")
MANIFEST_PATH = Path("tools/artifacts/split_manifest.json")
IMG_EXTS = (".jpg", ".jpeg", ".png")

V1_CLASSES = ["A", "B", "L", "V", "Y"]

SPLIT_BY_PARTICIPANT = {
    "train": {"P1", "P2", "P3", "P4", "P5", "P6"},
    "validation": {"P7", "P8"},
    "test": {"P9", "P10"},
}

PARTICIPANT_RE = re.compile(r"^P(\d+)_([A-Z0-9])_(\d+)\.")


def participant_to_split() -> dict[str, str]:
    mapping = {}
    for split, participants in SPLIT_BY_PARTICIPANT.items():
        for p in participants:
            if p in mapping:
                raise ValueError(f"Participant {p} assigned to multiple splits")
            mapping[p] = split
    return mapping


def build_manifest():
    if not DATASET_ROOT.is_dir():
        sys.exit(f"[ERROR] Dataset directory not found: {DATASET_ROOT}")

    p2s = participant_to_split()
    manifest = {
        "vocabulary": V1_CLASSES,
        "feature_dim": 63,
        "split_policy": {k: sorted(v) for k, v in SPLIT_BY_PARTICIPANT.items()},
        "source_root": str(DATASET_ROOT),
        "splits": {"train": [], "validation": [], "test": []},
        "counts": defaultdict(lambda: defaultdict(int)),
    }

    unparseable = []
    out_of_vocab = []
    missing_participants = set()

    for cls in V1_CLASSES:
        cls_dir = DATASET_ROOT / cls
        if not cls_dir.is_dir():
            sys.exit(f"[ERROR] Missing class folder: {cls_dir}")
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            m = PARTICIPANT_RE.match(img.name)
            if not m:
                unparseable.append(str(img))
                continue
            participant = f"P{m.group(1)}"
            # Cross-check class letter in filename against folder
            if m.group(2).upper() != cls.upper():
                out_of_vocab.append((str(img), cls, m.group(2)))
                continue
            split = p2s.get(participant)
            if split is None:
                missing_participants.add(participant)
                continue
            manifest["splits"][split].append(
                {"path": str(img), "class": cls, "participant": participant}
            )
            manifest["counts"][split][cls] += 1

    # Integrity checks
    errors = []
    if unparseable:
        errors.append(f"{len(unparseable)} unparseable filenames (first: {unparseable[:3]})")
    if out_of_vocab:
        errors.append(f"{len(out_of_vocab)} class/filename mismatches (first: {out_of_vocab[:3]})")
    if missing_participants:
        errors.append(f"Participants with no split: {sorted(missing_participants)}")

    for split, participants in SPLIT_BY_PARTICIPANT.items():
        observed = {row["participant"] for row in manifest["splits"][split]}
        if observed != participants:
            errors.append(
                f"{split}: expected participants {sorted(participants)}, "
                f"observed {sorted(observed)}"
            )
        for cls in V1_CLASSES:
            expected_per_participant = 100  # 1000 images / 10 participants
            expected = len(participants) * expected_per_participant
            actual = manifest["counts"][split][cls]
            if actual != expected:
                errors.append(
                    f"{split}/{cls}: expected {expected} images, got {actual}"
                )

    if errors:
        print("[ERROR] Manifest integrity checks failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    manifest["counts"] = {k: dict(v) for k, v in manifest["counts"].items()}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Dataset manifest written.")
    for split in ("train", "validation", "test"):
        total = sum(manifest["counts"][split].values())
        print(f"  {split:10s}: {total:5d} images  "
              f"({manifest['counts'][split]})")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    build_manifest()