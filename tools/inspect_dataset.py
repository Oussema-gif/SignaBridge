#!/usr/bin/env python3
"""
ASL-HG Dataset Inspection Script (v2 — matches actual archive layout)
Verifies class balance, participant IDs, and pre-existing split characteristics.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

from PIL import Image

DATASET_ROOT = Path("tools/datasets/asl_hg")
RAW_ROOT = DATASET_ROOT / "asl_dataset"
PROCESSED_ROOT = DATASET_ROOT / "asl_processed"
REPORT_PATH = Path("tools/artifacts/dataset_report.json")

EXPECTED_CLASSES = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
IMG_EXTS = (".jpg", ".jpeg", ".png")
PARTICIPANT_RE = re.compile(r"^P(\d+)_")


def extract_participant(filename: str):
    m = PARTICIPANT_RE.match(filename)
    return f"P{m.group(1)}" if m else None


def list_images(folder: Path):
    return [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS]


def inspect_raw(root: Path):
    if not root.exists():
        print(f"[ERROR] Missing directory: {root}")
        return None

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_counts = {}
    participants_per_class = {}
    all_participants = set()
    unparseable = 0

    for cls_dir in class_dirs:
        images = list_images(cls_dir)
        class_counts[cls_dir.name] = len(images)
        per_class_participants = defaultdict(int)
        for f in images:
            p = extract_participant(f.name)
            if p:
                per_class_participants[p] += 1
                all_participants.add(p)
            else:
                unparseable += 1
        participants_per_class[cls_dir.name] = {
            "count": len(per_class_participants),
            "ids": sorted(per_class_participants.keys(),
                          key=lambda x: int(x[1:])),
            "images_per_participant": dict(per_class_participants),
        }

    total = sum(class_counts.values())
    print(f"\n=== RAW (asl_dataset) ===")
    print(f"Classes: {len(class_dirs)}")
    print(f"Total images: {total}")
    if class_counts:
        print(f"Per-class min/max: {min(class_counts.values())} / "
              f"{max(class_counts.values())}")
    print(f"Global participants: {sorted(all_participants, key=lambda x: int(x[1:]))}")
    print(f"Unparseable filenames: {unparseable}")

    missing = set(EXPECTED_CLASSES) - set(class_counts)
    extra = set(class_counts) - set(EXPECTED_CLASSES)
    if missing:
        print(f"[WARN] Missing expected classes: {sorted(missing)}")
    if extra:
        print(f"[WARN] Unexpected classes: {sorted(extra)}")

    # Confirm every class has data from all participants (signer-independence feasibility)
    incomplete = {c: v for c, v in participants_per_class.items()
                  if v["count"] < len(all_participants)}
    if incomplete:
        print(f"[WARN] Classes missing some participants: "
              f"{ {c: v['ids'] for c, v in incomplete.items()} }")

    return {
        "root": str(root),
        "class_count": len(class_dirs),
        "total_images": total,
        "class_counts": class_counts,
        "participants_global": sorted(all_participants, key=lambda x: int(x[1:])),
        "participants_per_class": participants_per_class,
        "unparseable_filenames": unparseable,
        "missing_classes": sorted(missing),
        "extra_classes": sorted(extra),
    }


def inspect_processed(root: Path):
    if not root.exists():
        print(f"\n[ERROR] Missing directory: {root}")
        return None

    split_info = {}
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[WARN] Missing split directory: {split_dir}")
            continue
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        class_counts = {}
        participants_in_split = defaultdict(int)
        for cls_dir in class_dirs:
            images = list_images(cls_dir)
            class_counts[cls_dir.name] = len(images)
            for f in images:
                p = extract_participant(f.name)
                if p:
                    participants_in_split[p] += 1
        split_info[split] = {
            "class_count": len(class_dirs),
            "total_images": sum(class_counts.values()),
            "class_counts": class_counts,
            "participants_in_split": dict(participants_in_split),
        }
        print(f"\n=== PROCESSED / {split} ===")
        print(f"Classes: {len(class_dirs)}")
        print(f"Total images: {sum(class_counts.values())}")
        print(f"Participants present: {sorted(participants_in_split, key=lambda x: int(x[1:]))}")

    # Leakage diagnostic: participants in both train and test
    if "train" in split_info and "test" in split_info:
        train_p = set(split_info["train"]["participants_in_split"])
        test_p = set(split_info["test"]["participants_in_split"])
        overlap = train_p & test_p
        print("\n=== SPLIT LEAKAGE CHECK ===")
        print(f"Participants in TRAIN: {sorted(train_p, key=lambda x: int(x[1:]))}")
        print(f"Participants in TEST:  {sorted(test_p, key=lambda x: int(x[1:]))}")
        print(f"Participants in BOTH:  {sorted(overlap, key=lambda x: int(x[1:]))}")
        if overlap:
            print("[WARN] Pre-existing split is NOT signer-independent. "
                  "Do not use it for evaluation.")
        split_info["participants_in_both_splits"] = sorted(
            overlap, key=lambda x: int(x[1:]))

    return split_info


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("ASL-HG Dataset Inspection (v2)")
    print("=" * 60)

    raw = inspect_raw(RAW_ROOT)
    processed = inspect_processed(PROCESSED_ROOT)

    report = {
        "dataset": "ASL-HG",
        "doi": "10.17632/j4y5w2c8w9.1",
        "license": "CC BY 4.0",
        "filename_pattern": "P{participant}_{class}_{index}.jpg",
        "raw": raw,
        "processed": processed,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()