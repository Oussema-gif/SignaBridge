#!/usr/bin/env python3
"""
SignaBridge — Stage 2 parity test vector generator.

Samples 20 real validation images (4 per class) plus 2 synthetic edge cases
and emits tools/artifacts/parity_vectors.json. This file is the single source
of truth for the Python/Java feature-extraction parity test that runs in
Stage 3.

Contract (see tools/feature_contract.md):
    For each vector:
      - 'landmarks'           : [[x, y, z] × 21]
      - 'expected_features'   : [63 floats]  (null if should_raise)
      - 'should_raise'        : bool          (true if Java MUST raise)
      - 'tolerance'           : float         (per-vector absolute tolerance)

A conforming Java implementation must:
    - Produce a 63-element float array per vector where should_raise is false
    - Match every element to within tolerance
    - Throw FeatureExtractionException for vectors where should_raise is true
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_extractor import extract_features, FeatureExtractionError

ARTIFACTS = Path("tools/artifacts")
NPZ_PATH = ARTIFACTS / "features_validation.npz"
PARITY_PATH = ARTIFACTS / "parity_vectors.json"

SAMPLES_PER_CLASS = 4
TOLERANCE = 1e-5
SEED = 42


def main() -> None:
    if not NPZ_PATH.exists():
        raise SystemExit(
            f"[ERROR] {NPZ_PATH} not found. Run extract_landmarks_batch.py first."
        )

    data = np.load(NPZ_PATH, allow_pickle=True)
    landmarks = data["landmarks"]
    features = data["features"]
    labels = data["labels"]
    participants = data["participants"]

    rng = np.random.default_rng(SEED)
    vectors: list[dict] = []

    for cls in sorted(set(labels.tolist())):
        mask = labels == cls
        idxs = np.where(mask)[0]
        chosen = rng.choice(idxs, size=SAMPLES_PER_CLASS, replace=False)
        for i in chosen:
            vectors.append(
                {
                    "source": "real",
                    "class": str(labels[i]),
                    "participant": str(participants[i]),
                    "landmarks": landmarks[i].astype(np.float32).tolist(),
                    "expected_features": features[i].astype(np.float32).tolist(),
                    "should_raise": False,
                    "tolerance": TOLERANCE,
                }
            )

    # Synthetic edge case 1: valid hand with non-trivial middle-MCP offset
    synthetic = np.zeros((21, 3), dtype=np.float32)
    synthetic[9] = [0.0, 0.1, 0.0]
    synthetic[4] = [0.05, 0.15, 0.0]
    synthetic[8] = [0.02, 0.20, 0.0]
    expected = extract_features(synthetic)
    vectors.append(
        {
            "source": "synthetic",
            "class": "synthetic-valid",
            "participant": "synthetic",
            "landmarks": synthetic.tolist(),
            "expected_features": expected.tolist(),
            "should_raise": False,
            "tolerance": TOLERANCE,
        }
    )

    # Synthetic edge case 2: all landmarks coincide → must raise on both sides
    degenerate = np.zeros((21, 3), dtype=np.float32)
    try:
        extract_features(degenerate)
        raise AssertionError("extract_features should have raised on degenerate input")
    except FeatureExtractionError:
        pass  # expected
    vectors.append(
        {
            "source": "synthetic",
            "class": "synthetic-degenerate",
            "participant": "synthetic",
            "landmarks": degenerate.tolist(),
            "expected_features": None,
            "should_raise": True,
            "tolerance": TOLERANCE,
        }
    )

    # Synthetic edge case 3: NaN in landmarks → must raise on both sides
    nan_input = np.zeros((21, 3), dtype=np.float32)
    nan_input[5, 0] = np.nan
    try:
        extract_features(nan_input)
        raise AssertionError("extract_features should have raised on NaN input")
    except FeatureExtractionError:
        pass
    vectors.append(
        {
            "source": "synthetic",
            "class": "synthetic-nan",
            "participant": "synthetic",
            "landmarks": nan_input.tolist(),
            "expected_features": None,
            "should_raise": True,
            "tolerance": TOLERANCE,
        }
    )

    doc = {
        "description": "Python/Java feature extraction parity vectors",
        "contract_reference": "tools/feature_contract.md",
        "generated_from": str(NPZ_PATH),
        "rng_seed": SEED,
        "n_vectors": len(vectors),
        "n_real": sum(1 for v in vectors if v["source"] == "real"),
        "n_synthetic": sum(1 for v in vectors if v["source"] == "synthetic"),
        "vectors": vectors,
    }

    PARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARITY_PATH, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"Wrote {len(vectors)} parity vectors to {PARITY_PATH}")
    print(f"  real={doc['n_real']}, synthetic={doc['n_synthetic']}, "
          f"tolerance={TOLERANCE}")


if __name__ == "__main__":
    main()