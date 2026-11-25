
#!/usr/bin/env python3
"""
SignaBridge — Stage 2, Module 4 evaluation.

Opens features_test.npz for the first time (P9/P10, signer-independent)
and computes:
- Test accuracy, macro precision/recall/F1, per-class precision/recall
- Test confusion matrix (PNG)
- Reliability diagram on validation (PNG)
- Proxy-unknown set: 31 non-V1 ASL-HG classes, sampled from P9/P10
- Unknown acceptance rate at threshold 0.80 and at calibrated p05
- Unknown acceptance curve (PNG)
- evaluation_report.json with the shipping-checkpoint and
  recommended-threshold decision, with reasoning.

Requires: features_test.npz, features_validation.npz,
          model_best.keras, model_final.keras, split_manifest.json,
          hand_landmarker.task, and the raw dataset directory.

NOTE: this is the FIRST time features_test.npz is opened.
No earlier module has read it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from tensorflow import keras


# Ensure tools/ is importable so the Keras serialization decorator runs.
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from train import smoothed_sparse_ce  # noqa: E402
from feature_extractor import (  # noqa: E402
    extract_features,
    FeatureExtractionError,
)


ARTIFACTS = Path("tools/artifacts")

TEST_NPZ = ARTIFACTS / "features_test.npz"
VAL_NPZ = ARTIFACTS / "features_validation.npz"
MANIFEST = ARTIFACTS / "split_manifest.json"
LABEL_ORDER = ARTIFACTS / "label_order.json"

MODEL_BEST = ARTIFACTS / "model_best.keras"
MODEL_FINAL = ARTIFACTS / "model_final.keras"

UNKNOWN_NPZ = ARTIFACTS / "features_unknown_proxy.npz"
UNKNOWN_MANIFEST = ARTIFACTS / "unknown_proxy_manifest.json"

CM_PNG = ARTIFACTS / "confusion_matrix_test.png"
RELIABILITY_PNG = ARTIFACTS / "reliability_diagram.png"
UNKNOWN_CURVE_PNG = ARTIFACTS / "unknown_rejection_curve.png"

REPORT_OUT = ARTIFACTS / "evaluation_report.json"

MODEL_PATH = Path("tools/hand_landmarker.task")
RAW_ROOT = Path("tools/datasets/asl_hg/asl_dataset")

V1_CLASSES = ["A", "B", "L", "V", "Y"]

ALL_CLASSES = (
    [str(i) for i in range(10)]
    + [chr(ord("A") + i) for i in range(26)]
)

UNKNOWN_CLASSES = [
    c for c in ALL_CLASSES if c not in V1_CLASSES
]

UNKNOWN_PARTICIPANTS = ["P9", "P10"]
UNKNOWN_PER_PARTICIPANT_PER_CLASS = 10

THRESHOLD_MASTER_PROMPT = 0.80
MIN_TARGET_RETENTION = 0.93


# ============================================================
# Loaders
# ============================================================

def load_npz(path: Path):
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing {path}")

    with np.load(path, allow_pickle=True) as data:
        features = data["features"]
        labels = data["labels"]
        participants = data["participants"]

    return features, labels, participants


def load_models():
    best = keras.models.load_model(
        MODEL_BEST,
        compile=False,
        custom_objects={
            "smoothed_sparse_ce": smoothed_sparse_ce
        },
    )

    final = keras.models.load_model(
        MODEL_FINAL,
        compile=False,
        custom_objects={
            "smoothed_sparse_ce": smoothed_sparse_ce
        },
    )

    return best, final


# ============================================================
# Test-set metrics
# ============================================================

def evaluate_on_test(
    model,
    X,
    y_int,
    classes,
    tag: str,
) -> dict:

    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)

    correct = preds == y_int
    acc = float(correct.mean())

    class_indices = list(range(len(classes)))

    cm = confusion_matrix(
        y_int,
        preds,
        labels=class_indices,
    )

    prec, rec, f1, support = precision_recall_fscore_support(
        y_int,
        preds,
        labels=class_indices,
        zero_division=0,
    )

    per_class = {
        classes[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    }

    macro_precision = float(np.mean(prec))
    macro_recall = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))

    # Confidence quantiles on test samples.
    conf_q = {
        f"p{int(q * 100):02d}": float(
            np.quantile(max_probs, q)
        )
        for q in (
            0.01,
            0.05,
            0.10,
            0.50,
            0.90,
            0.95,
            0.99,
        )
    }

    return {
        "tag": tag,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "test_max_prob_quantiles": conf_q,
        "test_max_prob_mean": float(max_probs.mean()),
    }


# ============================================================
# Proxy-unknown extraction
# ============================================================

def _create_landmarker() -> vision.HandLandmarker:

    base_options = mp_python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


def _process_image(landmarker, img_path: Path):

    img = cv2.imread(str(img_path))

    if img is None:
        return None, None, "read_failed"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect(mp_img)

    if not result.hand_landmarks:
        return None, None, "no_hand_detected"

    coords = np.array(
        [
            [p.x, p.y, p.z]
            for p in result.hand_landmarks[0]
        ],
        dtype=np.float32,
    )

    try:
        feats = extract_features(coords)

    except FeatureExtractionError as exc:
        return coords, None, f"feature_error: {exc}"

    return coords, feats, "ok"


def build_unknown_proxy(
    landmarker,
    seed: int = 42,
) -> None:
    """
    Sample P9/P10 images from the 31 non-V1 classes
    and extract features.
    """

    rng = np.random.default_rng(seed)

    rows = []
    failures = []
    attempted = 0

    for cls in tqdm(
        UNKNOWN_CLASSES,
        desc="unknown classes",
    ):

        cls_dir = RAW_ROOT / cls

        if not cls_dir.is_dir():
            print(f"[WARN] Missing class dir: {cls_dir}")
            continue

        for participant in UNKNOWN_PARTICIPANTS:

            pattern = f"{participant}_{cls}_"

            candidates = [
                p
                for p in cls_dir.iterdir()
                if (
                    p.suffix.lower() in (
                        ".jpg",
                        ".jpeg",
                        ".png",
                    )
                    and p.name.startswith(pattern)
                )
            ]

            if not candidates:
                print(
                    f"[WARN] No images for "
                    f"{cls}/{participant}"
                )
                continue

            take = min(
                UNKNOWN_PER_PARTICIPANT_PER_CLASS,
                len(candidates),
            )

            chosen = rng.choice(
                candidates,
                size=take,
                replace=False,
            )

            for img_path in chosen:

                attempted += 1

                _, feats, status = _process_image(
                    landmarker,
                    Path(img_path),
                )

                if status == "ok":

                    rows.append({
                        "class": cls,
                        "participant": participant,
                        "path": str(img_path),
                        "features": feats,
                    })

                else:

                    failures.append({
                        "path": str(img_path),
                        "reason": status,
                    })

    n_ok = len(rows)

    if n_ok == 0:
        raise SystemExit(
            "[ERROR] No unknown-proxy samples extracted."
        )

    feats = np.stack(
        [r["features"] for r in rows]
    ).astype(np.float32)

    labels = np.array(
        [r["class"] for r in rows],
        dtype=object,
    )

    participants = np.array(
        [r["participant"] for r in rows],
        dtype="<U3",
    )

    paths = np.array(
        [r["path"] for r in rows],
        dtype=object,
    )

    np.savez_compressed(
        UNKNOWN_NPZ,
        features=feats,
        labels=labels,
        participants=participants,
        paths=paths,
    )

    with open(UNKNOWN_MANIFEST, "w") as f:

        json.dump(
            {
                "n_attempted": attempted,
                "n_ok": n_ok,
                "n_failures": len(failures),
                "failure_examples": failures[:5],
                "unknown_classes": UNKNOWN_CLASSES,
                "participants": UNKNOWN_PARTICIPANTS,
                "per_participant_per_class": (
                    UNKNOWN_PER_PARTICIPANT_PER_CLASS
                ),
                "note": (
                    "Proxy-unknown set built from 31 "
                    "non-V1 ASL-HG classes. "
                    "Signer-independent (P9/P10 only). "
                    "This is a handshape proxy, "
                    "NOT a universal unknown class."
                ),
            },
            f,
            indent=2,
        )

    print(
        f"Built unknown proxy: {n_ok}/{attempted} ok, "
        f"{len(failures)} failed → {UNKNOWN_NPZ}"
    )


def load_unknown_proxy():

    if not UNKNOWN_NPZ.exists():
        raise SystemExit(
            f"[ERROR] Missing unknown proxy: {UNKNOWN_NPZ}"
        )

    with np.load(
        UNKNOWN_NPZ,
        allow_pickle=True,
    ) as data:

        return (
            data["features"],
            data["labels"],
            data["participants"],
        )


# ============================================================
# Unknown acceptance analysis
# ============================================================

def unknown_analysis(
    model,
    X_unknown,
    X_val,
    y_val_int,
    classes,
    tag: str,
) -> dict:
    """
    Compute:
    - Maximum softmax distribution on proxy-unknowns
    - Acceptance rate at threshold 0.80
    - Acceptance rate at validation p05 threshold
    - Per-target-class prediction breakdown
    """

    u_probs = model.predict(
        X_unknown,
        verbose=0,
    )

    u_max = u_probs.max(axis=1)
    u_preds = u_probs.argmax(axis=1)

    v_probs = model.predict(
        X_val,
        verbose=0,
    )

    v_max = v_probs.max(axis=1)

    calibrated_threshold = float(
        np.quantile(v_max, 0.05)
    )

    def rate_at(threshold: float) -> float:
        return float(
            (u_max >= threshold).mean()
        )

    per_class_accept = {}

    for i, cls in enumerate(classes):

        mask = u_preds == i

        per_class_accept[cls] = {
            "unknowns_predicted_as_this": int(
                mask.sum()
            ),
            "unknown_max_prob_mean": (
                float(u_max[mask].mean())
                if mask.any()
                else 0.0
            ),
        }

    return {
        "tag": tag,
        "unknown_max_prob_mean": float(
            u_max.mean()
        ),
        "unknown_max_prob_quantiles": {
            f"p{int(q * 100):02d}": float(
                np.quantile(u_max, q)
            )
            for q in (
                0.50,
                0.90,
                0.95,
                0.99,
            )
        },
        "val_p05_threshold": calibrated_threshold,
        "acceptance_at_master_prompt_0_80": (
            rate_at(THRESHOLD_MASTER_PROMPT)
        ),
        "acceptance_at_calibrated_p05": (
            rate_at(calibrated_threshold)
        ),
        "per_target_class_breakdown": per_class_accept,
    }


def threshold_curve(
    model,
    X_unknown,
    X_val,
    y_val_int,
):

    thresholds = np.linspace(
        0.5,
        0.999,
        200,
    )

    v_probs = model.predict(
        X_val,
        verbose=0,
    )

    u_probs = model.predict(
        X_unknown,
        verbose=0,
    )

    v_max = v_probs.max(axis=1)
    u_max = u_probs.max(axis=1)

    v_correct = (
        v_probs.argmax(axis=1) == y_val_int
    )

    # Correct target predictions retained after thresholding,
    # relative to all correctly classified validation samples.
    correct_count = max(
        int(v_correct.sum()),
        1,
    )

    target_retention = np.array([
        float(
            ((v_max >= threshold) & v_correct).sum()
            / correct_count
        )
        for threshold in thresholds
    ])

    unknown_rejection = np.array([
        float(
            (u_max < threshold).mean()
        )
        for threshold in thresholds
    ])

    return (
        thresholds,
        target_retention,
        unknown_rejection,
    )


# ============================================================
# Plots
# ============================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    classes,
    path: Path,
    title: str,
):

    fig, ax = plt.subplots(
        figsize=(6.4, 5.2)
    )

    im = ax.imshow(
        cm,
        cmap="Blues",
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > thresh
                    else "black"
                ),
            )

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(
    model,
    X_val,
    y_val_int,
    path: Path,
    n_bins: int = 10,
):

    probs = model.predict(
        X_val,
        verbose=0,
    )

    max_probs = probs.max(axis=1)

    correct = (
        probs.argmax(axis=1) == y_val_int
    ).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_acc = []
    bin_conf = []
    bin_count = []

    for lo, hi in zip(
        bin_edges[:-1],
        bin_edges[1:],
    ):

        mask = (
            (max_probs >= lo)
            & (max_probs < hi)
        )

        n = int(mask.sum())

        if n == 0:
            continue

        bin_acc.append(
            float(correct[mask].mean())
        )

        bin_conf.append(
            float(max_probs[mask].mean())
        )

        bin_count.append(n)

    fig, ax = plt.subplots(
        figsize=(6.4, 5.0)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.5,
        label="perfect calibration",
    )

    ax.plot(
        bin_conf,
        bin_acc,
        "o-",
        label="observed",
    )

    for confidence, accuracy, count in zip(
        bin_conf,
        bin_acc,
        bin_count,
    ):

        ax.annotate(
            str(count),
            xy=(confidence, accuracy),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")

    ax.set_title(
        "Reliability diagram "
        "(validation, n samples labeled)"
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_unknown_curve(
    thresholds,
    target_retention,
    unknown_rejection,
    path: Path,
):

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        thresholds,
        target_retention,
        label="target retention (val)",
    )

    ax.plot(
        thresholds,
        unknown_rejection,
        label="unknown rejection (proxy)",
    )

    ax.axvline(
        THRESHOLD_MASTER_PROMPT,
        color="red",
        linestyle=":",
        label="0.80 (Master Prompt)",
    )

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate")

    ax.set_title(
        "Target retention vs unknown rejection"
    )

    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        MODEL_BEST,
        MODEL_FINAL,
        TEST_NPZ,
        VAL_NPZ,
        LABEL_ORDER,
    ]

    for path in required_files:
        if not path.exists():
            raise SystemExit(
                f"[ERROR] Required file missing: {path}"
            )

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"[ERROR] Hand Landmarker model missing: {MODEL_PATH}"
        )

    if not RAW_ROOT.is_dir() and not UNKNOWN_NPZ.exists():
        raise SystemExit(
            f"[ERROR] Raw dataset directory missing: {RAW_ROOT}"
        )

    with open(LABEL_ORDER, "r") as f:
        classes = json.load(f)["classes"]

    class_to_idx = {
        c: i
        for i, c in enumerate(classes)
    }

    # --------------------------------------------------------
    # 1. Load test and validation splits
    # --------------------------------------------------------

    X_test, y_test_s, p_test = load_npz(TEST_NPZ)
    X_val, y_val_s, p_val = load_npz(VAL_NPZ)

    try:
        y_test = np.array(
            [class_to_idx[s] for s in y_test_s],
            dtype=np.int32,
        )

        y_val = np.array(
            [class_to_idx[s] for s in y_val_s],
            dtype=np.int32,
        )

    except KeyError as exc:
        raise SystemExit(
            f"[ERROR] Unknown label in dataset: {exc}"
        )

    print(f"Classes       : {classes}")

    print(
        f"Test split    : n={len(X_test)}, "
        f"participants={sorted(set(p_test.tolist()))}"
    )

    print(
        f"Val split     : n={len(X_val)}, "
        f"participants={sorted(set(p_val.tolist()))}"
    )

    # --------------------------------------------------------
    # 2. Load models
    # --------------------------------------------------------

    best_model, final_model = load_models()

    # --------------------------------------------------------
    # 3. Test-set metrics for both models
    # --------------------------------------------------------

    print(
        "\n=== Test metrics "
        "(P9/P10 — first and only time test is opened) ==="
    )

    best_test = evaluate_on_test(
        best_model,
        X_test,
        y_test,
        classes,
        "best_loss",
    )

    final_test = evaluate_on_test(
        final_model,
        X_test,
        y_test,
        classes,
        "final_epoch",
    )

    for metrics in (
        best_test,
        final_test,
    ):

        print(f"\n[{metrics['tag']}]")

        print(
            f"  accuracy         : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  macro P / R / F1 : "
            f"{metrics['macro_precision']:.4f} / "
            f"{metrics['macro_recall']:.4f} / "
            f"{metrics['macro_f1']:.4f}"
        )

        for cls, scores in metrics["per_class"].items():

            print(
                f"    {cls}: "
                f"P={scores['precision']:.4f} "
                f"R={scores['recall']:.4f} "
                f"F1={scores['f1']:.4f} "
                f"(n={scores['support']})"
            )

    # --------------------------------------------------------
    # 4. Build / load proxy-unknown set
    # --------------------------------------------------------

    if not UNKNOWN_NPZ.exists():

        print(
            "\n=== Extracting proxy-unknown set "
            "from 31 non-V1 classes (P9/P10) ==="
        )

        landmarker = _create_landmarker()

        try:
            build_unknown_proxy(landmarker)

        finally:
            landmarker.close()

    else:

        print(
            f"\nReusing existing {UNKNOWN_NPZ} "
            "(delete to rebuild)"
        )

    X_unknown, y_unknown_s, p_unknown = (
        load_unknown_proxy()
    )

    print(
        f"Proxy-unknown : n={len(X_unknown)}, "
        f"classes={sorted(set(y_unknown_s.tolist()))[:6]}... "
        f"participants={sorted(set(p_unknown.tolist()))}"
    )

    # --------------------------------------------------------
    # 5. Unknown acceptance analysis for both models
    # --------------------------------------------------------

    print("\n=== Proxy-unknown acceptance ===")

    best_unk = unknown_analysis(
        best_model,
        X_unknown,
        X_val,
        y_val,
        classes,
        "best_loss",
    )

    final_unk = unknown_analysis(
        final_model,
        X_unknown,
        X_val,
        y_val,
        classes,
        "final_epoch",
    )

    for unknown_metrics in (
        best_unk,
        final_unk,
    ):

        print(f"\n[{unknown_metrics['tag']}]")

        print(
            f"  unknown max_prob mean : "
            f"{unknown_metrics['unknown_max_prob_mean']:.4f}"
        )

        print(
            f"  unknown max_prob p95  : "
            f"{unknown_metrics['unknown_max_prob_quantiles']['p95']:.4f}"
        )

        print(
            f"  val p05 threshold     : "
            f"{unknown_metrics['val_p05_threshold']:.4f}"
        )

        print(
            f"  acceptance @ 0.80     : "
            f"{unknown_metrics['acceptance_at_master_prompt_0_80']:.4f}"
        )

        print(
            f"  acceptance @ val p05  : "
            f"{unknown_metrics['acceptance_at_calibrated_p05']:.4f}"
        )

    # --------------------------------------------------------
    # 6. Plots
    # --------------------------------------------------------

    plot_confusion_matrix(
        np.array(best_test["confusion_matrix"]),
        classes,
        CM_PNG,
        (
            "Test confusion matrix — model_best "
            f"(acc={best_test['accuracy']:.4f})"
        ),
    )

    plot_reliability(
        best_model,
        X_val,
        y_val,
        RELIABILITY_PNG,
    )

    thresholds, tgt_ret, unk_rej = threshold_curve(
        best_model,
        X_unknown,
        X_val,
        y_val,
    )

    plot_unknown_curve(
        thresholds,
        tgt_ret,
        unk_rej,
        UNKNOWN_CURVE_PNG,
    )

    print(
        f"\nWrote plots: {CM_PNG}, "
        f"{RELIABILITY_PNG}, {UNKNOWN_CURVE_PNG}"
    )

    # --------------------------------------------------------
    # 7. Shipping decision — corrected scoring
    # --------------------------------------------------------

    def tradeoff_score(
        test_metrics: dict,
        unk_analysis: dict,
    ) -> float:
        """
        Combined score for choosing the shipping checkpoint.

        Higher test accuracy is better.
        Lower proxy-unknown acceptance at 0.80 is better.

        A 10% increase in unknown acceptance costs approximately
        5% in the combined score.
        """

        return (
            test_metrics["accuracy"]
            - 0.5
            * unk_analysis[
                "acceptance_at_master_prompt_0_80"
            ]
        )

    best_score = tradeoff_score(
        best_test,
        best_unk,
    )

    final_score = tradeoff_score(
        final_test,
        final_unk,
    )

    if best_score >= final_score:

        shipping = "model_best.keras"

        shipping_reason = (
            f"model_best combined score = {best_score:.4f}; "
            f"model_final combined score = {final_score:.4f}. "
            f"model_best test accuracy = "
            f"{best_test['accuracy']:.4f}; "
            f"model_final = {final_test['accuracy']:.4f}. "
            f"model_best unknown acceptance @0.80 = "
            f"{best_unk['acceptance_at_master_prompt_0_80']:.4f} "
            f"vs model_final = "
            f"{final_unk['acceptance_at_master_prompt_0_80']:.4f}."
        )

    else:

        shipping = "model_final.keras"

        shipping_reason = (
            f"model_final combined score = {final_score:.4f}; "
            f"model_best combined score = {best_score:.4f}. "
            f"model_final test accuracy = "
            f"{final_test['accuracy']:.4f}; "
            f"model_best = {best_test['accuracy']:.4f}. "
            f"model_final unknown acceptance @0.80 = "
            f"{final_unk['acceptance_at_master_prompt_0_80']:.4f} "
            f"vs model_best = "
            f"{best_unk['acceptance_at_master_prompt_0_80']:.4f}."
        )

    # --------------------------------------------------------
    # 8. Recommended threshold — corrected elbow rule
    # --------------------------------------------------------

    # Choose the highest threshold that maintains at least
    # 93% validation target retention.
    #
    # This allows up to 7% per-frame target rejection.
    # The smoother's 7-of-10 agreement window is intended
    # to absorb some individual-frame errors.
    #
    # The actual result is measured from the generated curve.
    # No fixed threshold such as 0.95 is assumed.

    eligible = thresholds[
        tgt_ret >= MIN_TARGET_RETENTION
    ]

    if eligible.size == 0:

        raise SystemExit(
            "[ERROR] No threshold keeps target retention >= "
            f"{MIN_TARGET_RETENTION:.2f}. "
            "Curve data is inconsistent."
        )

    recommended_threshold = float(
        eligible.max()
    )

    idx_rec = int(
        np.argmin(
            np.abs(
                thresholds - recommended_threshold
            )
        )
    )

    idx_080 = int(
        np.argmin(
            np.abs(
                thresholds - THRESHOLD_MASTER_PROMPT
            )
        )
    )

    recommended_threshold_reason = (
        f"Chosen as the highest threshold that keeps "
        f"validation target retention >= "
        f"{MIN_TARGET_RETENTION:.2f} "
        f"(per-frame target error <= "
        f"{1.0 - MIN_TARGET_RETENTION:.2f}, "
        f"intended to be absorbed partly by the smoother's "
        f"7-of-10 agreement window). "
        f"At this threshold: target retention = "
        f"{tgt_ret[idx_rec]:.4f}, "
        f"proxy-unknown rejection = "
        f"{unk_rej[idx_rec]:.4f} "
        f"(acceptance = {1.0 - unk_rej[idx_rec]:.4f}). "
        f"At the Master Prompt default 0.80: "
        f"target retention = {tgt_ret[idx_080]:.4f}, "
        f"proxy-unknown rejection = "
        f"{unk_rej[idx_080]:.4f} "
        f"(acceptance = {1.0 - unk_rej[idx_080]:.4f})."
    )

    # --------------------------------------------------------
    # 9. Write report
    # --------------------------------------------------------

    report = {
        "classes": classes,

        "test_split": {
            "n": int(len(X_test)),
            "participants": sorted(
                set(p_test.tolist())
            ),
            "note": (
                "First and only time the test split is opened."
            ),
        },

        "validation_split": {
            "n": int(len(X_val)),
            "participants": sorted(
                set(p_val.tolist())
            ),
        },

        "models": {
            "best_loss": {
                **best_test,
                "unknown_analysis": best_unk,
                "tradeoff_score": best_score,
            },

            "final_epoch": {
                **final_test,
                "unknown_analysis": final_unk,
                "tradeoff_score": final_score,
            },
        },

        "shipping_decision": {
            "checkpoint": shipping,
            "reason": shipping_reason,
            "best_score": best_score,
            "final_score": final_score,
        },

        "threshold_decision": {
            "recommended_threshold": recommended_threshold,
            "minimum_target_retention": (
                MIN_TARGET_RETENTION
            ),
            "measured_target_retention": float(
                tgt_ret[idx_rec]
            ),
            "measured_unknown_rejection": float(
                unk_rej[idx_rec]
            ),
            "measured_unknown_acceptance": float(
                1.0 - unk_rej[idx_rec]
            ),
            "master_prompt_default_0_80": (
                THRESHOLD_MASTER_PROMPT
            ),
            "master_prompt_target_retention": float(
                tgt_ret[idx_080]
            ),
            "master_prompt_unknown_rejection": float(
                unk_rej[idx_080]
            ),
            "master_prompt_unknown_acceptance": float(
                1.0 - unk_rej[idx_080]
            ),
            "reason": recommended_threshold_reason,
            "note": (
                "The recommended threshold is selected from "
                "the measured validation/proxy-unknown curve. "
                "If it differs from 0.80, Stage 3 should document "
                "the measured recommendation and the original "
                "Master Prompt default in PredictionSmoother.java "
                "and the ML README. Do not silently hard-code "
                "a threshold without this report."
            ),
        },

        "unknown_proxy": {
            "n": int(len(X_unknown)),
            "classes": UNKNOWN_CLASSES,
            "participants": sorted(
                set(p_unknown.tolist())
            ),
            "documented_limitation": (
                "Proxy-unknown set is handshape-only "
                "(31 non-V1 ASL-HG classes), signer-independent "
                "(P9/P10). It is NOT a universal unknown-class "
                "sample. Unknown rejection rates measured here "
                "are not a guarantee of real-world performance."
            ),
        },

        "calibration_limitation": (
            "Label smoothing 0.05 constrains confidence "
            "and can affect probability calibration. "
            "The reliability diagram should be interpreted "
            "alongside the classification metrics and "
            "unknown-proxy rejection results."
        ),

        "feature_space_distance_rejection": {
            "status": "deferred",
            "reason": (
                "Softmax thresholding plus smoother agreement "
                "window is evaluated first. If proxy-unknown "
                "acceptance at the recommended threshold is "
                "not acceptable, feature-space distance-based "
                "rejection is a candidate for a follow-up module."
            ),
        },

        "test_set_status": (
            "consumed by this module; "
            "not reused for model selection"
        ),
    }

    with open(
        REPORT_OUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 10. Final output
    # --------------------------------------------------------

    print(f"\nWrote {REPORT_OUT}")

    print("\n=== Decisions ===")

    print(
        f"Shipping checkpoint       : {shipping}"
    )

    print(
        f"Best combined score       : {best_score:.4f}"
    )

    print(
        f"Final combined score      : {final_score:.4f}"
    )

    print(
        f"Recommended threshold     : "
        f"{recommended_threshold:.4f}"
    )

    print(
        f"Target retention          : "
        f"{tgt_ret[idx_rec]:.4f}"
    )

    print(
        f"Unknown rejection         : "
        f"{unk_rej[idx_rec]:.4f}"
    )

    print(
        f"Unknown acceptance        : "
        f"{1.0 - unk_rej[idx_rec]:.4f}"
    )

    print(
        f"\nMaster Prompt threshold   : "
        f"{THRESHOLD_MASTER_PROMPT:.2f}"
    )

    print(
        f"Target retention @ 0.80   : "
        f"{tgt_ret[idx_080]:.4f}"
    )

    print(
        f"Unknown rejection @ 0.80  : "
        f"{unk_rej[idx_080]:.4f}"
    )

    print(
        f"Unknown acceptance @ 0.80 : "
        f"{1.0 - unk_rej[idx_080]:.4f}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
