
#!/usr/bin/env python3
"""
SignaBridge — Stage 2, Module 3 (revised).

Trains two baselines on the signer-independent train split:

1. MLP: 63 -> 64 (ReLU, Dropout 0.1) -> 32 (ReLU, Dropout 0.1)
   -> 5 (softmax), Adam, label smoothing 0.05, early stopping on
   val_loss, checkpoint on val_loss (min). Saves both best-loss and
   final-epoch checkpoints.

2. k-NN: k=5, Euclidean distance.

The validation split is used for model selection only.
The test split is NOT opened here; that happens in Module 4 (evaluate.py).

Outputs (tools/artifacts/):
    label_order.json
    model_best.keras
    model_final.keras
    training_history.json
    training_report.json
    training_curves.png
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACTS = Path("tools/artifacts")

TRAIN_NPZ = ARTIFACTS / "features_train.npz"
VAL_NPZ = ARTIFACTS / "features_validation.npz"
MANIFEST = ARTIFACTS / "split_manifest.json"

MODEL_BEST = ARTIFACTS / "model_best.keras"
MODEL_FINAL = ARTIFACTS / "model_final.keras"
HISTORY_OUT = ARTIFACTS / "training_history.json"
REPORT_OUT = ARTIFACTS / "training_report.json"
CURVES_OUT = ARTIFACTS / "training_curves.png"
LABEL_ORDER_OUT = ARTIFACTS / "label_order.json"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
KNN_K = 5
LABEL_SMOOTHING = 0.05
DROPOUT_RATE = 0.1


# ============================================================
# CUSTOM LABEL-SMOOTHED SPARSE CROSS-ENTROPY
# ============================================================

@tf.keras.utils.register_keras_serializable(package="SignaBridge")
def smoothed_sparse_ce(y_true, y_pred):
    """
    Sparse label-smoothed categorical cross-entropy.

    TF 2.17's SparseCategoricalCrossentropy does not expose
    label_smoothing, so we implement it explicitly:

    1. Convert integer labels to one-hot vectors.
    2. Blend with the uniform distribution.
    3. Compute categorical cross-entropy with epsilon clipping.

    Registered with Keras serialization so Keras can resolve
    this function when restoring saved .keras models.
    """

    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.int32)
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

    y_onehot = tf.one_hot(
        y_true_int,
        num_classes,
        dtype=tf.float32
    )

    y_smooth = (
        y_onehot * (1.0 - LABEL_SMOOTHING)
        + LABEL_SMOOTHING / tf.cast(num_classes, tf.float32)
    )

    eps = tf.keras.backend.epsilon()
    y_pred_clip = tf.clip_by_value(y_pred, eps, 1.0)

    return tf.reduce_mean(
        -tf.reduce_sum(
            y_smooth * tf.math.log(y_pred_clip),
            axis=-1
        )
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_npz(path: Path):
    if not path.exists():
        raise SystemExit(
            f"[ERROR] Missing {path}. "
            "Run extract_landmarks_batch.py first."
        )

    with np.load(path, allow_pickle=True) as d:
        return (
            d["features"],
            d["labels"],
            d["participants"]
        )


def to_int_labels(
    y_str: np.ndarray,
    class_to_idx: dict[str, int]
) -> np.ndarray:

    unknown = set(y_str.tolist()) - set(class_to_idx)

    if unknown:
        raise SystemExit(
            f"[ERROR] Unrecognised labels: {sorted(unknown)}"
        )

    return np.array(
        [class_to_idx[s] for s in y_str],
        dtype=np.int32
    )


# ============================================================
# MLP MODEL
# ============================================================

def build_mlp(num_classes: int) -> keras.Model:

    model = keras.Sequential(
        [
            keras.layers.Input(
                shape=(63,),
                name="features"
            ),

            keras.layers.Dense(
                64,
                activation="relu",
                name="hidden_1"
            ),

            keras.layers.Dropout(
                DROPOUT_RATE,
                name="drop_1"
            ),

            keras.layers.Dense(
                32,
                activation="relu",
                name="hidden_2"
            ),

            keras.layers.Dropout(
                DROPOUT_RATE,
                name="drop_2"
            ),

            keras.layers.Dense(
                num_classes,
                activation="softmax",
                name="logits"
            ),
        ],
        name="signabridge_mlp",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=smoothed_sparse_ce,
        metrics=["accuracy"],
    )

    return model


# ============================================================
# CONFUSION MATRIX AND PER-CLASS ACCURACY
# ============================================================

def confusion_and_per_class(model, X, y, classes):

    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)

    cm = confusion_matrix(
        y,
        preds,
        labels=list(range(len(classes)))
    )

    per_class = {
        classes[i]: float(cm[i, i] / cm[i].sum())
        if cm[i].sum()
        else 0.0
        for i in range(len(classes))
    }

    return probs, preds, cm, per_class


# ============================================================
# MODEL EVALUATION ON VALIDATION DATA
# ============================================================

def summarize_model(model, X_val, y_val, classes, tag: str) -> dict:

    probs, preds, cm, per_class = confusion_and_per_class(
        model,
        X_val,
        y_val,
        classes
    )

    acc = float((preds == y_val).mean())

    # Pure CE: no label smoothing.
    # Keep this unchanged for best/final model comparison.
    pure_ce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    loss_pure = float(pure_ce(y_val, probs).numpy())

    max_conf = probs.max(axis=1)

    return {
        "tag": tag,
        "val_accuracy": acc,
        "val_loss_pure_ce": loss_pure,
        "val_conf_mean": float(max_conf.mean()),
        "val_conf_p50": float(np.percentile(max_conf, 50)),
        "val_conf_p05": float(np.percentile(max_conf, 5)),
        "val_conf_p95": float(np.percentile(max_conf, 95)),
        "val_confusion_matrix": cm.tolist(),
        "val_per_class_accuracy": per_class,
    }


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:

    set_seeds(SEED)

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load manifest and classes
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise SystemExit(
            f"[ERROR] Missing {MANIFEST}. "
            "Generate the split manifest first."
        )

    with open(MANIFEST) as f:
        manifest = json.load(f)

    classes = sorted(manifest["vocabulary"])

    class_to_idx = {
        c: i
        for i, c in enumerate(classes)
    }

    print(f"Canonical class order: {classes}")

    # --------------------------------------------------------
    # Load train and validation datasets
    # --------------------------------------------------------

    X_train, y_train_s, p_train = load_npz(TRAIN_NPZ)
    X_val, y_val_s, p_val = load_npz(VAL_NPZ)

    y_train = to_int_labels(
        y_train_s,
        class_to_idx
    )

    y_val = to_int_labels(
        y_val_s,
        class_to_idx
    )

    print(
        f"Train: X={X_train.shape}, "
        f"y={Counter(y_train.tolist())}, "
        f"participants={sorted(set(p_train.tolist()))}"
    )

    print(
        f"Val  : X={X_val.shape}, "
        f"y={Counter(y_val.tolist())}, "
        f"participants={sorted(set(p_val.tolist()))}"
    )

    # --------------------------------------------------------
    # Basic dataset validation
    # --------------------------------------------------------

    if X_train.ndim != 2 or X_train.shape[1] != 63:
        raise SystemExit(
            f"[ERROR] Expected train features shape (N, 63), "
            f"got {X_train.shape}"
        )

    if X_val.ndim != 2 or X_val.shape[1] != 63:
        raise SystemExit(
            f"[ERROR] Expected validation features shape (N, 63), "
            f"got {X_val.shape}"
        )

    if len(X_train) == 0 or len(X_val) == 0:
        raise SystemExit(
            "[ERROR] Train or validation dataset is empty."
        )

    if len(X_train) != len(y_train):
        raise SystemExit(
            "[ERROR] Train feature and label counts do not match."
        )

    if len(X_val) != len(y_val):
        raise SystemExit(
            "[ERROR] Validation feature and label counts do not match."
        )

    # --------------------------------------------------------
    # Check participant leakage
    # --------------------------------------------------------

    overlap = (
        set(p_train.tolist())
        &
        set(p_val.tolist())
    )

    if overlap:
        raise SystemExit(
            "[ERROR] Participant leak between train and validation: "
            f"{sorted(overlap)}"
        )

    # --------------------------------------------------------
    # Save canonical label ordering
    # --------------------------------------------------------

    with open(LABEL_ORDER_OUT, "w") as f:

        json.dump(
            {
                "classes": classes,
                "indices": class_to_idx,
                "source": (
                    "tools/artifacts/split_manifest.json "
                    "→ vocabulary (sorted)"
                ),
                "must_match_in": [
                    "tools/train.py",
                    "tools/export_tflite.py",
                    "android/app/src/main/assets/labels.json",
                    "Room sign table ordering",
                ],
            },
            f,
            indent=2,
        )

    print(f"Wrote {LABEL_ORDER_OUT}")

    # ========================================================
    # MLP TRAINING
    # ========================================================

    print(
        "\n=== Training MLP "
        "(label smoothing + dropout + best-loss checkpoint) ==="
    )

    model = build_mlp(
        num_classes=len(classes)
    )

    model.summary()

    param_count = model.count_params()

    # --------------------------------------------------------
    # Best-loss checkpoint
    # --------------------------------------------------------

    ckpt_best = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_BEST),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    # --------------------------------------------------------
    # Final-epoch checkpoint
    # --------------------------------------------------------

    ckpt_final = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_FINAL),
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        save_freq="epoch",
        verbose=0,
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=False,
        verbose=1,
    )

    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            ckpt_best,
            ckpt_final,
            early_stop
        ],
        shuffle=True,
        verbose=2,
    )

    # --------------------------------------------------------
    # Best-loss epoch
    # --------------------------------------------------------

    val_losses = history.history["val_loss"]

    best_loss_epoch = int(
        np.argmin(val_losses)
    )

    epochs_trained = len(val_losses)

    print(
        f"\nBest-loss epoch: "
        f"{best_loss_epoch + 1}/{epochs_trained} "
        f"val_loss={val_losses[best_loss_epoch]:.4f}"
    )

    # --------------------------------------------------------
    # Load saved models
    # compile=False because evaluation only uses predict().
    # --------------------------------------------------------

    best_model = keras.models.load_model(
        MODEL_BEST,
        compile=False
    )

    final_model = keras.models.load_model(
        MODEL_FINAL,
        compile=False
    )

    # --------------------------------------------------------
    # Evaluate both models
    # --------------------------------------------------------

    best_summary = summarize_model(
        best_model,
        X_val,
        y_val,
        classes,
        "best_loss"
    )

    final_summary = summarize_model(
        final_model,
        X_val,
        y_val,
        classes,
        "final_epoch"
    )

    print("\n--- best-loss checkpoint on validation ---")

    print(
        f"  accuracy          : "
        f"{best_summary['val_accuracy']:.4f}"
    )

    print(
        f"  pure CE loss      : "
        f"{best_summary['val_loss_pure_ce']:.4f}"
    )

    print(
        f"  mean max-conf     : "
        f"{best_summary['val_conf_mean']:.4f}"
    )

    print(
        f"  5th / 95th conf   : "
        f"{best_summary['val_conf_p05']:.4f} / "
        f"{best_summary['val_conf_p95']:.4f}"
    )

    print("\n--- final-epoch checkpoint on validation ---")

    print(
        f"  accuracy          : "
        f"{final_summary['val_accuracy']:.4f}"
    )

    print(
        f"  pure CE loss      : "
        f"{final_summary['val_loss_pure_ce']:.4f}"
    )

    print(
        f"  mean max-conf     : "
        f"{final_summary['val_conf_mean']:.4f}"
    )

    print(
        f"  5th / 95th conf   : "
        f"{final_summary['val_conf_p05']:.4f} / "
        f"{final_summary['val_conf_p95']:.4f}"
    )

    # ========================================================
    # k-NN BASELINE
    # ========================================================

    print("\n=== Training k-NN baseline ===")

    knn = KNeighborsClassifier(
        n_neighbors=KNN_K,
        metric="euclidean",
        n_jobs=-1
    )

    knn.fit(
        X_train,
        y_train
    )

    knn_val_acc = float(
        knn.score(X_val, y_val)
    )

    knn_val_preds = knn.predict(X_val)

    cm_knn = confusion_matrix(
        y_val,
        knn_val_preds,
        labels=list(range(len(classes)))
    )

    print(
        f"k-NN (k={KNN_K}) "
        f"validation accuracy: {knn_val_acc:.4f}"
    )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(HISTORY_OUT, "w") as f:

        json.dump(
            history.history,
            f,
            indent=2
        )

    print(f"Wrote {HISTORY_OUT}")

    # ========================================================
    # TRAINING CURVES
    # ========================================================

    fig, (ax_loss, ax_acc) = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    ax_loss.plot(
        history.history["loss"],
        label="train"
    )

    ax_loss.plot(
        history.history["val_loss"],
        label="val"
    )

    ax_loss.axvline(
        best_loss_epoch,
        color="green",
        linestyle="--",
        label=f"best val_loss (epoch {best_loss_epoch + 1})"
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Smoothed CE loss")
    ax_loss.set_title(
        "Cross-entropy (label smoothing 0.05)"
    )

    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(
        history.history["accuracy"],
        label="train"
    )

    ax_acc.plot(
        history.history["val_accuracy"],
        label="val"
    )

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")

    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        CURVES_OUT,
        dpi=120
    )

    plt.close(fig)

    print(f"Wrote {CURVES_OUT}")

    # ========================================================
    # TRAINING REPORT
    # ========================================================

    report = {

        "seed": SEED,

        "class_order": classes,

        "train_split": {
            "n": int(len(X_train)),
            "participants": sorted(
                set(p_train.tolist())
            ),
            "class_counts": {
                classes[i]: int(v)
                for i, v in enumerate(
                    np.bincount(
                        y_train,
                        minlength=len(classes)
                    )
                )
            },
        },

        "validation_split": {
            "n": int(len(X_val)),
            "participants": sorted(
                set(p_val.tolist())
            ),
            "class_counts": {
                classes[i]: int(v)
                for i, v in enumerate(
                    np.bincount(
                        y_val,
                        minlength=len(classes)
                    )
                )
            },
        },

        "mlp": {
            "architecture": [
                63,
                64,
                32,
                len(classes)
            ],

            "dropout": DROPOUT_RATE,
            "label_smoothing": LABEL_SMOOTHING,
            "activation_hidden": "relu",
            "activation_output": "softmax",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "checkpoint_monitor": "val_loss (min)",
            "param_count": int(param_count),
            "epochs_trained": epochs_trained,
            "best_loss_epoch": best_loss_epoch + 1,
            "best_loss_value": float(
                val_losses[best_loss_epoch]
            ),

            "models": {
                "best_loss": best_summary,
                "final_epoch": final_summary,
            },
        },

        "knn": {
            "k": KNN_K,
            "metric": "euclidean",
            "val_accuracy": knn_val_acc,
            "val_confusion_matrix": cm_knn.tolist(),
        },

        "test_set_status": (
            "untouched — evaluated in Module 4 (evaluate.py)"
        ),
    }

    with open(REPORT_OUT, "w") as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(f"Wrote {REPORT_OUT}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n=== Summary ===")

    print(
        f"MLP best-loss acc : "
        f"{best_summary['val_accuracy']:.4f} "
        f"(mean max-conf {best_summary['val_conf_mean']:.4f})"
    )

    print(
        f"MLP final acc     : "
        f"{final_summary['val_accuracy']:.4f} "
        f"(mean max-conf {final_summary['val_conf_mean']:.4f})"
    )

    print(
        f"k-NN val acc      : "
        f"{knn_val_acc:.4f} "
        f"(k={KNN_K})"
    )

    print(
        f"Saved             : "
        f"{MODEL_BEST}, {MODEL_FINAL}"
    )

    print("Test set          : NOT touched")


if __name__ == "__main__":
    main()
