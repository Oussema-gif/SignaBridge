#!/usr/bin/env python3
"""
SignaBridge — Export Keras model to TFLite.
Uses model_final.keras (selected as the shipping checkpoint by evaluate.py).
"""

import json
import sys
from pathlib import Path
import tensorflow as tf

ARTIFACTS = Path("tools/artifacts")
MODEL_KERAS = ARTIFACTS / "model_final.keras"
MODEL_TFLITE = ARTIFACTS / "gesture_classifier.tflite"
ANDROID_ASSETS = Path("android/app/src/main/assets")

def main():
    if not MODEL_KERAS.exists():
        raise SystemExit(f"[ERROR] Missing {MODEL_KERAS}. Run train.py first.")

    print(f"Loading {MODEL_KERAS}...")
    model = tf.keras.models.load_model(
        MODEL_KERAS,
        compile=False,
        custom_objects={"smoothed_sparse_ce": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}
    )

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Disable quantization to preserve exact float32 parity with Python
    converter.optimizations = [] 
    tflite_model = converter.convert()

    MODEL_TFLITE.write_bytes(tflite_model)
    print(f"Saved {MODEL_TFLITE} ({len(tflite_model)/1024:.1f} KB)")

    # Copy to Android assets
    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
    dest = ANDROID_ASSETS / "gesture_classifier.tflite"
    dest.write_bytes(tflite_model)
    print(f"Copied to {dest}")

    # Generate a small metadata file for Android to validate shapes
    input_shape = model.input_shape[1:]  # (63,)
    output_shape = model.output_shape[1:] # (5,)
    metadata = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "classes": json.load(open(ARTIFACTS / "label_order.json"))["classes"],
        "model_source": str(MODEL_KERAS),
    }
    (ANDROID_ASSETS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {ANDROID_ASSETS / 'model_metadata.json'}")

if __name__ == "__main__":
    main()