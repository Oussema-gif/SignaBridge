package com.signabridge.app;

import androidx.annotation.NonNull;

/**
 * SignaBridge — Java implementation of the 63-D feature contract.
 * Must exactly mirror tools/feature_extractor.py.
 */
public class FeatureExtractor {

    public static final int NUM_LANDMARKS = 21;
    public static final int FEATURE_DIM = NUM_LANDMARKS * 3; // 63
    private static final int WRIST_INDEX = 0;
    private static final int MIDDLE_MCP_INDEX = 9;
    private static final float MIN_SCALE = 1e-6f;

    public static class FeatureExtractionException extends Exception {
        public FeatureExtractionException(String message) {
            super(message);
        }
    }

    /**
     * Extracts features from a 21x3 array of normalized landmarks.
     *
     * @param landmarks 2D array [21][3] (x, y, z)
     * @return 1D array of 63 float features
     * @throws FeatureExtractionException if landmarks are invalid or degenerate
     */
    public static float[] extract(float[][] landmarks) throws FeatureExtractionException {
        if (landmarks == null || landmarks.length != NUM_LANDMARKS) {
            throw new FeatureExtractionException("Expected 21 landmarks.");
        }

        float[] features = new float[FEATURE_DIM];

        // 1. Validate and locate wrist & middle MCP
        float wristX = landmarks[WRIST_INDEX][0];
        float wristY = landmarks[WRIST_INDEX][1];
        float wristZ = landmarks[WRIST_INDEX][2];

        if (Float.isNaN(wristX) || Float.isNaN(wristY) || Float.isNaN(wristZ) ||
            Float.isInfinite(wristX) || Float.isInfinite(wristY) || Float.isInfinite(wristZ)) {
            throw new FeatureExtractionException("Landmark contains NaN or Inf.");
        }

        float mcpX = landmarks[MIDDLE_MCP_INDEX][0] - wristX;
        float mcpY = landmarks[MIDDLE_MCP_INDEX][1] - wristY;
        float mcpZ = landmarks[MIDDLE_MCP_INDEX][2] - wristZ;

        // 2. Compute scale (Euclidean distance)
        float scale = (float) Math.sqrt(mcpX * mcpX + mcpY * mcpY + mcpZ * mcpZ);
        if (scale < MIN_SCALE) {
            throw new FeatureExtractionException("Degenerate hand scale: " + scale);
        }

        // 3. Translate and scale
        int outIdx = 0;
        for (int i = 0; i < NUM_LANDMARKS; i++) {
            float x = landmarks[i][0];
            float y = landmarks[i][1];
            float z = landmarks[i][2];

            if (Float.isNaN(x) || Float.isNaN(y) || Float.isNaN(z) ||
                Float.isInfinite(x) || Float.isInfinite(y) || Float.isInfinite(z)) {
                throw new FeatureExtractionException("Landmark " + i + " contains NaN or Inf.");
            }

            features[outIdx++] = (x - wristX) / scale;
            features[outIdx++] = (y - wristY) / scale;
            features[outIdx++] = (z - wristZ) / scale;
        }

        return features;
    }
}