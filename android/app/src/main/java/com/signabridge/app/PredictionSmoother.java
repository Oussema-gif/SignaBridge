package com.signabridge.app;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;

/**
 * SignaBridge — Prediction stability state machine.
 * Requires 7 out of the last 10 frames to agree.
 */
public class PredictionSmoother {
    private static final int WINDOW_SIZE = 10;
    private static final int AGREEMENT_THRESHOLD = 7;
    private static final float MIN_CONFIDENCE = 0.9488f;

    private final ArrayDeque<String> history = new ArrayDeque<>();
    private String lastRecognized = null;

    public RecognitionResult process(String predictedLabel, float confidence) {
        if (confidence < MIN_CONFIDENCE) {
            history.clear();
            lastRecognized = null;
            return RecognitionResult.uncertain(predictedLabel, confidence);
        }

        history.addLast(predictedLabel);
        if (history.size() > WINDOW_SIZE) {
            history.removeFirst();
        }

        // Count frequencies
        Map<String, Integer> counts = new HashMap<>();
        for (String label : history) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        // Find the most frequent label
        String bestLabel = null;
        int bestCount = 0;
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > bestCount) {
                bestCount = entry.getValue();
                bestLabel = entry.getKey();
            }
        }

        // Check for agreement
        if (bestCount >= AGREEMENT_THRESHOLD) {
            // Only trigger a new recognition event if the label changed
            if (!bestLabel.equals(lastRecognized)) {
                lastRecognized = bestLabel;
                return RecognitionResult.recognized(bestLabel, confidence);
            }
            return RecognitionResult.recognized(bestLabel, confidence);
        }

        return RecognitionResult.uncertain(predictedLabel, confidence);
    }

    public RecognitionResult onNoHand() {
        history.clear();
        lastRecognized = null;
        return RecognitionResult.noHand();
    }
}