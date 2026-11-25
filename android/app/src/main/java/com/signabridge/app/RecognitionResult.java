package com.signabridge.app;

public class RecognitionResult {
    public enum State { NO_HAND, UNCERTAIN, RECOGNIZED }
    public final State state;
    public final String label;
    public final float confidence;

    public RecognitionResult(State state, String label, float confidence) {
        this.state = state;
        this.label = label;
        this.confidence = confidence;
    }

    public static RecognitionResult noHand() {
        return new RecognitionResult(State.NO_HAND, null, 0f);
    }
    public static RecognitionResult uncertain(String label, float confidence) {
        return new RecognitionResult(State.UNCERTAIN, label, confidence);
    }
    public static RecognitionResult recognized(String label, float confidence) {
        return new RecognitionResult(State.RECOGNIZED, label, confidence);
    }
}