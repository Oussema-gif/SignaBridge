package com.signabridge.app;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class TfliteGestureClassifier {
    private static final String TAG = "SignaBridge/Classifier";
    private static final int INPUT_SIZE = 63;
    private static final int OUTPUT_SIZE = 5;

    private final Interpreter interpreter;
    private final ByteBuffer inputBuffer;
    private final float[][] outputBuffer = new float[1][OUTPUT_SIZE];
    private final List<String> labels = new ArrayList<>();

    public TfliteGestureClassifier(Context context) throws Exception {
        // 1. Load labels from assets
        try (InputStream is = context.getAssets().open("labels.json")) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            String json = new String(buffer, StandardCharsets.UTF_8);
            org.json.JSONObject root = new org.json.JSONObject(json);
            org.json.JSONArray classes = root.getJSONArray("classes");
            for (int i = 0; i < classes.length(); i++) {
                labels.add(classes.getString(i));
            }
        }
        Log.d(TAG, "Loaded labels: " + labels);

        // 2. Load model into a MappedByteBuffer
        MappedByteBuffer modelBuffer;
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd("gesture_classifier.tflite");
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

        // 3. Initialize Interpreter
        interpreter = new Interpreter(modelBuffer);

        // 4. Prepare reusable input buffer
        inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * 4); // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder());
    }

    /**
     * Runs inference on a 63-D feature vector.
     * @return float array of probabilities for the 5 classes.
     */
    public float[] classify(float[] features) {
        if (features.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Expected " + INPUT_SIZE + " features, got " + features.length);
        }

        inputBuffer.rewind();
        for (float f : features) {
            inputBuffer.putFloat(f);
        }

        interpreter.run(inputBuffer, outputBuffer);
        return outputBuffer[0];
    }

    public String getLabel(int index) {
        return labels.get(index);
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}