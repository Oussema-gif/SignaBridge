package com.signabridge.app;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class FeatureExtractorTest {

    private static final float TOLERANCE = 1e-5f;

    @Test
    public void testPythonJavaParity() throws Exception {
        // Load parity_vectors.json from test resources
        InputStream is = getClass().getClassLoader().getResourceAsStream("parity_vectors.json");
        if (is == null) {
            fail("parity_vectors.json not found in test resources");
            return;
        }

        // Read InputStream into a String
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        
        JSONObject root = new JSONObject(sb.toString());
        JSONArray vectors = root.getJSONArray("vectors");

        int tested = 0;
        for (int i = 0; i < vectors.length(); i++) {
            JSONObject vec = vectors.getJSONObject(i);
            boolean shouldRaise = vec.getBoolean("should_raise");
            JSONArray lmJson = vec.getJSONArray("landmarks");
            
            float[][] landmarks = new float[21][3];
            for (int r = 0; r < 21; r++) {
                JSONArray pt = lmJson.getJSONArray(r);
                landmarks[r][0] = (float) pt.getDouble(0);
                landmarks[r][1] = (float) pt.getDouble(1);
                landmarks[r][2] = (float) pt.getDouble(2);
            }

            try {
                float[] result = FeatureExtractor.extract(landmarks);
                if (shouldRaise) {
                    fail("Expected exception for vector " + i + " but got none");
                }
                
                JSONArray expectedJson = vec.getJSONArray("expected_features");
                float[] expected = new float[63];
                for (int j = 0; j < 63; j++) {
                    expected[j] = (float) expectedJson.getDouble(j);
                }

                assertArrayEquals("Vector " + i + " (" + vec.getString("class") + ") mismatch", 
                        expected, result, TOLERANCE);
                tested++;

            } catch (FeatureExtractor.FeatureExtractionException e) {
                if (!shouldRaise) {
                    fail("Unexpected exception for vector " + i + ": " + e.getMessage());
                }
                // Expected exception, pass
            }
        }
        
        System.out.println("Parity test completed. " + tested + " vectors verified.");
        assertTrue("No vectors were tested", tested > 0);
    }
}