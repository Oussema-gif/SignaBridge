package com.signabridge.app.ui.recognize;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.DecelerateInterpolator;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import com.signabridge.app.FeatureExtractor;
import com.signabridge.app.LandmarkOverlayView;
import com.signabridge.app.PredictionSmoother;
import com.signabridge.app.R;
import com.signabridge.app.RecognitionResult;
import com.signabridge.app.ScanningLineView;
import com.signabridge.app.TfliteGestureClassifier;
import com.signabridge.app.data.SignRepository;
import com.signabridge.app.services.TtsManager;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RecognizeFragment extends Fragment {
    private static final String TAG = "SignaBridge/Fragment";
    private PreviewView previewView;
    private LandmarkOverlayView overlayView;
    private ScanningLineView scanningLine;
    private TextView statusText;
    private TextView recognizedSignText;
    private TextView confidenceLabel;
    private MaterialButton speakButton;

    private ExecutorService analysisExecutor;
    private ProcessCameraProvider cameraProvider;
    private HandLandmarker handLandmarker;
    private TfliteGestureClassifier gestureClassifier;
    private PredictionSmoother smoother;
    
    private TtsManager ttsManager;
    private SignRepository repository;
    private String currentRecognizedSign = null;

    private boolean isFrontCamera = true;
    private ActivityResultLauncher<String> cameraPermissionLauncher;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_recognize, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        previewView = view.findViewById(R.id.previewView);
        overlayView = view.findViewById(R.id.overlayView);
        scanningLine = view.findViewById(R.id.scanningLine);
        statusText = view.findViewById(R.id.statusText);
        recognizedSignText = view.findViewById(R.id.recognizedSignText);
        confidenceLabel = view.findViewById(R.id.confidenceLabel);
        speakButton = view.findViewById(R.id.speakButton);
        playEntranceAnimations(view);

        analysisExecutor = Executors.newSingleThreadExecutor();
        smoother = new PredictionSmoother();
        ttsManager = new TtsManager(requireContext());
        repository = new SignRepository(requireContext());

        speakButton.setOnClickListener(v -> {
            if (currentRecognizedSign != null && ttsManager.isReady()) {
                ttsManager.speak(currentRecognizedSign);
            }
        });

        cameraPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                granted -> {
                    if (granted) startCamera();
                    else Toast.makeText(getContext(), "Camera permission required", Toast.LENGTH_LONG).show();
                }
        );

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        if (getContext() == null) return;
        ListenableFuture<ProcessCameraProvider> providerFuture = ProcessCameraProvider.getInstance(requireContext());
        providerFuture.addListener(() -> {
            try {
                cameraProvider = providerFuture.get();
                initRecognizers();
                bindUseCases();
                statusText.setText("Ready");
                statusText.postDelayed(() -> statusText.setVisibility(View.GONE), 2000);
            } catch (Exception e) {
                Log.e(TAG, "Init failed", e);
                statusText.setText("Error: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    private void initRecognizers() throws Exception {
        BaseOptions baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .setDelegate(Delegate.CPU)
                .build();

        HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setResultListener(this::onHandLandmarkerResult)
                .setErrorListener(e -> Log.e(TAG, "MP Error", e))
                .build();
        handLandmarker = HandLandmarker.createFromOptions(requireContext(), options);
        gestureClassifier = new TfliteGestureClassifier(requireContext());
    }

    private void bindUseCases() {
        if (cameraProvider == null) return;
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        previewView.setScaleX(1f);

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        analysis.setAnalyzer(analysisExecutor, this::analyzeFrame);

        CameraSelector selector = isFrontCamera ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(getViewLifecycleOwner(), selector, preview, analysis);
    }

    private void analyzeFrame(@NonNull ImageProxy image) {
        try {
            if (handLandmarker == null) return;
            Bitmap bmp = image.toBitmap();
            int rotation = image.getImageInfo().getRotationDegrees();
            if (rotation != 0) {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotation);
                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
            }
            MPImage mpImage = new BitmapImageBuilder(bmp).build();
            handLandmarker.detectAsync(mpImage, SystemClock.uptimeMillis());
        } catch (Exception e) {
            Log.e(TAG, "analyzeFrame error", e);
        } finally {
            image.close();
        }
    }

    private void onHandLandmarkerResult(HandLandmarkerResult result, MPImage input) {
        if (result == null || result.landmarks().isEmpty()) {
            updateUI(smoother.onNoHand(), null);
            return;
        }

        List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> lms = result.landmarks().get(0);
        float[][] landmarks = new float[21][3];
        List<float[]> overlayPoints = new ArrayList<>(21);
        
        for (int i = 0; i < 21; i++) {
            float x = lms.get(i).x();
            float y = lms.get(i).y();
            landmarks[i][0] = x;
            landmarks[i][1] = y;
            landmarks[i][2] = lms.get(i).z();
            float overlayX = isFrontCamera ? (1.0f - x) : x;
            overlayPoints.add(new float[]{overlayX, y});
        }

        float[] features;
        try {
            features = FeatureExtractor.extract(landmarks);
        } catch (FeatureExtractor.FeatureExtractionException e) {
            updateUI(smoother.onNoHand(), overlayPoints);
            return;
        }

        float[] probs = gestureClassifier.classify(features);
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
        
        String predictedLabel = gestureClassifier.getLabel(maxIdx);
        float confidence = probs[maxIdx];

        updateUI(smoother.process(predictedLabel, confidence), overlayPoints);
    }

    private void updateUI(RecognitionResult result, List<float[]> overlayPoints) {
        if (getActivity() == null) return;
        getActivity().runOnUiThread(() -> {
            overlayView.setLandmarks(overlayPoints);
            switch (result.state) {
                case RECOGNIZED:
                    currentRecognizedSign = result.label;
                    recognizedSignText.setText(result.label);
                    recognizedSignText.setTextColor(getResources().getColor(R.color.accent_mint));
                    confidenceLabel.setText("Recognized");
                    scanningLine.setScanning(false);
                    speakButton.setEnabled(true);
                    repository.addHistory(result.label, result.confidence);
                    break;
                case UNCERTAIN:
                    currentRecognizedSign = null;
                    recognizedSignText.setText("?");
                    recognizedSignText.setTextColor(getResources().getColor(R.color.text_secondary));
                    confidenceLabel.setText("Analyzing gesture...");
                    scanningLine.setScanning(true);
                    speakButton.setEnabled(false);
                    break;
                case NO_HAND:
                default:
                    currentRecognizedSign = null;
                    recognizedSignText.setText("—");
                    recognizedSignText.setTextColor(getResources().getColor(R.color.accent_mint));
                    confidenceLabel.setText("Waiting for hand...");
                    scanningLine.setScanning(true);
                    speakButton.setEnabled(false);
                    break;
            }
        });
    }

    private void playEntranceAnimations(View root) {
        int[] ids = {R.id.headerTitle, R.id.headerSubtitle, R.id.cameraContainer, R.id.resultCard, R.id.speakButton};
        for (int i = 0; i < ids.length; i++) {
            View target = root.findViewById(ids[i]);
            if (target == null) continue;
            target.setAlpha(0f);
            target.setTranslationY(14f);
            target.animate()
                    .alpha(1f)
                    .translationY(0f)
                    .setStartDelay(i * 55L)
                    .setDuration(360L)
                    .setInterpolator(new DecelerateInterpolator())
                    .start();
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (cameraProvider != null) cameraProvider.unbindAll();
        if (analysisExecutor != null) analysisExecutor.shutdown();
        if (handLandmarker != null) { handLandmarker.close(); handLandmarker = null; }
        if (gestureClassifier != null) { gestureClassifier.close(); gestureClassifier = null; }
        if (ttsManager != null) { ttsManager.shutdown(); ttsManager = null; }
    }
}
