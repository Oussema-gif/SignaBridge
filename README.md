# SignaBridge

SignaBridge is a premium, fully offline Android application that recognizes a limited vocabulary of static American Sign Language (ASL) hand signs using the smartphone camera. 

## Features
- **Real-time Recognition:** On-device inference using MediaPipe and TensorFlow Lite.
- **Sign Library:** Browse the supported vocabulary (A, B, L, V, Y).
- **Favorites:** Save frequently used signs for quick reference.
- **History:** View past recognitions with confidence scores.
- **Text-to-Speech:** Read recognized signs aloud.
- **Privacy-First:** 100% offline. No camera frames, images, or data ever leave the device.
- **Premium UI:** Material Design 3, Light/Dark themes, custom branding.

## Architecture
- **Language:** Java
- **Camera:** CameraX
- **Computer Vision:** MediaPipe Tasks Vision (HandLandmarker)
- **Machine Learning:** TensorFlow Lite (MLP Classifier)
- **Local Database:** Room over SQLite
- **Architecture:** MVVM (Lightweight)

## Installation
1. Clone the repository.
2. Open the `android/` folder in Android Studio.
3. Build and install on an Android device (Min SDK 28).
4. *(Optional)* To retrain the model, see `ML_README.md`.

## Usage
1. Grant camera permission.
2. Position your hand within the camera frame.
3. Hold a supported ASL sign (A, B, L, V, Y).
4. The recognized letter will appear in the center badge.
5. Tap "Read Aloud" to hear the letter.

## Known Limitations
- **Limited Vocabulary:** Only 5 static signs (A, B, L, V, Y) are supported. It is not a full ASL translator.
- **Test Population:** The ML model was evaluated on 2 unseen signers (P9, P10) from the ASL-HG dataset.
- **Unknown Rejection:** Operates via a calibrated softmax threshold (0.9488) and a 7/10 frame smoothing window. Sustained non-target gestures may occasionally be misclassified if they mimic a target shape.
- **Hardware:** Requires an ARM64 Android device with a camera.