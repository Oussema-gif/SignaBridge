package com.signabridge.app.services;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import java.util.Locale;

public class TtsManager {
    private static final String TAG = "SignaBridge/TTS";
    private TextToSpeech tts;
    private boolean isInitialized = false;
    private boolean isLanguageAvailable = false;

    public TtsManager(Context context) {
        tts = new TextToSpeech(context, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int result = tts.setLanguage(Locale.US);
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "Language not supported");
                } else {
                    isLanguageAvailable = true;
                }
                isInitialized = true;
            } else {
                Log.e(TAG, "TTS Initialization failed");
            }
        });
    }

    public void speak(String text) {
        if (isInitialized && isLanguageAvailable) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "SignaBridge_Utterance");
        }
    }

    public boolean isReady() {
        return isInitialized && isLanguageAvailable;
    }

    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
    }
}