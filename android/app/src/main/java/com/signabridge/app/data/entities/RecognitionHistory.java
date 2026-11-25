package com.signabridge.app.data.entities;

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "recognition_history")
public class RecognitionHistory {
    @PrimaryKey(autoGenerate = true)
    public int id;
    
    @NonNull
    public String signCode;
    public float confidence;
    public long recognizedAt;

    public RecognitionHistory(@NonNull String signCode, float confidence, long recognizedAt) {
        this.signCode = signCode;
        this.confidence = confidence;
        this.recognizedAt = recognizedAt;
    }
}