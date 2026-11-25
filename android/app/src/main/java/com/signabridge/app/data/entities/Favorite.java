package com.signabridge.app.data.entities;

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.Index;
import androidx.room.PrimaryKey;

@Entity(tableName = "favorites",
        foreignKeys = @ForeignKey(entity = Sign.class, parentColumns = "code", childColumns = "signCode", onDelete = ForeignKey.CASCADE),
        indices = {@Index("signCode")}) // Added index to prevent table scan warnings
public class Favorite {
    @PrimaryKey(autoGenerate = true)
    public int id;
    
    @NonNull
    public String signCode;
    public long addedAt;

    public Favorite(@NonNull String signCode, long addedAt) {
        this.signCode = signCode;
        this.addedAt = addedAt;
    }
}