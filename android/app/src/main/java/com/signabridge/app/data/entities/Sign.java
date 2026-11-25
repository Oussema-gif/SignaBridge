package com.signabridge.app.data.entities;

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "signs")
public class Sign {
    @PrimaryKey
    @NonNull
    public String code; // "A", "B", "L", "V", "Y"
    public String label;
    public String description;
    public String category;

    public Sign(@NonNull String code, String label, String description, String category) {
        this.code = code;
        this.label = label;
        this.description = description;
        this.category = category;
    }
}