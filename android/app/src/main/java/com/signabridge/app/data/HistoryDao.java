package com.signabridge.app.data;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import com.signabridge.app.data.entities.HistoryItem;
import com.signabridge.app.data.entities.RecognitionHistory;
import java.util.List;

@Dao
public interface HistoryDao {
    @Insert
    void insert(RecognitionHistory history);

    @Query("SELECT s.code as signCode, s.label as label, h.confidence as confidence, h.recognizedAt as recognizedAt " +
           "FROM signs s INNER JOIN recognition_history h ON s.code = h.signCode " +
           "ORDER BY h.recognizedAt DESC")
    List<HistoryItem> getHistoryItems();

    @Query("SELECT * FROM recognition_history ORDER BY recognizedAt DESC LIMIT 1")
    RecognitionHistory getLatestHistory();

    @Query("DELETE FROM recognition_history")
    void clearHistory();

    // Keep only the 200 most recent records
    @Query("DELETE FROM recognition_history WHERE id NOT IN (SELECT id FROM recognition_history ORDER BY recognizedAt DESC LIMIT 200)")
    void trimHistory();
}