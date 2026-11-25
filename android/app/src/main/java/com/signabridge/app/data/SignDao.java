package com.signabridge.app.data;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import com.signabridge.app.data.entities.Sign;
import java.util.List;

@Dao
public interface SignDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insertAll(List<Sign> signs);

    @Query("SELECT * FROM signs")
    List<Sign> getAllSigns();
}