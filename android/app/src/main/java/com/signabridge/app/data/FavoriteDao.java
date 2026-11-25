package com.signabridge.app.data;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import com.signabridge.app.data.entities.Favorite;
import com.signabridge.app.data.entities.Sign;
import java.util.List;

@Dao
public interface FavoriteDao {
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    void insert(Favorite favorite);

    @Query("DELETE FROM favorites WHERE signCode = :signCode")
    void deleteBySignCode(String signCode);

    @Query("SELECT * FROM favorites")
    List<Favorite> getAllFavorites();

    @Query("SELECT s.* FROM signs s INNER JOIN favorites f ON s.code = f.signCode ORDER BY f.addedAt DESC")
    List<Sign> getFavoriteSigns();
}