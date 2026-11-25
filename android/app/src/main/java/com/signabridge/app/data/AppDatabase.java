package com.signabridge.app.data;

import android.content.Context;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import com.signabridge.app.data.entities.Favorite;
import com.signabridge.app.data.entities.RecognitionHistory;
import com.signabridge.app.data.entities.Sign;

@Database(entities = {Sign.class, Favorite.class, RecognitionHistory.class}, version = 1, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {
    private static AppDatabase instance;

    public abstract SignDao signDao();
    public abstract FavoriteDao favoriteDao();
    public abstract HistoryDao historyDao();

    public static synchronized AppDatabase getInstance(Context context) {
        if (instance == null) {
            instance = Room.databaseBuilder(context.getApplicationContext(),
                    AppDatabase.class, "signabridge_database")
                    .fallbackToDestructiveMigration()
                    .build();
        }
        return instance;
    }
}