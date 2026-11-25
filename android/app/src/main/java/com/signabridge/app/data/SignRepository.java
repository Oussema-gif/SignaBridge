package com.signabridge.app.data;

import android.content.Context;
import android.util.Log;
import com.signabridge.app.data.entities.Favorite;
import com.signabridge.app.data.entities.HistoryItem;
import com.signabridge.app.data.entities.RecognitionHistory;
import com.signabridge.app.data.entities.Sign;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SignRepository {
    private final SignDao signDao;
    private final FavoriteDao favoriteDao;
    private final HistoryDao historyDao;
    private final Context context;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public SignRepository(Context context) {
        this.context = context.getApplicationContext();
        AppDatabase db = AppDatabase.getInstance(this.context);
        this.signDao = db.signDao();
        this.favoriteDao = db.favoriteDao();
        this.historyDao = db.historyDao();
        seedDatabaseIfEmpty();
    }

    private void seedDatabaseIfEmpty() {
        executor.execute(() -> {
            if (signDao.getAllSigns().isEmpty()) {
                try {
                    InputStream is = context.getAssets().open("signs.json");
                    byte[] buffer = new byte[is.available()];
                    is.read(buffer);
                    String json = new String(buffer, StandardCharsets.UTF_8);
                    JSONObject root = new JSONObject(json);
                    JSONArray classes = root.getJSONArray("classes");
                    List<Sign> signs = new ArrayList<>();
                    for (int i = 0; i < classes.length(); i++) {
                        JSONObject obj = classes.getJSONObject(i);
                        signs.add(new Sign(obj.getString("code"), obj.getString("label"), obj.getString("description"), obj.getString("category")));
                    }
                    signDao.insertAll(signs);
                } catch (Exception e) {
                    Log.e("SignaBridge/DB", "Failed to seed", e);
                }
            }
        });
    }

    public void getAllSigns(Callback<List<Sign>> callback) {
        executor.execute(() -> callback.onResult(signDao.getAllSigns()));
    }

    public void getFavoriteSigns(Callback<List<Sign>> callback) {
        executor.execute(() -> callback.onResult(favoriteDao.getFavoriteSigns()));
    }

    public void getFavoriteCodes(Callback<Set<String>> callback) {
        executor.execute(() -> {
            List<Favorite> favs = favoriteDao.getAllFavorites();
            Set<String> codes = new HashSet<>();
            for (Favorite f : favs) codes.add(f.signCode);
            callback.onResult(codes);
        });
    }

    public void toggleFavorite(String signCode, boolean isFavorite) {
        executor.execute(() -> {
            if (isFavorite) {
                favoriteDao.deleteBySignCode(signCode);
            } else {
                favoriteDao.insert(new Favorite(signCode, System.currentTimeMillis()));
            }
        });
    }

    public void addHistory(String signCode, float confidence) {
        executor.execute(() -> {
            RecognitionHistory latest = historyDao.getLatestHistory();
            long now = System.currentTimeMillis();
            // Deduplicate within 2 seconds
            if (latest != null && latest.signCode.equals(signCode) && (now - latest.recognizedAt) < 2000) {
                return;
            }
            historyDao.insert(new RecognitionHistory(signCode, confidence, now));
            historyDao.trimHistory();
        });
    }

    public void getHistoryItems(Callback<List<HistoryItem>> callback) {
        executor.execute(() -> callback.onResult(historyDao.getHistoryItems()));
    }

    public void clearHistory() {
        executor.execute(() -> historyDao.clearHistory());
    }

    public interface Callback<T> {
        void onResult(T result);
    }
}