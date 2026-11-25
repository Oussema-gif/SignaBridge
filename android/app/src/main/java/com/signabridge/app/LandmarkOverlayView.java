package com.signabridge.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.Collections;
import java.util.List;

public class LandmarkOverlayView extends View {
    private final Paint linePaint = new Paint();
    private final Paint pointPaint = new Paint();
    private List<float[]> landmarks = Collections.emptyList();

    private static final int[][] CONNECTIONS = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4},
            {0, 5}, {5, 6}, {6, 7}, {7, 8},
            {5, 9}, {9, 10}, {10, 11}, {11, 12},
            {9, 13}, {13, 14}, {14, 15}, {15, 16},
            {13, 17}, {17, 18}, {18, 19}, {19, 20},
            {0, 17}
    };

    public LandmarkOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        float density = context.getResources().getDisplayMetrics().density;

        // Neon Cyan Lines
        linePaint.setColor(Color.parseColor("#00F0FF"));
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(3f * density);
        linePaint.setAntiAlias(true);
        linePaint.setShadowLayer(8f, 0f, 0f, Color.parseColor("#00F0FF")); // Glow effect

        // White Nodes
        pointPaint.setColor(Color.WHITE);
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setAntiAlias(true);
    }

    public void setLandmarks(List<float[]> newLandmarks) {
        this.landmarks = newLandmarks == null ? Collections.emptyList() : newLandmarks;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (landmarks.isEmpty() || landmarks.size() < 21) return;

        float width = getWidth();
        float height = getHeight();

        for (int[] connection : CONNECTIONS) {
            float[] start = landmarks.get(connection[0]);
            float[] end = landmarks.get(connection[1]);
            canvas.drawLine(start[0] * width, start[1] * height, end[0] * width, end[1] * height, linePaint);
        }

        float pointRadius = 3f * getResources().getDisplayMetrics().density;
        for (float[] lm : landmarks) {
            canvas.drawCircle(lm[0] * width, lm[1] * height, pointRadius, pointPaint);
        }
    }
}
