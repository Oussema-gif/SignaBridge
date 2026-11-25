package com.signabridge.app;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;

public class ScanningLineView extends View {
    private Paint paint;
    private float lineY = 0f;
    private ValueAnimator animator;
    private boolean isScanning = true;

    public ScanningLineView(Context context, AttributeSet attrs) {
        super(context, attrs);
        paint = new Paint();
        paint.setStrokeWidth(6f);
        paint.setAntiAlias(true);
        paint.setShader(new LinearGradient(0, 0, 0, 150,
                new int[]{Color.TRANSPARENT, Color.parseColor("#35E6C0"), Color.TRANSPARENT},
                null, Shader.TileMode.CLAMP));
    }

    public void setScanning(boolean scanning) {
        this.isScanning = scanning;
        if (scanning) startAnimation(); else stopAnimation();
    }

    private void startAnimation() {
        if (animator != null && animator.isRunning()) return;
        animator = ValueAnimator.ofFloat(0f, 1f);
        animator.setDuration(2000);
        animator.setRepeatMode(ValueAnimator.REVERSE);
        animator.setRepeatCount(ValueAnimator.INFINITE);
        animator.setInterpolator(new LinearInterpolator());
        animator.addUpdateListener(animation -> {
            lineY = (float) animation.getAnimatedValue() * getHeight();
            invalidate();
        });
        animator.start();
    }

    private void stopAnimation() {
        if (animator != null) {
            animator.cancel();
            lineY = -100f; // Hide line
            invalidate();
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (isScanning) {
            canvas.drawLine(0, lineY, getWidth(), lineY, paint);
        }
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        if (isScanning) startAnimation();
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        stopAnimation();
    }
}
