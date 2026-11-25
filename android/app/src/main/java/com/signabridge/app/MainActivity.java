package com.signabridge.app;

import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.NavigationUI;

import com.google.android.material.bottomnavigation.BottomNavigationView;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        View content = ((ViewGroup) findViewById(android.R.id.content)).getChildAt(0);
        View navHost = findViewById(R.id.nav_host_fragment);
        View navCard = findViewById(R.id.bottomNavCard);

        // Keep the floating navigation above gesture and three-button system bars.
        ViewCompat.setOnApplyWindowInsetsListener(content, (view, windowInsets) -> {
            Insets bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars());
            ViewGroup.MarginLayoutParams navMargins = (ViewGroup.MarginLayoutParams) navCard.getLayoutParams();
            navMargins.bottomMargin = dp(10) + bars.bottom;
            navCard.setLayoutParams(navMargins);

            ViewGroup.MarginLayoutParams hostMargins = (ViewGroup.MarginLayoutParams) navHost.getLayoutParams();
            hostMargins.bottomMargin = dp(94) + bars.bottom;
            navHost.setLayoutParams(hostMargins);
            return windowInsets;
        });
        ViewCompat.requestApplyInsets(content);

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment);
        if (navHostFragment != null) {
            NavController navController = navHostFragment.getNavController();
            BottomNavigationView bottomNav = findViewById(R.id.bottom_nav);
            NavigationUI.setupWithNavController(bottomNav, navController);
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
