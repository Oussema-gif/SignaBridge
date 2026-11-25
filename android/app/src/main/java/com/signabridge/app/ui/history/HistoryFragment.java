package com.signabridge.app.ui.history;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.signabridge.app.R;
import com.signabridge.app.data.SignRepository;

public class HistoryFragment extends Fragment {
    private SignRepository repository;
    private HistoryAdapter adapter;
    private RecyclerView recyclerView;
    private TextView emptyText;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_history, container, false);
        recyclerView = view.findViewById(R.id.historyRecyclerView);
        emptyText = view.findViewById(R.id.emptyHistoryText);
        
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        repository = new SignRepository(requireContext());
        adapter = new HistoryAdapter();
        recyclerView.setAdapter(adapter);

        view.findViewById(R.id.clearHistoryBtn).setOnClickListener(v -> {
            new AlertDialog.Builder(requireContext())
                .setTitle("Clear History")
                .setMessage("Are you sure you want to delete all recognition history?")
                .setPositiveButton("Clear", (dialog, which) -> {
                    repository.clearHistory();
                    loadData();
                })
                .setNegativeButton("Cancel", null)
                .show();
        });

        loadData();
        return view;
    }

    private void loadData() {
        repository.getHistoryItems(items -> {
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (items.isEmpty()) {
                        emptyText.setVisibility(View.VISIBLE);
                        recyclerView.setVisibility(View.GONE);
                    } else {
                        emptyText.setVisibility(View.GONE);
                        recyclerView.setVisibility(View.VISIBLE);
                        adapter.setItems(items);
                    }
                });
            }
        });
    }

    @Override
    public void onResume() {
        super.onResume();
        loadData();
    }
}