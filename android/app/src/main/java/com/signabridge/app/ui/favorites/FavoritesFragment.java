package com.signabridge.app.ui.favorites;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.signabridge.app.R;
import com.signabridge.app.data.SignRepository;
import com.signabridge.app.ui.library.SignAdapter;
import java.util.HashSet;

public class FavoritesFragment extends Fragment {
    private SignRepository repository;
    private SignAdapter adapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_favorites, container, false);
        RecyclerView recyclerView = view.findViewById(R.id.favRecyclerView);
        TextView emptyText = view.findViewById(R.id.emptyFavText);
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));

        repository = new SignRepository(requireContext());
        adapter = new SignAdapter(repository);
        recyclerView.setAdapter(adapter);

        repository.getFavoriteSigns(signs -> {
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    if (signs.isEmpty()) {
                        emptyText.setVisibility(View.VISIBLE);
                        recyclerView.setVisibility(View.GONE);
                    } else {
                        emptyText.setVisibility(View.GONE);
                        recyclerView.setVisibility(View.VISIBLE);
                        adapter.setSigns(signs, new HashSet<>()); // Favorites all true by definition here, but handled by adapter
                    }
                });
            }
        });
        return view;
    }
}