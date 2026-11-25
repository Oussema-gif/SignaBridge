package com.signabridge.app.ui.library;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.signabridge.app.R;
import com.signabridge.app.data.SignRepository;

public class LibraryFragment extends Fragment {
    private SignRepository repository;
    private SignAdapter adapter;
    private EditText searchInput;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_library, container, false);
        RecyclerView recyclerView = view.findViewById(R.id.libraryRecyclerView);
        searchInput = view.findViewById(R.id.searchInput);
        
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));

        repository = new SignRepository(requireContext());
        adapter = new SignAdapter(repository);
        recyclerView.setAdapter(adapter);

        searchInput.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {}
            @Override public void afterTextChanged(Editable s) {
                adapter.filter(s.toString());
            }
        });

        loadData();
        return view;
    }

    private void loadData() {
        repository.getAllSigns(signs -> repository.getFavoriteCodes(favorites -> {
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> adapter.setSigns(signs, favorites));
            }
        }));
    }

    @Override
    public void onResume() {
        super.onResume();
        loadData();
    }
}
