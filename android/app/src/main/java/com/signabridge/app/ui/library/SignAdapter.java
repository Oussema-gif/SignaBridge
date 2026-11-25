package com.signabridge.app.ui.library;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.signabridge.app.R;
import com.signabridge.app.data.SignRepository;
import com.signabridge.app.data.entities.Sign;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SignAdapter extends RecyclerView.Adapter<SignAdapter.SignViewHolder> {
    private List<Sign> originalSigns = new ArrayList<>();
    private List<Sign> filteredSigns = new ArrayList<>();
    private Set<String> favoriteCodes = new HashSet<>();
    private final SignRepository repository;

    public SignAdapter(SignRepository repository) {
        this.repository = repository;
    }

    public void setSigns(List<Sign> signs, Set<String> favorites) {
        this.originalSigns = signs;
        this.filteredSigns = new ArrayList<>(signs);
        this.favoriteCodes = favorites;
        notifyDataSetChanged();
    }

    public void filter(String query) {
        filteredSigns.clear();
        if (query.isEmpty()) {
            filteredSigns.addAll(originalSigns);
        } else {
            String lowerQuery = query.toLowerCase().trim();
            for (Sign sign : originalSigns) {
                if (sign.code.toLowerCase().contains(lowerQuery) || 
                    sign.label.toLowerCase().contains(lowerQuery)) {
                    filteredSigns.add(sign);
                }
            }
        }
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public SignViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_sign, parent, false);
        return new SignViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull SignViewHolder holder, int position) {
        Sign sign = filteredSigns.get(position);
        holder.code.setText(sign.code);
        holder.label.setText(sign.label);
        holder.category.setText(sign.category);

        boolean isFav = favoriteCodes.contains(sign.code);
        holder.star.setImageResource(isFav ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
        holder.star.setColorFilter(holder.itemView.getContext().getResources().getColor(isFav ? R.color.accent_mint : R.color.text_secondary));

        holder.star.setOnClickListener(v -> {
            boolean newFav = !favoriteCodes.contains(sign.code);
            repository.toggleFavorite(sign.code, isFav);
            if (newFav) favoriteCodes.add(sign.code);
            else favoriteCodes.remove(sign.code);
            notifyItemChanged(position);
        });
    }

    @Override
    public int getItemCount() {
        return filteredSigns.size();
    }

    static class SignViewHolder extends RecyclerView.ViewHolder {
        TextView code, label, category;
        ImageView star;
        public SignViewHolder(@NonNull View itemView) {
            super(itemView);
            code = itemView.findViewById(R.id.signCodeText);
            label = itemView.findViewById(R.id.signLabelText);
            category = itemView.findViewById(R.id.signCategoryText);
            star = itemView.findViewById(R.id.favoriteIcon);
        }
    }
}
