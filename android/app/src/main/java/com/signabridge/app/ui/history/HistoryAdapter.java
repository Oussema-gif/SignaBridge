package com.signabridge.app.ui.history;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.signabridge.app.R;
import com.signabridge.app.data.entities.HistoryItem;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class HistoryAdapter extends RecyclerView.Adapter<HistoryAdapter.ViewHolder> {
    private List<HistoryItem> items = new ArrayList<>();
    private final SimpleDateFormat timeFormat = new SimpleDateFormat("MMM d, h:mm a", Locale.US);

    public void setItems(List<HistoryItem> items) {
        this.items = items;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        return new ViewHolder(LayoutInflater.from(parent.getContext()).inflate(R.layout.item_history, parent, false));
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        HistoryItem item = items.get(position);
        holder.code.setText(item.signCode);
        holder.label.setText(item.label);
        holder.time.setText(timeFormat.format(new Date(item.recognizedAt)) + " • " + Math.round(item.confidence * 100) + "%");
    }

    @Override
    public int getItemCount() { return items.size(); }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView code, label, time;
        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            code = itemView.findViewById(R.id.historyCode);
            label = itemView.findViewById(R.id.historyLabel);
            time = itemView.findViewById(R.id.historyTime);
        }
    }
}