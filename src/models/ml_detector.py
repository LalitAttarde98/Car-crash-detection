import torch.nn as nn

class AccidentTrajectoryPredictor(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=64, num_classes=10, num_heads=2, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Linear(feature_dim, hidden_dim)

        temp_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temp_layer, num_layers=num_layers)

        self.spatial_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, track_padding_mask=None):
        B, T, F, C = x.shape

        x = self.embedding(x)  

        x_temp = x.view(B * T, F, -1) 
        temp_encoded = self.temporal_transformer(x_temp) 
        temp_encoded = temp_encoded.view(B, T, F, -1) 

        frame_features = temp_encoded.max(dim=1)[0] # (B, F, hidden) extreme frame
        time_logits = self.time_head(frame_features).squeeze(-1) 

        track_features = temp_encoded.max(dim=2)[0] # (B, T, hidden) extreme track
        
        spatial_out, _ = self.spatial_attention(
            query=track_features, 
            key=track_features, 
            value=track_features,
            key_padding_mask=track_padding_mask
        )
        track_features = track_features + spatial_out 

        location_logits = self.location_head(track_features).squeeze(-1) #(B, T)

        video_feature = track_features.max(dim=1)[0] # (B, hidden) extreme object
        type_logits = self.type_head(video_feature) 

        return {
            'time_logits': time_logits,         
            'location_logits': location_logits, 
            'type_logits': type_logits        
        }