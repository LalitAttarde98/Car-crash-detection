import torch
import numpy as np

def collate_and_prepare_batch(batch, class_map):

    max_tracks = max((len(item['tracks']) for item in batch), default=0)
    max_tracks = max(max_tracks, 1) 
    
    window_size = int(next(iter(batch[0]['tracks'].values()))[-1][0]) + 1
    feature_dim = int(next(iter(batch[0]['tracks'].values()))[0][1].shape[0])
    batch_size = len(batch)

    x_padded = torch.zeros((batch_size, max_tracks, window_size, feature_dim), dtype=torch.float32)
    track_padding_mask = torch.ones((batch_size, max_tracks), dtype=torch.bool)

    is_accident = torch.zeros(batch_size, dtype=torch.float32)
    location = torch.zeros((batch_size, max_tracks), dtype=torch.float32)
    accident_type_list = []

    for b, item in enumerate(batch):
        is_accident[b] = float(item['is_accident'])
        accident_type_list.append(item['type'])
        anomalous_ids = set(item['anomalous_track_ids'])

        sorted_tracks = sorted(item['tracks'].items()) #

        for track_idx, (track_id, features) in enumerate(sorted_tracks):
            track_padding_mask[b, track_idx] = False

            if track_id in anomalous_ids:
                location[b, track_idx] = 1.0

            if features:
                frames = [int(f[0]) for f in features if 0 <= int(f[0]) < window_size]
                vecs = [f[1] for f in features if 0 <= int(f[0]) < window_size]
                
                if frames:
                    x_padded[b, track_idx, frames] = torch.tensor(np.array(vecs), dtype=torch.float32)
                
    type_targets = torch.tensor([class_map[t] for t in accident_type_list], dtype=torch.long)

    return {
        'x': x_padded,
        'track_padding_mask': track_padding_mask,
        'is_accident_targets': is_accident,
        'location_targets': location,
        'type_targets': type_targets,
    }