import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataloader.accident_dataset import AccidentDataset
from dataloader.collate_fn import collate_and_prepare_batch
from models.ml_detector import AccidentTrajectoryPredictor
from configs.accident_prediction_config import data_cfg, train_cfg, model_cfg


def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AccidentDataset(**data_cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        collate_fn=lambda batch: collate_and_prepare_batch(batch, class_map=data_cfg['class_map'])
    )
    num_batches = len(data_loader)

    model = AccidentTrajectoryPredictor(**model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    if os.path.exists(train_cfg['ckpt_path']):
        checkpoint = torch.load(train_cfg['ckpt_path'], map_location=torch.device(device)) 
        model.load_state_dict(checkpoint)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(train_cfg['epochs']):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")

        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            track_padding_mask = batch['track_padding_mask'].to(device)
            is_accident_targets = batch['is_accident_targets'].to(device)
            location_targets = batch['location_targets'].to(device)            
            type_targets = batch['type_targets'].to(device)

            outputs = model(x, track_padding_mask=track_padding_mask)

            time_logits = outputs['time_logits']
            loc_logits = outputs['location_logits']
            type_logits = outputs['type_logits']

            window_anomaly_pred = time_logits.max(dim=1)[0]
            loss_anomaly = bce_loss(window_anomaly_pred, is_accident_targets)
            loss_type = ce_loss(type_logits, type_targets)

            mask = ~track_padding_mask
            if mask.any():
                loc_logits = loc_logits[mask]
                location_targets = location_targets[mask]
                loss_location = bce_loss(loc_logits, location_targets)
            else:
                loss_location = torch.tensor(0.0, device=device)

            loss = (train_cfg['loss_weights']['anomaly'] * loss_anomaly +
                          train_cfg['loss_weights']['type'] * loss_type +
                          train_cfg['loss_weights']['location'] * loss_location)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Anomaly': f"{loss_anomaly.item():.4f}",
                'Type': f"{loss_type.item():.4f}",
                'Location': f"{loss_location.item():.4f}"
            })

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{train_cfg['epochs']}] finished. Average Loss: {avg_loss:.4f}")

        if (epoch+1)%10 == 0:
            save_path = f'spatio_temporal_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    run_train()
