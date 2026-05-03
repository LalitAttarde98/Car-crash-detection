import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText

from rfdetr import RFDETRLarge
from dataloader.accident_dataset import AccidentDataset
from configs.accident_prediction_config import data_cfg, model_cfg, predict_cfg

from pipeline.test import run_test
from pipeline.optimize_hyperparam import run_optimization
from pipeline.inference import run_inference
from pipeline.trainer import run_train

def initialize_smolvlm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    ).to(device)
    return model, processor

def get_common_resources(use_vlm=False, num_workers=4, data_cfg=data_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RFDETRLarge()

    valid_classes = data_cfg['infer_CLASSES']
    inv_class_map = {v: k for k, v in data_cfg.get('class_map', {}).items()}

    dataset = AccidentDataset(**data_cfg)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True, 
        prefetch_factor=2 if num_workers > 0 else None
    )

    vlm_model, vlm_processor = initialize_smolvlm() if use_vlm else (None, None)

    return model, dataloader, valid_classes, inv_class_map, vlm_model, vlm_processor

def main():
    parser = argparse.ArgumentParser(description="Accident Detection and Prediction Pipeline")
    parser.add_argument("mode", 
        choices=["train", "test", "optimize", "inference"], 
        help="Choose which pipeline to execute."
    )
    parser.add_argument( "--use-vlm", 
        action="store_true", 
        help="Enable VLM for crash type classification"
    )
    args = parser.parse_args()

    num_workers = 0 if args.mode == "optimize" else 4
    use_vlm = args.use_vlm and args.mode in ["test", "inference"]
    data_cfg['mode'] = args.mode

    if args.mode == "train":
        run_train()
        return

    model, dataloader, valid_classes, inv_class_map, vlm_model, vlm_processor = get_common_resources(
        use_vlm=use_vlm, num_workers=num_workers, data_cfg=data_cfg
    )

    if args.mode == "test":
        run_test(
            model, dataloader, valid_classes, inv_class_map, 
            predict_cfg, model_cfg, vlm_model, vlm_processor
        )

    elif args.mode == "optimize":
        run_optimization(
            model, dataloader, valid_classes, data_cfg, model_cfg, predict_cfg
        )

    elif args.mode == "inference":
        run_inference(
            model, dataloader, valid_classes, inv_class_map, 
            predict_cfg, model_cfg, vlm_model, vlm_processor
        )

if __name__ == "__main__":
    main()