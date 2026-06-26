import torch
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np

# Import testing functions and models
from testing.testing_functions import topk_accuracy_from_features, flickr30k_retrieval_at_1
from adapter_training import ProjectionAdapter, DeepResidualAdapter, ResidualAdapter, load_config
import os
from pathlib import Path

def get_adapter(config_path="config.yaml", device="cpu"):
    config = load_config(config_path)
    arch_name = config.get("active_architecture", "mlp_3layer")
    arch_config = config["architectures"][arch_name]
    
    if arch_config["type"] == "mlp":
        model = ProjectionAdapter(
            input_dim=config["model"]["input_dim"],
            output_dim=config["model"]["output_dim"],
            hidden_dims=arch_config["hidden_dims"],
            activation=arch_config.get("activation", "GELU"),
            use_layer_norm=arch_config.get("use_layer_norm", False),
            dropout=arch_config.get("dropout", 0.0)
        )
    elif arch_config["type"] == "residual":
        model = ResidualAdapter(
            input_dim=config["model"]["input_dim"],
            output_dim=config["model"]["output_dim"],
            bottleneck_dim=arch_config["bottleneck_dim"],
            activation=arch_config.get("activation", "ReLU")
        )
    elif arch_config["type"] == "deep_residual":
        model = DeepResidualAdapter(
            input_dim=config["model"]["input_dim"],
            hidden_dim=arch_config["hidden_dim"],
            output_dim=config["model"]["output_dim"],
            num_blocks=arch_config.get("num_blocks", 6),
            dropout=arch_config.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_config['type']}")
    
    # Try finding the best model from the models/ directory
    models_dir = Path(config.get("training", {}).get("models_dir", "models"))
    found_model_path = None
    
    # Check if there is a matching run folder in models/
    if models_dir.exists():
        # Find all folders that match the architecture
        matching_dirs = []
        for d in models_dir.iterdir():
            if d.is_dir() and arch_name.replace("_", "") in d.name.replace("_", ""):
                # Prioritize directories that have best.pt
                if (d / "best.pt").exists() or (d / "last.pt").exists():
                    matching_dirs.append(d)
        
        # Sort by modification time (newest first)
        if matching_dirs:
            matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            latest_run = matching_dirs[0]
            if (latest_run / "best.pt").exists():
                found_model_path = latest_run / "best.pt"
            else:
                found_model_path = latest_run / "last.pt"
                
    # Fallback to root directory if not found in models/
    if not found_model_path:
        root_path = Path(f"dino_to_clip_adapter_{arch_name}.pt")
        if root_path.exists():
            found_model_path = root_path
            
    if not found_model_path:
        print(f"\n[ERROR] Weights file not found for architecture '{arch_name}'.")
        print(f"I looked in '{models_dir}/' and the root directory.")
        print(f"You must first train the '{arch_name}' adapter before evaluating it.")
        exit(1)
        
    print(f"Loading weights from {found_model_path}...")
    try:
        model.load_state_dict(torch.load(found_model_path, map_location="cpu", weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(found_model_path, map_location="cpu"))
        
    model.to(device)
    model.eval()
    return model, arch_name

def evaluate_imagenet(adapter, device, batch_size):
    print("\n--- Evaluating on ImageNet-1K ---")
    repo_id = "StanislavLev/tiny-clip-image-encoders-adapter"
    
    image_ds = load_dataset(repo_id, data_files="imagenet1k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/validation-00000.parquet", split="train")
    text_ds = load_dataset(repo_id, data_files="imagenet1k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/validation-00000.parquet", split="train")
    meta_ds = load_dataset(repo_id, data_files="imagenet1k/metadata/validation-00000.parquet", split="train")
    
    image_embeddings_np = image_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
    text_embeddings_np = text_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
    captions = meta_ds["caption"]
    
    unique_captions = []
    caption_to_idx = {}
    class_features_list = []
    labels = []
    
    for i, caption in enumerate(captions):
        if caption not in caption_to_idx:
            caption_to_idx[caption] = len(unique_captions)
            unique_captions.append(caption)
            class_features_list.append(text_embeddings_np[i])
        labels.append(caption_to_idx[caption])
        
    class_features = torch.from_numpy(np.array(class_features_list)).to(device)
    raw_image_features = torch.from_numpy(image_embeddings_np).to(device)
    labels = torch.tensor(labels, device=device)
    
    mapped_image_features = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(raw_image_features), batch_size), desc="Projecting ImageNet"):
            batch = raw_image_features[i:i+batch_size]
            projected = adapter(batch)
            mapped_image_features.append(projected)
            
    mapped_image_features = torch.cat(mapped_image_features, dim=0)
    
    acc = topk_accuracy_from_features(
        image_features=mapped_image_features,
        class_features=class_features,
        labels=labels,
        topk=(1, 5),
        batch_size=batch_size,
        device=device
    )
    return acc["top1"], acc["top5"]

def evaluate_flickr30k(adapter, device, batch_size):
    print("\n--- Evaluating on Flickr30k ---")
    repo_id = "StanislavLev/tiny-clip-image-encoders-adapter"
    
    image_ds = load_dataset(repo_id, data_files="flickr30k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/test.parquet", split="train")
    text_ds = load_dataset(repo_id, data_files="flickr30k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/test.parquet", split="train")
    
    image_embeddings_np = image_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
    text_embeddings_np = text_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
    
    image_ids = image_ds["image_id"]
    text_image_ids = text_ds["image_id"]
    
    raw_image_features = torch.from_numpy(image_embeddings_np).to(device)
    text_features = torch.from_numpy(text_embeddings_np).to(device)
    
    mapped_image_features = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(raw_image_features), batch_size), desc="Projecting Flickr30k"):
            batch = raw_image_features[i:i+batch_size]
            projected = adapter(batch)
            mapped_image_features.append(projected)
            
    mapped_image_features = torch.cat(mapped_image_features, dim=0)
    
    retrieval_metrics = flickr30k_retrieval_at_1(
        image_features=mapped_image_features,
        text_features=text_features,
        image_ids=image_ids,
        text_image_ids=text_image_ids,
        batch_size=batch_size,
        device=device
    )
    
    return retrieval_metrics["flickr30k_i2t_at_1"], retrieval_metrics["flickr30k_t2i_at_1"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adapter on ImageNet and Flickr30k")
    parser.add_argument("--batch-size", type=int, default=8192, help="Projection batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter, arch_name = get_adapter(device=device)
    
    print(f"\nEvaluating Adapter: {arch_name}")
    
    in_top1, in_top5 = evaluate_imagenet(adapter, device, args.batch_size)
    f_i2t, f_t2i = evaluate_flickr30k(adapter, device, args.batch_size)
    
    # Format and print table
    print("\n" + "="*80)
    print(f"{'Adapter Name':<30} | {'Flickr30k I2T@1':<15} | {'Flickr30k T2I@1':<15} | {'ImageNet Top-1':<15} | {'ImageNet Top-5':<15}")
    print("-" * 80)
    print(f"{arch_name:<30} | {f_i2t*100:>14.2f}% | {f_t2i*100:>14.2f}% | {in_top1*100:>14.2f}% | {in_top5*100:>14.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
