import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np

# Import testing functions and models
from testing.testing_functions import imagenet_zero_shot_accuracy
from adapter_training import ProjectionAdapter, DeepResidualAdapter, ResidualAdapter, load_config

def get_adapter(config_path="config.yaml", device="cpu"):
    config = load_config(config_path)
    arch_name = config.get("active_architecture", "mlp_3layer")
    arch_config = config["architectures"][arch_name]
    
    print(f"Loading adapter architecture: {arch_name} ({arch_config['type']})")
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
    
    # Try loading from the root directory based on arch_name
    model_path = f"dino_to_clip_adapter_{arch_name}.pt"
    
    print(f"Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Attempting to use generic adapter name 'dino_to_clip_adapter.pt'.")
        model.load_state_dict(torch.load("dino_to_clip_adapter.pt", map_location="cpu"))
        
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Adapter on ImageNet Zero-Shot using Cached Features")
    parser.add_argument("--batch-size", type=int, default=8192, help="Projection batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = get_adapter(device=device)
    
    print("\n--- Adapter Architecture in Use ---")
    print(adapter)
    print("-----------------------------------\n")
    
    print("\nLoading pre-extracted ImageNet-1K features from Hugging Face Hub...")
    repo_id = "StanislavLev/tiny-clip-image-encoders-adapter"
    
    # Load Image Embeddings (DINOv3)
    print("Loading image embeddings (DINOv3)...")
    image_ds = load_dataset(
        repo_id, 
        data_files="imagenet1k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/validation-00000.parquet", 
        split="train"
    )
    # Load Text Embeddings (TinyCLIP)
    print("Loading text embeddings (TinyCLIP)...")
    text_ds = load_dataset(
        repo_id, 
        data_files="imagenet1k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/validation-00000.parquet", 
        split="train"
    )
    # Load Metadata (Captions)
    print("Loading metadata...")
    meta_ds = load_dataset(
        repo_id, 
        data_files="imagenet1k/metadata/validation-00000.parquet", 
        split="train"
    )
    
    print("\nProcessing embeddings into memory...")
    # Extract columns efficiently
    image_embeddings_np = image_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
    text_embeddings_np = text_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
    captions = meta_ds["caption"]
    
    # 1. We need exactly 1000 unique class embeddings for Zero-Shot
    print("Extracting 1000 unique ImageNet classes and assigning labels...")
    unique_captions = []
    caption_to_idx = {}
    class_features_list = []
    
    labels = []
    
    for i, caption in enumerate(tqdm(captions, desc="Mapping classes")):
        if caption not in caption_to_idx:
            caption_to_idx[caption] = len(unique_captions)
            unique_captions.append(caption)
            class_features_list.append(text_embeddings_np[i])
        
        labels.append(caption_to_idx[caption])
        
    assert len(unique_captions) == 1000, f"Expected 1000 unique classes, found {len(unique_captions)}"
    
    # Convert to PyTorch Tensors
    class_features = torch.from_numpy(np.array(class_features_list)).to(device)
    raw_image_features = torch.from_numpy(image_embeddings_np).to(device)
    labels = torch.tensor(labels, device=device)
    
    print("\nProjecting DINOv3 features through the trained adapter...")
    mapped_image_features = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(raw_image_features), args.batch_size), desc="Projecting"):
            batch = raw_image_features[i:i+args.batch_size]
            projected = adapter(batch)
            mapped_image_features.append(projected)
            
    mapped_image_features = torch.cat(mapped_image_features, dim=0)
    
    print("\nCalculating Top-1 Accuracy...")
    accuracy = imagenet_zero_shot_accuracy(
        image_features=mapped_image_features,
        class_features=class_features,
        labels=labels,
        batch_size=args.batch_size,
        device=device
    )
    
    print(f"\n--- Final Result ---")
    print(f"Top-1 Zero-Shot Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
