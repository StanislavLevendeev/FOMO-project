import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import yaml
import argparse
import re
from pathlib import Path
from data_pipeline.data_loader import LAIONFeatureStore


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DINO-to-CLIP adapter.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file. Defaults to config.yaml.",
    )
    return parser.parse_args()


def safe_path_part(value):
    value = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_-]+", "-", value).strip("-")


def infer_image_embedding_folder(config):
    dataset_config = config.get("dataset", {})
    explicit = dataset_config.get("image_embedding_folder") or dataset_config.get("image_embedding_set")
    if explicit:
        return explicit

    input_dim = int(config.get("model", {}).get("input_dim", 384))
    if input_dim == 384:
        return "dinov3_vits16_pretrain_lvd1689m"
    if input_dim == 768:
        return "dinov3_vitb16_pretrain_lvd1689m"
    if input_dim == 1024:
        return "dinov3_vitl16_pretrain_lvd1689m"
    raise ValueError(
        f"Cannot infer LAION image embedding folder for input_dim={input_dim}. "
        "Set dataset.image_embedding_folder explicitly."
    )


def build_run_dir(config, arch_name):
    training_config = config["training"]
    output_root = Path(training_config.get("models_dir", "models"))
    early_stopping_label = (
        f"es-p{training_config.get('early_stopping_patience', 8)}"
        f"-d{training_config.get('early_stopping_delta', 0.001)}"
        if training_config.get("use_early_stopping", True)
        else "es-off"
    )
    run_name_parts = [
        f"arch-{arch_name}",
        f"bs{training_config['batch_size']}",
        f"lr{training_config['lr']}",
        f"wd{training_config['weight_decay']}",
        f"val{training_config.get('validation_fraction', 0.02)}",
        early_stopping_label,
    ]
    run_name = "_".join(safe_path_part(part) for part in run_name_parts)
    return output_root / run_name


def save_yaml(path, content):
    with open(path, "w") as f:
        yaml.safe_dump(content, f, sort_keys=False)


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Configured device {device_name!r}, but CUDA is not available.")
    return device


# ==========================================
# 1. Dataset Mapper
# ==========================================
class EmbeddingAlignmentDataset(Dataset):
    """Pairs each image embedding with its matching text embedding sequentially (1-to-1)."""
    def __init__(self, store: LAIONFeatureStore):
        self.store = store

    def __len__(self):
        return len(self.store.image_embeddings)

    def __getitem__(self, idx):
        return self.store.image_embeddings[idx], self.store.text_embeddings[idx]


# ==========================================
# 2. MLP Adapter
# ==========================================
class ProjectionAdapter(nn.Module):
    """
    Dynamic MLP Adapter whose structure is defined by configuration parameters.
    """
    def __init__(self, input_dim, output_dim, hidden_dims, activation="GELU", use_layer_norm=True, dropout=0.1):
        super().__init__()
        
        # Select activation function
        activation_cls = getattr(nn, activation, nn.GELU)
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Final projection layer to target space
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        # Learnable logit scale parameter (initialized to CLIP's default of 2.6592)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, x):
        projected = self.net(x)
        # Apply L2 normalization for cosine similarity compatibility
        return F.normalize(projected, p=2, dim=-1)


class ResidualAdapter(nn.Module):
    """
    Residual Bottleneck Adapter:
    Left path: Linear (input_dim -> output_dim)
    Right path: Linear (input_dim -> bottleneck_dim) -> Activation -> Linear (bottleneck_dim -> output_dim)
    Output is the sum of both paths.
    """
    def __init__(self, input_dim, output_dim, bottleneck_dim, activation="ReLU"):
        super().__init__()
        # Left path (skip projection to match dimensions)
        self.skip_proj = nn.Linear(input_dim, output_dim)
        
        # Right path (bottleneck path)
        activation_cls = getattr(nn, activation, nn.ReLU)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            activation_cls(),
            nn.Linear(bottleneck_dim, output_dim)
        )
        # Learnable logit scale parameter (initialized to CLIP's default of 2.6592)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, x):
        out = self.skip_proj(x) + self.bottleneck(x)
        # Apply L2 normalization for cosine similarity compatibility
        return F.normalize(out, p=2, dim=-1)


class DeepResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return x + out


class DeepResidualAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.Sequential(*[
            DeepResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        # Learnable logit scale parameter (initialized to CLIP's default of 2.6592)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        projected = self.output_proj(x)
        # Apply L2 normalization for cosine similarity compatibility
        return F.normalize(projected, p=2, dim=-1)


# ==========================================
# 3. Dynamic Contrastive Loss (InfoNCE)
# ==========================================
def contrastive_loss(img_features, txt_features, logit_scale):
    """Calculates bidirectional infoNCE loss scaling by learnable temperature."""
    txt_features = F.normalize(txt_features, p=2, dim=-1)
    
    # Scale cosine similarity matrix using the learnable logit scale exponent
    t_scale = logit_scale.exp()
    logits = t_scale * (img_features @ txt_features.T)
    labels = torch.arange(img_features.shape[0], device=img_features.device)
    
    # Calculate bidirectional cross-entropy
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_t) / 2
    
    # Calculate additional metrics for batch-independent comparison
    with torch.no_grad():
        # Top-1 accuracy (bidirectional)
        acc_i = (logits.argmax(dim=-1) == labels).float().mean()
        acc_t = (logits.T.argmax(dim=-1) == labels).float().mean()
        accuracy = (acc_i + acc_t) / 2
        
        # Normalized loss = loss / log(B). Bounded between 0 (perfect) and 1 (random guessing).
        import math
        norm_loss = loss / math.log(img_features.shape[0])
        
    return loss, norm_loss.item(), accuracy.item()


def evaluate(model, dataloader, device):
    """Evaluate adapter quality on held-out embedding pairs."""
    model.eval()
    total_loss = 0.0
    total_norm_loss = 0.0
    total_accuracy = 0.0

    with torch.inference_mode():
        for img, txt in dataloader:
            img, txt = img.to(device), txt.to(device)
            projected_img = model(img)
            loss, norm_loss, accuracy = contrastive_loss(
                projected_img, txt, model.logit_scale
            )
            total_loss += loss.item()
            total_norm_loss += norm_loss
            total_accuracy += accuracy

    num_batches = len(dataloader)
    model.train()
    return (
        total_loss / num_batches,
        total_norm_loss / num_batches,
        total_accuracy / num_batches,
    )


# ==========================================
# 4. Training Loop
# ==========================================
def main():
    args = parse_args()
    config = load_config(args.config)
    short_cache = config["dataset"]["cache_dir"]

    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Training on: {device}")
    print(f"Using config: {args.config}")

    # Load dataset
    print("Loading data from Hugging Face...")
    image_embedding_folder = infer_image_embedding_folder(config)
    text_embedding_folder = config.get("dataset", {}).get(
        "text_embedding_folder",
        "tinyclip_vit_39m_16_text_19m_yfcc15m",
    )
    print(f"Using image embedding folder: {image_embedding_folder}")
    print(f"Using text embedding folder: {text_embedding_folder}")
    store = LAIONFeatureStore.from_hub(
        repo_id=config["dataset"]["repo_id"],
        cache_dir=short_cache,
        image_embedding_folder=image_embedding_folder,
        text_embedding_folder=text_embedding_folder,
    )

    dataset = EmbeddingAlignmentDataset(store)
    val_fraction = float(config["training"].get("validation_fraction", 0.02))
    split_seed = int(config["training"].get("split_seed", 42))

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("training.validation_fraction must be >= 0.0 and < 1.0")

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    if val_size > 0:
        generator = torch.Generator().manual_seed(split_seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        print(
            f"Using {train_size:,} training pairs and {val_size:,} validation pairs "
            f"(validation_fraction={val_fraction:.3f})."
        )
    else:
        train_dataset = dataset
        val_dataset = None
        print("No validation split configured; training on all available pairs.")

    dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        drop_last=True,
        pin_memory=True
    )
    if len(dataloader) == 0:
        raise ValueError(
            "Training split is smaller than one full batch. Reduce training.batch_size "
            "or lower training.validation_fraction."
        )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            drop_last=True,
            pin_memory=True
        )
        if len(val_dataloader) == 0:
            raise ValueError(
                "Validation split is smaller than one full batch. Increase "
                "training.validation_fraction or reduce training.batch_size."
            )

    # Initialize model, optimizer with weight decay, and scheduler based on configuration
    arch_name = config.get("active_architecture", "mlp_2layer")
    arch_config = config["architectures"][arch_name]
    print(f"Building adapter with architecture: {arch_name} ({arch_config['type']})")
    
    if arch_config["type"] == "mlp":
        model = ProjectionAdapter(
            input_dim=config["model"]["input_dim"],
            output_dim=config["model"]["output_dim"],
            hidden_dims=arch_config["hidden_dims"],
            activation=arch_config.get("activation", "GELU"),
            use_layer_norm=arch_config.get("use_layer_norm", False),
            dropout=arch_config.get("dropout", 0.0)
        ).to(device)
    elif arch_config["type"] == "residual":
        model = ResidualAdapter(
            input_dim=config["model"]["input_dim"],
            output_dim=config["model"]["output_dim"],
            bottleneck_dim=arch_config["bottleneck_dim"],
            activation=arch_config.get("activation", "ReLU")
        ).to(device)
    elif arch_config["type"] == "deep_residual":
        model = DeepResidualAdapter(
            input_dim=config["model"]["input_dim"],
            hidden_dim=arch_config["hidden_dim"],
            output_dim=config["model"]["output_dim"],
            num_blocks=arch_config.get("num_blocks", 6),
            dropout=arch_config.get("dropout", 0.1)
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture type: {arch_config['type']}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["training"]["lr"]), 
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    # Learning rate scheduler with warmup (1 epoch linear warmup, then cosine decay)
    epochs = config["training"]["epochs"]
    warmup_epochs = 1
    
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )

    use_early_stopping = bool(config["training"].get("use_early_stopping", True))
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 8))
    early_stopping_delta = float(config["training"].get("early_stopping_delta", 0.001))

    if use_early_stopping and val_dataloader is None:
        raise ValueError(
            "use_early_stopping requires training.validation_fraction > 0.0"
        )

    run_dir = build_run_dir(config, arch_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    config_snapshot_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.yaml"
    save_yaml(config_snapshot_path, config)
    print(f"Saving run artifacts to: {run_dir}")

    print("\nStarting high-performance training loop...")
    model.train()
    best_val_norm_loss = float("inf")
    best_val_accuracy = None
    best_epoch = 0
    last_significant_improvement_epoch = 0
    last_metrics = {}
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_norm_loss = 0.0
        epoch_accuracy = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for img, txt in pbar:
            img, txt = img.to(device), txt.to(device)
            
            optimizer.zero_grad()
            projected_img = model(img)
            
            # Compute loss using the dynamic model temperature scale
            loss, norm_loss, accuracy = contrastive_loss(projected_img, txt, model.logit_scale)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()
            epoch_norm_loss += norm_loss
            epoch_accuracy += accuracy
            
            # Update progress bar with current batch metrics
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "norm_loss": f"{norm_loss:.3f}",
                "acc": f"{accuracy * 100:.1f}%"
            })
            
        scheduler.step()
        
        # Calculate logit scale and dynamic temperature for tracking
        avg_loss = epoch_loss / len(dataloader)
        avg_norm_loss = epoch_norm_loss / len(dataloader)
        avg_acc = epoch_accuracy / len(dataloader)
        current_temp = 1.0 / model.logit_scale.exp().item()
        last_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_normalized_loss": avg_norm_loss,
            "train_accuracy": avg_acc,
            "temperature": current_temp,
            "early_stopped": False,
        }
        log_message = (
            f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.3f} | "
            f"Normalized Loss: {avg_norm_loss:.3f} | Accuracy: {avg_acc * 100:.2f}% | "
            f"Temp Scale: {current_temp:.4f}"
        )

        if val_dataloader is not None:
            val_loss, val_norm_loss, val_acc = evaluate(model, val_dataloader, device)
            last_metrics.update(
                {
                    "validation_loss": val_loss,
                    "validation_normalized_loss": val_norm_loss,
                    "validation_accuracy": val_acc,
                }
            )
            log_message += (
                f" | Val Loss: {val_loss:.3f} | Val Normalized Loss: {val_norm_loss:.3f} | "
                f"Val Accuracy: {val_acc * 100:.2f}%"
            )
            improved = val_norm_loss < best_val_norm_loss - early_stopping_delta

            if improved:
                best_val_norm_loss = val_norm_loss
                best_val_accuracy = val_acc
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), best_path)
                log_message += " | Saved best checkpoint"
            else:
                bad_epochs += 1

            if use_early_stopping and bad_epochs >= early_stopping_patience:
                last_metrics["early_stopped"] = True
                print(log_message)
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch was {best_epoch + 1} with "
                    f"Val Normalized Loss: {best_val_norm_loss:.3f}."
                )
                break

        print(log_message)

    torch.save(model.state_dict(), last_path)
    metrics = {
        "run_dir": str(run_dir),
        "config_path": args.config,
        "architecture": arch_name,
        "last": last_metrics,
        "best": {
            "epoch": best_epoch + 1 if best_val_accuracy is not None else None,
            "validation_normalized_loss": (
                best_val_norm_loss if best_val_accuracy is not None else None
            ),
            "validation_accuracy": best_val_accuracy,
            "checkpoint": str(best_path) if best_val_accuracy is not None else None,
        },
        "checkpoints": {
            "last": str(last_path),
            "best": str(best_path) if best_val_accuracy is not None else None,
        },
    }
    save_yaml(metrics_path, metrics)

    print(f"\nSaved last adapter checkpoint to {last_path}")
    if best_val_accuracy is not None:
        print(f"Saved best adapter checkpoint to {best_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
