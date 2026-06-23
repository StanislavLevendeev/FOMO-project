import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
from data_pipeline.data_loader import LAIONFeatureStore

# Load configuration from YAML file
config_path = "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

short_cache = config["dataset"]["cache_dir"]


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


# ==========================================
# 4. Training Loop
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load dataset
    print("Loading data from Hugging Face...")
    store = LAIONFeatureStore.from_hub(
        repo_id=config["dataset"]["repo_id"],
        cache_dir=short_cache
    )
    
    dataloader = DataLoader(
        EmbeddingAlignmentDataset(store), 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        drop_last=True,
        pin_memory=True
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

    print("\nStarting high-performance training loop...")
    model.train()
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
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.3f} | Normalized Loss: {avg_norm_loss:.3f} | Accuracy: {avg_acc * 100:.2f}% | Temp Scale: {current_temp:.4f}")

    # Save output with name based on active architecture
    save_prefix = config["training"].get("save_prefix", "dino_to_clip_adapter")
    save_path = f"{save_prefix}_{arch_name}.pt"
        
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved robust high-performance adapter to {save_path}")


if __name__ == "__main__":
    main()