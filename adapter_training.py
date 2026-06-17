import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_pipeline.data_loader import TinyCLIPFeatureStore

short_cache = "C:/hf_cache"


# ==========================================
# 1. Dataset Mapper
# ==========================================
class EmbeddingAlignmentDataset(Dataset):
    """Pairs each image embedding with all its matching text embeddings."""
    def __init__(self, store: TinyCLIPFeatureStore):
        self.image_embeddings = store.image_embeddings
        self.text_embeddings = store.text_embeddings
        self.pairs = []

        for image_id, text_indices in store.image_id_to_text_indices.items():
            if image_id in store.image_id_to_image_idx:
                img_idx = store.image_id_to_image_idx[image_id]
                for txt_idx in text_indices:
                    self.pairs.append((img_idx, txt_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_idx, txt_idx = self.pairs[idx]
        return self.image_embeddings[img_idx], self.text_embeddings[txt_idx]


# ==========================================
# 2. MLP Adapter
# ==========================================
class ProjectionAdapter(nn.Module):
    """
    Robust 2-layer MLP with LayerNorm and Dropout projecting
    768-dim DINOv2 to 512-dim CLIP space with a learnable temperature scale.
    """
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        # Learnable logit scale parameter (initialized to CLIP's default of 2.6592)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, x):
        projected = self.net(x)
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
    return (loss_i + loss_t) / 2


# ==========================================
# 4. Training Loop
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load dataset
    print("Loading data from Hugging Face...")
    store = TinyCLIPFeatureStore.from_hub(
        split="train",
        image_config="image_embeddings__dinov2_base",
        text_config="text_embeddings__tinyclip_vit_39m_16_text_19m_yfcc15m",
        cache_dir=short_cache
    )
    
    dataloader = DataLoader(
        EmbeddingAlignmentDataset(store), 
        batch_size=256, 
        shuffle=True, 
        drop_last=True
    )

    # Initialize model, optimizer with weight decay, and scheduler
    model = ProjectionAdapter(input_dim=768, hidden_dim=1024, output_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Cosine annealing scheduler helps fine-tune learning rate toward convergence
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting high-performance training loop...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for img, txt in dataloader:
            img, txt = img.to(device), txt.to(device)
            
            optimizer.zero_grad()
            projected_img = model(img)
            
            # Compute loss using the dynamic model temperature scale
            loss = contrastive_loss(projected_img, txt, model.logit_scale)
            loss.backward()
            
            # Clip gradients to prevent exploding weights
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Calculate logit scale and dynamic temperature for tracking
        avg_loss = epoch_loss / len(dataloader)
        current_temp = 1.0 / model.logit_scale.exp().item()
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f} | Temp Scale: {current_temp:.4f}")

    # Save output
    save_path = "dino_to_clip_adapter.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved robust high-performance adapter to {save_path}")


if __name__ == "__main__":
    main()