import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPTextModelWithProjection

# ==========================================
# 1. Import Adapter Architecture
# ==========================================
# Instead of redefining the class here, we import it directly from our training script.
# Any changes you make to the architecture in train_adapter.py will automatically apply here!
from adapter_training import ProjectionAdapter


# ==========================================
# 2. Load Processors, Models, & Adapter
# ==========================================
# Note: We use DINOv2 Base to match the 768-dim weights trained in the adapter
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base")

clip_proc = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M")
clip_text = CLIPTextModelWithProjection.from_pretrained("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M")

# Initialize and load your trained adapter weights
adapter = ProjectionAdapter(input_dim=768, hidden_dim=1024, output_dim=512)
adapter.load_state_dict(torch.load("dino_to_clip_adapter.pt", map_location="cpu"))
adapter.eval() # Set adapter to evaluation mode


# ==========================================
# 3. Process Live Data & Extract Features
# ==========================================
image_url = "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1200&q=80"
image = Image.open(requests.get(image_url, stream=True).raw)

# Adjust your labels to stress-test the model's fine-grained classification:
text = ["a road racing bike", "a time-trial triathlon bike", "a mountain bike", "a group of cyclists"]

with torch.no_grad():
    # A. Get DINOv2 features [1, 768]
    img_inputs = dino_proc(images=image, return_tensors="pt")
    dino_embeddings = dino_model(**img_inputs).last_hidden_state[:, 0, :]
    
    # B. Get TinyCLIP text features [3, 512]
    txt_inputs = clip_proc(text=text, return_tensors="pt", padding=True)
    text_embeddings = clip_text(**txt_inputs).text_embeds
    
    # C. Project DINOv2 features to CLIP space [1, 512]
    mapped_image_embeddings = adapter(dino_embeddings)


# ==========================================
# 4. Compute Cosine Similarity & Probabilities
# ==========================================
# L2-normalize text embeddings (mapped image features are already normalized by the adapter)
text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

# Calculate cosine similarities and scale by the trained temperature
# Note: logit_scale might have changed during training; we use model.logit_scale.exp()
temp_scale = adapter.logit_scale.exp()
logits = temp_scale * (mapped_image_embeddings @ text_embeddings.T)

# Apply Softmax to get probabilities
probs = logits.softmax(dim=-1).squeeze(0)


# ==========================================
# 5. Output Results
# ==========================================
print("\n--- Zero-Shot Class Probabilities ---")
for label, prob in zip(text, probs):
    print(f"{label}: {prob.item() * 100:.2f}%")