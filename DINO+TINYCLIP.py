import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPTextModelWithProjection

# 1. Load Data
image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
text = ["a photo of a cat", "a photo of a dog", "a photo of cats on a couch"]

# 2. Load Processors & Models
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dino_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

clip_proc = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M")
clip_text = CLIPTextModelWithProjection.from_pretrained("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M")

# 3. Extract Embeddings (No gradients needed for inference)
with torch.no_grad():
    # DINO Image Features (Grab the CLS token at index 0)
    img_inputs = dino_proc(images=image, return_tensors="pt")
    image_embeddings = dino_model(**img_inputs).last_hidden_state[:, 0, :]
    
    # TinyCLIP Text Features
    txt_inputs = clip_proc(text=text, return_tensors="pt", padding=True)
    text_embeddings = clip_text(**txt_inputs).text_embeds

# 4. Check Dimensions
print("Image embeddings:", image_embeddings.shape) # Expected: [1, 384]
print("Text embeddings: ", text_embeddings.shape)  # Expected: [3, 512]


