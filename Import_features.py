from datasets import load_dataset

# Define a short, simple path at the root of your drive
short_cache = "C:/hf_cache"

# Load the data using the new short cache directory
train_text_data = load_dataset(
    "StanislavLev/tiny-clip-image-encoders-adapter", 
    "flickr30k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m",
    split="train",
    cache_dir=short_cache  # <-- This fixes the crash
)

# Do the same for the images
train_image_data = load_dataset(
    "StanislavLev/tiny-clip-image-encoders-adapter", 
    "flickr30k/image_embeddings/dinov2_base",
    split="train",
    cache_dir=short_cache
)

