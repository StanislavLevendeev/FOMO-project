from data_pipeline.data_loader import TinyCLIPFeatureStore


store = TinyCLIPFeatureStore.from_hub(
    split="train",
    image_config="image_embeddings__dinov2_base",
    text_config="text_embeddings__tinyclip_vit_39m_16_text_19m_yfcc15m",
)

print(store.image_embeddings.shape)
print(store.text_embeddings.shape)

image_emb = store.get_image_embedding("1000")
text_embs = store.get_text_embeddings_for_image("1000")
captions = store.get_captions("1000")

print(image_emb.shape)
print(text_embs.shape)
print(captions)
 