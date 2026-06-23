import time
from torch.utils.data import DataLoader
from data_pipeline.data_loader import LAIONFeatureStore
from adapter_training import EmbeddingAlignmentDataset, short_cache

def test_dataloader():
    print("1. Initializing LAIONFeatureStore...")
    start_time = time.time()
    store = LAIONFeatureStore.from_hub(cache_dir=short_cache)
    print(f"Store initialized in {time.time() - start_time:.2f} seconds.")

    print("\n2. Creating Dataset and DataLoader...")
    dataset = EmbeddingAlignmentDataset(store)
    print(f"Total pairs in dataset: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True
    )
    print(f"Total batches per epoch: {len(dataloader)}")

    print("\n3. Testing Batch Fetching (fetching 3 batches)...")
    fetch_start = time.time()
    for i, (img_batch, txt_batch) in enumerate(dataloader):
        print(f"\nBatch {i+1} fetched in {time.time() - fetch_start:.2f}s:")
        print(f" - Image batch shape: {img_batch.shape} (dtype: {img_batch.dtype})")
        print(f" - Text batch shape: {txt_batch.shape} (dtype: {txt_batch.dtype})")
        
        # Stop after 3 batches to keep the test short
        if i == 2:
            break
            
        fetch_start = time.time()
            
    print("\nDataLoader test completed successfully!")

if __name__ == "__main__":
    test_dataloader()
