import tensorflow_datasets as tfds
import tensorflow as tf
import os
import tqdm

def main():
    # The directory containing dataset_info.json
    data_dir = '/workspace/data/libero_object_no_noops/1.0.0'
    print(f"Loading dataset from: {data_dir}")

    try:
        builder = tfds.builder_from_directory(data_dir)
    except Exception as e:
        print(f"Error loading builder directly: {e}")
        return

    print("Dataset builder loaded successfully.")
    
    # We need to download and prepare the data if not already done, but here it seems to be already prepared.
    # builder.download_and_prepare() # Skip this as we assume data is there.
    
    total_episodes = 0
    total_frames = 0

    # Iterate over all splits
    # builder.info.splits gives info about splits.
    split_names = list(builder.info.splits.keys())
    print(f"Found splits: {split_names}")

    for split_name in split_names:
        print(f"Processing split: {split_name}")
        
        # Load the dataset split
        # as_dataset returns a tf.data.Dataset
        ds = builder.as_dataset(split=split_name)
        
        split_episodes = 0
        split_frames = 0
        
        # Iterate over episodes
        # We use tqdm for progress bar
        pbar = tqdm.tqdm(ds, desc=f"Counting {split_name}")
        for episode in pbar:
            split_episodes += 1
            
            # Count frames in the episode
            # episode['steps'] is a tf.data.Dataset
            # We can use reduce to count, or iterate.
            # Using reduce might be faster in graph mode, but iteration is simple.
            
            # steps = episode['steps']
            # count = steps.reduce(0, lambda x, _: x + 1).numpy()
            
            # Let's try simple iteration first
            count = 0
            for _ in episode['steps']:
                count += 1
            
            split_frames += count
            pbar.set_postfix({"frames": split_frames})
            
        print(f"Split {split_name}: {split_episodes} episodes, {split_frames} frames")
        total_episodes += split_episodes
        total_frames += split_frames

    print("-" * 30)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Frames: {total_frames}")

if __name__ == "__main__":
    main()
