import os
import argparse
import shutil
import math
import random
from pathlib import Path

def split_dataset(source_dir, split_percent):
    # Convert input to an absolute path to handle relative inputs correctly
    src = Path(source_dir).resolve()

    # Extract the name of the source directory (the "lastdir")
    folder_prefix = src.name

    # Define new roots using the parent directory and the new naming convention
    # This puts them side-by-side with your original folder
    train_root = src.parent / f"{folder_prefix}_training_set"
    test_root = src.parent / f"{folder_prefix}_test_set"

    # 1. Identify classes
    all_files = [f for f in src.iterdir() if f.is_file()]
    classes = set(f.name.split('_')[0] for f in all_files)

    print(f"Detected classes: {', '.join(classes)}")
    print(f"Creating sets in: {src.parent}")

    # Create directories inside the new prefixed roots
    (test_root).mkdir(parents=True, exist_ok=True)
    (train_root).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        # Filter and shuffle
        cls_files = [f for f in all_files if f.name.startswith(f"{cls}_")]
        random.shuffle(cls_files)

        # 2. Calculate split index
        split_idx = math.floor(len(cls_files) * (split_percent / 100))
        train_files = cls_files[:split_idx]
        test_files = cls_files[split_idx:]

        # 3. Copy files (using absolute paths to avoid confusion)
        for f in train_files:
            shutil.copy2(f, train_root / f.name)
        for f in test_files:
            shutil.copy2(f, test_root / f.name)

        print(f"Class {cls}: {len(train_files)} train, {len(test_files)} test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset with dynamic folder naming.")
    parser.add_argument("source", type=str, help="Source directory")
    parser.add_argument("percent", type=float, help="Percentage for training (0-100)")

    args = parser.parse_args()

    if 0 <= args.percent <= 100:
        split_dataset(args.source, args.percent)
        print(f"\nDone! Sets created with prefix based on '{Path(args.source).resolve().name}'.")
    else:
        print("Error: Percentage must be between 0 and 100.")
