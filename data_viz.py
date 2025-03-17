import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

# Ensure no OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# ----------------------------
# Dataset Class
# ----------------------------
class LaneDataset(Dataset):
    """
    Expects X and Y to be Numpy arrays:
      X.shape = (num_samples, 80, 160, 3)
      Y.shape = (num_samples, 80, 160, 1)
    Converts them to PyTorch format: (3, 80, 160) and (1, 80, 160).
    """
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Transpose images: (H, W, C) -> (C, H, W)
        x = self.images[idx].transpose(2, 0, 1)  # channel-first
        y = self.labels[idx].transpose(2, 0, 1)  # single channel, channel-first
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ----------------------------
# Showcase Dataset
# ----------------------------
def showcase_dataset(dataset, num_samples=5):
    """
    Displays a few samples from the dataset along with their labels.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3))
    fig.suptitle("Dataset Visualization: Input Images and Corresponding Labels", 
                 fontsize=18, fontweight='bold', color='navy')

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]

        # Convert images back to (H, W, C) for visualization
        img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        label = label.squeeze(0).numpy()   # (1, H, W) -> (H, W)

        # Normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min())
        label = (label - label.min()) / (label.max() - label.min())

        # Input Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {idx}: Input Image", fontsize=12, color='darkblue')
        axes[i, 0].axis('off')

        # Label
        axes[i, 1].imshow(label, cmap='gray')
        axes[i, 1].set_title(f"Sample {idx}: Label", fontsize=12, color='darkblue')
        axes[i, 1].axis('off')

    # Add a shared legend
    fig.text(0.5, 0.01, "Left: Input Images | Right: Corresponding Labels", 
             ha='center', fontsize=12, color='dimgray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()


# ----------------------------
# Print Dataset Info
# ----------------------------
def print_dataset_info(images, labels):
    """
    Prints detailed information about the dataset in the terminal.
    """
    print("\n=== Dataset Information ===")
    print(f"Number of samples: {len(images)}")
    print(f"Image dimensions: {images.shape[1:]} (Height, Width, Channels)")
    print(f"Label dimensions: {labels.shape[1:]} (Height, Width, Channels)")
    print(f"Image data type: {images.dtype}")
    print(f"Label data type: {labels.dtype}")
    print(f"Image pixel value range: {images.min()} to {images.max()}")
    print(f"Label pixel value range: {labels.min()} to {labels.max()}")
    print("===========================\n")


# ----------------------------
# Main Function
# ----------------------------
def main():
    # Load the dataset
    train_images = pickle.load(open("full_CNN_train.p", "rb"))
    labels = pickle.load(open("full_CNN_labels.p", "rb"))

    # Convert to numpy arrays
    train_images = np.array(train_images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Normalize labels (0..255 -> 0..1)
    labels /= 255.0

    # Print dataset info
    print_dataset_info(train_images, labels)

    # Create Dataset
    dataset = LaneDataset(train_images, labels)

    # Showcase the dataset
    showcase_dataset(dataset, num_samples=5)


if __name__ == "__main__":
    main()
