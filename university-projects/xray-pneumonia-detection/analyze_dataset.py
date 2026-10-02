"""
Analysis and visualization of chest X-ray dataset (Normal / Pneumonia)
up to the stage of data preparation and model training.

All plots are saved in the folder figures/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import random
import config  # import paths to TRAIN_DIR and TEST_DIR

# Create folder for saving plots if it does not exist
FIGURES_DIR = "analysis_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def analyze_data(base_dir, dataset_name="dataset"):
    """
    Function for analyzing and visualizing data from the specified directory (train/test).
    All results are saved in the figures/ folder.
    """
    classes = os.listdir(base_dir)
    class_counts = {}

    # Count the number of images in each class
    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        class_counts[cls] = len(os.listdir(cls_path))

    # Visualize class distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
    plt.title(f"Class Distribution — {dataset_name}")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{dataset_name}_class_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    # Sample images
    plt.figure(figsize=(8, 4))
    for i, cls in enumerate(classes):
        cls_path = os.path.join(base_dir, cls)
        sample_img = random.choice(os.listdir(cls_path))
        img = Image.open(os.path.join(cls_path, sample_img))
        plt.subplot(1, len(classes), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(cls)
        plt.axis("off")
    plt.suptitle(f"Sample Images — {dataset_name}")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{dataset_name}_sample_images.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    # Analyze image sizes
    sizes = []
    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        for img_name in os.listdir(cls_path)[:200]:
            img = Image.open(os.path.join(cls_path, img_name))
            sizes.append(img.size)

    widths, heights = zip(*sizes)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=widths, y=heights)
    plt.title(f"Image Size Distribution — {dataset_name}")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{dataset_name}_image_sizes.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    # Pixel intensity histogram
    plt.figure(figsize=(6, 4))
    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        sample_img = random.choice(os.listdir(cls_path))
        img = np.array(Image.open(os.path.join(cls_path, sample_img)).convert("L")).ravel()
        sns.histplot(img, bins=50, label=cls, alpha=0.6)
    plt.title(f"Pixel Intensity Distribution — {dataset_name}")
    plt.xlabel("Intensity (0–255)")
    plt.ylabel("Number of Pixels")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{dataset_name}_pixel_intensity.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    # Class statistics
    total = sum(class_counts.values())
    print(f"\nStatistics ({dataset_name}):")
    for cls, count in class_counts.items():
        percent = (count / total) * 100
        print(f"   {cls}: {count} images ({percent:.2f}%)")

    imbalance = max(class_counts.values()) / min(class_counts.values())
    if imbalance > 1.5:
        print("Warning: data is imbalanced between classes!")
    else:
        print("Classes are balanced.")


if __name__ == "__main__":
    print("Analyzing training dataset...")
    analyze_data(config.TRAIN_DIR, dataset_name="train")

    print("\nAnalyzing test dataset...")
    analyze_data(config.TEST_DIR, dataset_name="test")

    print("\nAnalysis completed. All plots saved in the 'figures/' folder.")
