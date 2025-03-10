import os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data.generator import ShapeGenerator

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a dataset of geometric shapes and characters")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="test_dataset", help="Output directory for the dataset")
    parser.add_argument("--image_size", type=int, default=32, help="Size of the generated images")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize shape generator
    generator = ShapeGenerator(
        output_dir=output_dir,
        image_size=args.image_size,
        num_samples=args.num_samples
    )
    
    # Generate dataset with fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    generator.generate_dataset()
    
    # Visualize a random subset of the generated images
    import json
    
    # Load metadata
    with open(os.path.join(output_dir, "train_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Select 25 random samples
    samples = random.sample(metadata, min(25, len(metadata)))
    
    # Create a grid of images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    
    for i, sample in enumerate(samples):
        # Load image
        image_path = os.path.join(output_dir, sample["image_path"])
        image = plt.imread(image_path)
        
        # Display image
        row, col = i // 5, i % 5
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(sample["prompt"], fontsize=8)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(output_dir, "sample_grid.png"), dpi=150)
    plt.close(fig)
    
    print(f"Dataset generated in {output_dir}")
    print(f"Sample grid saved to {os.path.join(output_dir, 'sample_grid.png')}")

if __name__ == "__main__":
    main()
