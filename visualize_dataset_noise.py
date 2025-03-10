import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.generator import ShapeGenerator
from model.diffusion import DiffusionModel

def main():
    # Create output directory
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create temporary directory for shape generator
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize shape generator
    generator = ShapeGenerator(
        output_dir=temp_dir,
        image_size=32,
        num_samples=10  # We only need 10 samples
    )
    
    # Initialize diffusion model (we only need it for the add_noise method)
    model = DiffusionModel().to(device)
    
    # Generate 10 samples
    samples = []
    prompts = []
    
    # Generate 5 shapes and 5 characters
    for i in range(5):
        # Generate a shape
        shape_type = generator.shape_types[i % len(generator.shape_types)]
        image, params = generator.generate_shape(shape_type)
        
        # Create prompt
        size_desc = ["small", "medium", "large"][i % 3]
        position_desc = ["centered", "top left", "bottom right", "top right", "bottom left"][i % 5]
        prompt = f"A {size_desc} {shape_type} {position_desc}"
        
        samples.append(image)
        prompts.append(prompt)
    
    for i in range(5):
        # Generate a character
        char = generator.characters[i % len(generator.characters)]
        image, params = generator.generate_character(char)
        
        # Create prompt
        size_desc = ["small", "medium", "large"][i % 3]
        style_desc = ["plain", "bold", "italic"][i % 3]
        prompt = f"The letter {char} in {size_desc} {style_desc} font"
        
        samples.append(image)
        prompts.append(prompt)
    
    # Convert samples to tensors
    tensor_samples = []
    for image in samples:
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        # Convert to tensor
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        tensor_samples.append(tensor)
    
    # Create a grid of images with different noise levels
    fig, axes = plt.subplots(10, 7, figsize=(14, 20))
    
    # Set column titles
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i, title in enumerate(["Original"] + [f"{level*100:.0f}% Noise" for level in noise_levels]):
        axes[0, i].set_title(title)
    
    # For each sample
    for i, (image_tensor, prompt) in enumerate(zip(tensor_samples, prompts)):
        # Move tensor to device
        image_tensor = image_tensor.to(device)
        
        # Display original image
        axes[i, 0].imshow(samples[i], cmap='gray')
        axes[i, 0].set_ylabel(prompt, fontsize=8, rotation=0, ha='right', va='center')
        axes[i, 0].set_yticks([])
        
        # Generate a consistent noise pattern for this image
        # Use a hash of the image as seed
        image_seed = int(image_tensor.sum().item() * 1000)
        
        # For each noise level
        for j, noise_level in enumerate(noise_levels):
            # Add noise to image
            with torch.no_grad():
                noisy_image, applied_noise = model.add_noise(
                    image_tensor.unsqueeze(0),  # Add batch dimension
                    noise_level,
                    seed=image_seed
                )
            
            # Convert back to numpy for display
            noisy_np = noisy_image[0, 0].cpu().numpy()
            
            # Clip values to [0, 1] for display
            noisy_np = np.clip(noisy_np, 0, 1)
            
            # Display noisy image
            axes[i, j+1].imshow(noisy_np, cmap='gray')
            axes[i, j+1].set_xticks([])
            axes[i, j+1].set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "dataset_noise_examples.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Visualization saved to {output_path}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
