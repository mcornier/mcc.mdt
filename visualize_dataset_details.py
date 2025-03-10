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
        num_samples=10
    )
    
    # Initialize diffusion model (we only need it for the add_noise method)
    model = DiffusionModel().to(device)
    
    # Generate examples of each shape type
    shape_examples = {}
    for shape_type in generator.shape_types:
        # Generate 3 examples of each shape type
        examples = []
        for i in range(3):
            # Use different size descriptions
            size_desc = ["small", "medium", "large"][i % 3]
            # Use different position descriptions
            position_desc = ["centered", "top left", "bottom right"][i % 3]
            
            # Generate shape with specific size and position
            image, params = generator.generate_shape(
                shape_type, 
                size_desc=size_desc, 
                position_desc=position_desc
            )
            prompt = f"A {size_desc} {shape_type} {position_desc}"
            
            examples.append((image, prompt))
        
        shape_examples[shape_type] = examples
    
    # Generate examples of characters
    char_examples = []
    for i, char in enumerate("ABCDE"):
        # Use different size descriptions
        size_desc = ["small", "medium", "large"][i % 3]
        # Use different style descriptions
        style_desc = ["plain", "bold", "italic"][i % 3]
        
        # Generate character with specific size and style
        image, params = generator.generate_character(
            char,
            size_desc=size_desc,
            style_desc=style_desc
        )
        prompt = f"The letter {char} in {size_desc} {style_desc} font"
        
        char_examples.append((image, prompt))
    
    # Create a figure for each shape type and characters
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Function to create a figure for a set of examples
    def create_figure(examples, title):
        fig, axes = plt.subplots(len(examples), len(noise_levels) + 1, figsize=(14, 3 * len(examples)))
        
        # Set title
        fig.suptitle(title, fontsize=16)
        
        # Set column titles
        for i, title in enumerate(["Original"] + [f"{level*100:.0f}% Noise" for level in noise_levels]):
            if len(examples) > 1:
                axes[0, i].set_title(title)
            else:
                axes[i].set_title(title)
        
        # For each example
        for i, (image, prompt) in enumerate(examples):
            # Convert to tensor
            normalized = image.astype(np.float32) / 255.0
            image_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Generate a consistent noise pattern for this image
            image_seed = int(image_tensor.sum().item() * 1000)
            
            # Display original image
            if len(examples) > 1:
                ax = axes[i, 0]
            else:
                ax = axes[0]
                
            ax.imshow(image, cmap='gray')
            ax.set_ylabel(prompt, fontsize=8, rotation=0, ha='right', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            
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
                if len(examples) > 1:
                    ax = axes[i, j+1]
                else:
                    ax = axes[j+1]
                    
                ax.imshow(noisy_np, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        return fig
    
    # Create and save figures for each shape type
    for shape_type, examples in shape_examples.items():
        fig = create_figure(examples, f"{shape_type.capitalize()} Examples with Different Noise Levels")
        fig.savefig(os.path.join(output_dir, f"{shape_type}_examples.png"), dpi=150)
        plt.close(fig)
    
    # Create and save figure for characters
    fig = create_figure(char_examples, "Character Examples with Different Noise Levels")
    fig.savefig(os.path.join(output_dir, "character_examples.png"), dpi=150)
    plt.close(fig)
    
    # Create a figure to demonstrate noise consistency
    # We'll use a circle and apply the same noise pattern at different levels
    circle_image, _ = generator.generate_shape("circle")
    normalized = circle_image.astype(np.float32) / 255.0
    circle_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate two different noise patterns
    noise_seed1 = 42
    noise_seed2 = 43
    
    fig, axes = plt.subplots(2, len(noise_levels), figsize=(14, 5))
    
    # Set title
    fig.suptitle("Noise Pattern Consistency Demonstration", fontsize=16)
    
    # For each noise level
    for i, noise_level in enumerate(noise_levels):
        # Set column titles
        axes[0, i].set_title(f"{noise_level*100:.0f}% Noise")
        
        # Add noise with pattern 1
        with torch.no_grad():
            noisy_image1, _ = model.add_noise(
                circle_tensor.unsqueeze(0),
                noise_level,
                seed=noise_seed1
            )
        
        # Add noise with pattern 2
        with torch.no_grad():
            noisy_image2, _ = model.add_noise(
                circle_tensor.unsqueeze(0),
                noise_level,
                seed=noise_seed2
            )
        
        # Convert to numpy and clip
        noisy_np1 = np.clip(noisy_image1[0, 0].cpu().numpy(), 0, 1)
        noisy_np2 = np.clip(noisy_image2[0, 0].cpu().numpy(), 0, 1)
        
        # Display images
        axes[0, i].imshow(noisy_np1, cmap='gray')
        axes[1, i].imshow(noisy_np2, cmap='gray')
        
        # Remove ticks
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Add row labels
    axes[0, 0].set_ylabel("Noise Pattern 1", fontsize=10)
    axes[1, 0].set_ylabel("Noise Pattern 2", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save figure
    fig.savefig(os.path.join(output_dir, "noise_consistency.png"), dpi=150)
    plt.close(fig)
    
    print(f"Visualizations saved to {output_dir}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
