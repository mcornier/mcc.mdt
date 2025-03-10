import os
import torch
import matplotlib.pyplot as plt

from model import DiffusionModel
from utils import TextTokenizer, plot_generated_shapes, plot_diffusion_steps


def main(seed=42):
    """
    Simple example of how to use the diffusion model.
    """
    # Set device and seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create output directory
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(max_length=12)
    
    # Initialize model
    model = DiffusionModel(
        text_embedding_dim=384,
        image_latent_dim=1024,
        transformer_dim=384,
        num_transformer_layers=3,
        num_heads=4,
        dropout=0.1,
        num_noise_levels=6
    ).to(device)
    
    # Load model weights if available
    model_path = "output/best_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"No pre-trained model found at {model_path}. Using randomly initialized weights.")
    
    model.eval()
    
    # Define prompts
    prompts = [
        "A large circle centered",
        "A small square top left",
        "A medium triangle bottom right",
        "The letter A in large bold font",
        "A small ellipse bottom"
    ]
    
    # Tokenize prompts
    token_ids = tokenizer.encode(prompts).to(device)
    
    # Generate images
    with torch.no_grad():
        generated_images = model.generate(
            token_ids,
            num_steps=50,
            temperature=1.0,
            seed=seed
        )
    
    # Plot and save generated images
    fig = plot_generated_shapes(generated_images, prompts)
    fig.savefig(os.path.join(output_dir, "example_generated_images.png"))
    plt.close(fig)
    print(f"Generated images saved to {os.path.join(output_dir, 'example_generated_images.png')}")
    
    # Generate diffusion steps for the first prompt
    steps = []
    step_titles = []
    num_viz_steps = 6
    
    with torch.no_grad():
        # Start with random noise using seed for consistency
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        image = torch.randn(1, 1, 32, 32, device=device)
        initial_noise = image.clone()
        torch.set_rng_state(rng_state)
        steps.append(image[0].cpu())
        step_titles.append("Initial Noise")
        
        # Generate steps
        for step in range(50):  # 50 generation steps
            # Only save a subset of steps
            if step % (50 // (num_viz_steps - 1)) == 0 and len(steps) < num_viz_steps:
                # Calculate current noise level
                noise_level = 1.0 - step / 49
                
                # Add noise to current image with consistent noise pattern
                noisy_image, _ = model.add_noise(image, noise_level, noise=initial_noise)
                
                # Process through model
                text_features, image_features = model.input_module(token_ids[0:1], noisy_image)
                combined_features = model.transformer_module(text_features, image_features)
                decoded_image, _ = model.output_module(combined_features)
                
                # Update image
                image = decoded_image
                
                # Save step
                steps.append(image[0].cpu())
                step_titles.append(f"Step {step}")
            else:
                # Calculate current noise level
                noise_level = 1.0 - step / 49
                
                # Add noise to current image with consistent noise pattern
                noisy_image, _ = model.add_noise(image, noise_level, noise=initial_noise)
                
                # Process through model
                text_features, image_features = model.input_module(token_ids[0:1], noisy_image)
                combined_features = model.transformer_module(text_features, image_features)
                decoded_image, _ = model.output_module(combined_features)
                
                # Update image
                image = decoded_image
    
    # Plot and save diffusion steps
    fig = plot_diffusion_steps(steps, step_titles)
    fig.savefig(os.path.join(output_dir, "example_diffusion_steps.png"))
    plt.close(fig)
    print(f"Diffusion steps saved to {os.path.join(output_dir, 'example_diffusion_steps.png')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of the diffusion model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(seed=args.seed)
