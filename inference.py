import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from model import DiffusionModel
from utils import TextTokenizer, plot_generated_shapes, plot_diffusion_steps, save_diffusion_animation


def load_model(model_path, args, device):
    """
    Load a trained model from a checkpoint.
    """
    # Initialize model
    model = DiffusionModel(
        text_embedding_dim=args.text_embedding_dim,
        image_latent_dim=args.image_latent_dim,
        transformer_dim=args.transformer_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_noise_levels=args.num_noise_levels
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def generate_from_prompt(model, tokenizer, prompt, args, device, seed=None):
    """
    Generate an image from a text prompt.
    
    Args:
        model: The diffusion model
        tokenizer: The text tokenizer
        prompt: Text prompt for generation
        args: Command line arguments
        device: Device to run on
        seed: Optional seed for consistent noise generation
    """
    # Tokenize prompt
    token_ids = tokenizer.encode(prompt).to(device)
    
    # Generate image
    with torch.no_grad():
        generated_image = model.generate(
            token_ids,
            num_steps=args.generation_steps,
            temperature=args.temperature,
            seed=seed
        )
    
    return generated_image


def generate_diffusion_steps(model, tokenizer, prompt, args, device, seed=None):
    """
    Generate and visualize the diffusion steps for a prompt.
    """
    # Tokenize prompt
    token_ids = tokenizer.encode(prompt).to(device)
    
    # Generate steps
    steps = []
    step_titles = []
    
    with torch.no_grad():
        # Start with random noise with seed for consistency
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            
        image = torch.randn(1, 1, 32, 32, device=device) * args.temperature
        initial_noise = image.clone()
        
        if seed is not None:
            torch.set_rng_state(rng_state)
        steps.append(image[0].cpu())
        step_titles.append("Initial Noise")
        
        # Generate steps
        for step in range(args.generation_steps):
            # Save steps at regular intervals
            if step % (args.generation_steps // (args.num_viz_steps - 1)) == 0 or step == args.generation_steps - 1:
                # Calculate current noise level
                noise_level = 1.0 - step / (args.generation_steps - 1)
                
                # Add noise to current image using consistent noise pattern
                noisy_image, _ = model.add_noise(image, noise_level, noise=initial_noise)
                
                # Process through model
                text_features, image_features = model.input_module(token_ids, noisy_image)
                combined_features = model.transformer_module(text_features, image_features)
                decoded_image, _ = model.output_module(combined_features)
                
                # Update image
                image = decoded_image
                
                # Save step
                steps.append(image[0].cpu())
                step_titles.append(f"Step {step}")
            else:
                # Calculate current noise level
                noise_level = 1.0 - step / (args.generation_steps - 1)
                
                # Add noise to current image using consistent noise pattern
                noisy_image, _ = model.add_noise(image, noise_level, noise=initial_noise)
                
                # Process through model
                text_features, image_features = model.input_module(token_ids, noisy_image)
                combined_features = model.transformer_module(text_features, image_features)
                decoded_image, _ = model.output_module(combined_features)
                
                # Update image
                image = decoded_image
    
    return steps, step_titles


def main(args):
    # Set seed for reproducibility if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(max_length=args.max_tokens)
    
    # Load model
    model = load_model(args.model_path, args, device)
    print(f"Model loaded from {args.model_path}")
    
    # Generate from prompts
    if args.prompt:
        # Single prompt mode
        prompt = args.prompt
        print(f"Generating image for prompt: '{prompt}'")
        
        # Generate image
        generated_image = generate_from_prompt(model, tokenizer, prompt, args, device, seed=args.seed)
        
        # Save image
        output_path = os.path.join(args.output_dir, "generated_image.png")
        plt.figure(figsize=(5, 5))
        plt.imshow(generated_image[0, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(prompt)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Image saved to {output_path}")
        
        # Generate diffusion steps if requested
        if args.save_steps:
            steps, step_titles = generate_diffusion_steps(model, tokenizer, prompt, args, device, seed=args.seed)
            
            # Plot steps
            fig = plot_diffusion_steps(steps, step_titles)
            steps_path = os.path.join(args.output_dir, "diffusion_steps.png")
            fig.savefig(steps_path)
            plt.close(fig)
            print(f"Diffusion steps saved to {steps_path}")
            
            # Save animation if requested
            if args.save_animation:
                animation_path = os.path.join(args.output_dir, "diffusion_animation.gif")
                save_diffusion_animation(steps, animation_path, fps=args.fps)
    
    elif args.prompts_file:
        # Multiple prompts mode
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Generating images for {len(prompts)} prompts")
        
        # Generate images for all prompts
        all_images = []
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: '{prompt}'")
            generated_image = generate_from_prompt(model, tokenizer, prompt, args, device, seed=args.seed)
            all_images.append(generated_image[0])
        
        # Stack images into a batch
        batch_images = torch.stack(all_images)
        
        # Plot grid of images
        fig = plot_generated_shapes(batch_images, prompts)
        grid_path = os.path.join(args.output_dir, "generated_grid.png")
        fig.savefig(grid_path)
        plt.close(fig)
        print(f"Grid of images saved to {grid_path}")
        
        # Save individual images if requested
        if args.save_individual:
            for i, (prompt, image) in enumerate(zip(prompts, all_images)):
                # Create a safe filename from the prompt
                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
                safe_prompt = safe_prompt[:50]  # Limit length
                
                # Save image
                img_path = os.path.join(args.output_dir, f"{i+1:02d}_{safe_prompt}.png")
                plt.figure(figsize=(5, 5))
                plt.imshow(image[0].cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title(prompt)
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close()
    
    else:
        print("Please provide either a prompt or a file containing prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    
    # Input parameters
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for image generation")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File containing text prompts (one per line)")
    parser.add_argument("--model_path", type=str, default="output/best_model.pt",
                        help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="output/generated",
                        help="Directory to save generated images")
    
    # Model parameters
    parser.add_argument("--text_embedding_dim", type=int, default=384,
                        help="Dimension of text embeddings")
    parser.add_argument("--image_latent_dim", type=int, default=1024,
                        help="Dimension of image latent representation")
    parser.add_argument("--transformer_dim", type=int, default=384,
                        help="Dimension of transformer")
    parser.add_argument("--num_transformer_layers", type=int, default=3,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--num_noise_levels", type=int, default=6,
                        help="Number of noise levels")
    parser.add_argument("--max_tokens", type=int, default=12,
                        help="Maximum number of tokens in prompt")
    
    # Generation parameters
    parser.add_argument("--generation_steps", type=int, default=50,
                        help="Number of steps for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--save_steps", action="store_true",
                        help="Save visualization of diffusion steps")
    parser.add_argument("--num_viz_steps", type=int, default=6,
                        help="Number of diffusion steps to visualize")
    parser.add_argument("--save_animation", action="store_true",
                        help="Save animation of diffusion steps")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second for animation")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual images when using prompts_file")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if CUDA is available")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    main(args)
