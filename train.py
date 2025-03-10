import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import DiffusionModel
from data import GeometricShapesDataset, create_dataloader
from utils import TextTokenizer, plot_generated_shapes, plot_diffusion_steps


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(max_length=args.max_tokens)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train"
    )
    
    val_dataloader = create_dataloader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val"
    )
    
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
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for batch in train_pbar:
            # Get batch data
            token_ids = batch["token_ids"].to(device)
            images = batch["image"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Generate seeds from image hashes for consistent noise patterns
            # Use a simple hash function based on the sum of pixel values
            image_seeds = [(img.sum() * 1000).int().item() for img in images]
            
            decoded_images, noise_pred, noise_target, applied_noise = model(token_ids, images, seed=image_seeds)
            
            # Compute loss
            loss = model.compute_loss(decoded_images, images, noise_pred, noise_target, applied_noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                # Get batch data
                token_ids = batch["token_ids"].to(device)
                images = batch["image"].to(device)
                
                # Forward pass
                
                # Generate seeds from image hashes for consistent noise patterns
                image_seeds = [(img.sum() * 1000).int().item() for img in images]
                
                decoded_images, noise_pred, noise_target, applied_noise = model(token_ids, images, seed=image_seeds)
                
                # Compute loss
                loss = model.compute_loss(decoded_images, images, noise_pred, noise_target, applied_noise)
                
                # Update progress bar
                val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average validation loss
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Generate samples
        if (epoch + 1) % args.sample_every == 0:
            generate_samples(model, tokenizer, device, args, epoch + 1)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print("Training completed!")


def generate_samples(model, tokenizer, device, args, epoch):
    """
    Generate samples from the model for visualization.
    """
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
        # Use a fixed seed for sample generation to ensure consistency across epochs
        sample_seed = 42  # Fixed seed for sample generation
        generated_images = model.generate(
            token_ids,
            num_steps=args.generation_steps,
            temperature=args.temperature,
            seed=sample_seed
        )
    
    # Plot and save generated images
    fig = plot_generated_shapes(generated_images, prompts)
    fig.savefig(os.path.join(args.output_dir, f"samples_epoch_{epoch}.png"))
    plt.close(fig)
    
    # Generate diffusion steps for the first prompt
    if args.save_diffusion_steps:
        steps = []
        step_titles = []
        num_viz_steps = 6  # Number of steps to visualize
        
        with torch.no_grad():
            # Start with random noise using fixed seed for consistency
            sample_seed = 42  # Fixed seed for visualization
            rng_state = torch.get_rng_state()
            torch.manual_seed(sample_seed)
            
            image = torch.randn(1, 1, 32, 32, device=device) * args.temperature
            initial_noise = image.clone()
            
            # Restore RNG state
            torch.set_rng_state(rng_state)
            steps.append(image[0].cpu())
            step_titles.append("Initial Noise")
            
            # Generate steps
            for step in range(args.generation_steps):
                # Only save a subset of steps
                if step % (args.generation_steps // (num_viz_steps - 1)) == 0 and len(steps) < num_viz_steps:
                    # Calculate current noise level
                    noise_level = 1.0 - step / (args.generation_steps - 1)
                    
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
        
        # Plot and save diffusion steps
        fig = plot_diffusion_steps(steps, step_titles)
        fig.savefig(os.path.join(args.output_dir, f"diffusion_steps_epoch_{epoch}.png"))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the diffusion model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/geometric_shapes",
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save models and results")
    
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
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every", type=int, default=5,
                        help="Generate samples every N epochs")
    
    # Generation parameters
    parser.add_argument("--generation_steps", type=int, default=50,
                        help="Number of steps for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--save_diffusion_steps", action="store_true",
                        help="Save visualization of diffusion steps")
    
    args = parser.parse_args()
    
    train(args)
