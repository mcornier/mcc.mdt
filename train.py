import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

from model import DiffusionModel
from data import GeometricShapesDataset, create_dataloader
from utils import TextTokenizer, plot_generated_shapes, plot_diffusion_steps


def print_gpu_memory_usage():
    """Print GPU memory usage for all available GPUs."""
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024  # GB
        reserved_memory = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024  # GB
        allocated_memory = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024  # GB
        free_memory = total_memory - reserved_memory
        print(f"GPU {i}: Total: {total_memory:.1f} GB | Reserved: {reserved_memory:.1f} GB | Allocated: {allocated_memory:.1f} GB | Free: {free_memory:.1f} GB")

def train(args):
    # Set memory management options for PyTorch
    if args.memory_efficient:
        # Enable memory-efficient features
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # List GPU information
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024  # Convert to GB
        print(f"GPU {i}: {gpu_name} with {gpu_mem:.1f} GB memory")
    
    # Set primary device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")
    
    # Exclude RTX 3070 if present (has less memory than 3090s)
    device_ids = None
    if num_gpus > 2:
        # Check if we have a mix of 3090 and 3070
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        if any('3070' in name for name in gpu_names):
            # Only use the 3090s
            device_ids = [i for i, name in enumerate(gpu_names) if '3090' in name]
            print(f"Using only RTX 3090 GPUs: {device_ids}")
    
    # Training phases configuration
    phases = [
        {"epochs": 15, "target_noise_offset": 1, "description": "Phase 1: 1-step denoising"},
        {"epochs": 10, "target_noise_offset": 2, "description": "Phase 2: 2-step denoising"},
        {"epochs": args.num_epochs - 25, "target_noise_offset": 0, "description": "Phase 3: Full denoising"}
    ]
    
    current_epoch = 0
    
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
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        if device_ids:
            print(f"Using DataParallel across GPUs: {device_ids}")
            model = DataParallel(model, device_ids=device_ids)
        else:
            print(f"Using DataParallel across {num_gpus} GPUs")
            model = DataParallel(model)
    
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
    
    # Training loop
    for phase in phases:
        print(f"\nStarting {phase['description']}")
        print(f"Training for {phase['epochs']} epochs with target_noise_offset={phase['target_noise_offset']}")
        
        for epoch_in_phase in range(phase['epochs']):
            current_epoch += 1
            
            # Training
            model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_dataloader, desc=f"Epoch {current_epoch}/{args.num_epochs} [Train]")
        
            # Implement gradient accumulation
            accumulation_steps = args.gradient_accumulation_steps
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_pbar):
                # Get batch data
                token_ids = batch["token_ids"].to(device)
                images = batch["image"].to(device)
                
                # Generate seeds from image hashes for consistent noise patterns
                # Use a simple hash function based on the sum of pixel values
                image_seeds = [(img.sum() * 1000).int().item() for img in images]
                
                # Forward pass
                decoded_images, noise_pred, noise_target, applied_noise, target_images = model(
                    token_ids, 
                    images, 
                    seed=image_seeds,
                    target_noise_offset=phase['target_noise_offset']
                )
                
                # Compute loss - handle both DataParallel and non-DataParallel cases
                if isinstance(model, DataParallel):
                    loss = model.module.compute_loss(
                        decoded_images, 
                        target_images, 
                        noise_pred, 
                        noise_target, 
                        applied_noise,
                        target_noise_offset=phase['target_noise_offset']
                    )
                else:
                    loss = model.compute_loss(
                        decoded_images, 
                        target_images, 
                        noise_pred, 
                        noise_target, 
                        applied_noise,
                        target_noise_offset=phase['target_noise_offset']
                    )
                
                # Scale the loss to account for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights only after accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Free up memory
                del decoded_images, noise_pred, noise_target, applied_noise, target_images
                if args.memory_efficient and (batch_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                
                # Update progress bar
                train_loss += loss.item() * accumulation_steps  # Scale back for reporting
                train_pbar.set_postfix({"loss": loss.item() * accumulation_steps})
                
                # Print memory usage periodically
                if args.memory_efficient and (batch_idx + 1) % 10 == 0:
                    print_gpu_memory_usage()
            
            # Calculate average training loss
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {current_epoch}/{args.num_epochs} [Val]")
        
            with torch.no_grad():
                for batch in val_pbar:
                    # Get batch data
                    token_ids = batch["token_ids"].to(device)
                    images = batch["image"].to(device)
                    
                    # Forward pass
                    
                    # Generate seeds from image hashes for consistent noise patterns
                    image_seeds = [(img.sum() * 1000).int().item() for img in images]
                    
                    decoded_images, noise_pred, noise_target, applied_noise, target_images = model(
                        token_ids, 
                        images, 
                        seed=image_seeds,
                        target_noise_offset=phase['target_noise_offset']
                    )
                    
                    # Compute loss - handle both DataParallel and non-DataParallel cases
                    if isinstance(model, DataParallel):
                        loss = model.module.compute_loss(
                            decoded_images, 
                            target_images, 
                            noise_pred, 
                            noise_target, 
                            applied_noise,
                            target_noise_offset=phase['target_noise_offset']
                        )
                    else:
                        loss = model.compute_loss(
                            decoded_images, 
                            target_images, 
                            noise_pred, 
                            noise_target, 
                            applied_noise,
                            target_noise_offset=phase['target_noise_offset']
                        )
                    
                    # Update progress bar
                    val_loss += loss.item()
                    val_pbar.set_postfix({"loss": loss.item()})
            
            # Calculate average validation loss
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"Epoch {current_epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if current_epoch % args.save_every == 0:
                torch.save({
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{current_epoch}.pt"))
            
            # Generate samples
            if current_epoch % args.sample_every == 0:
                generate_samples(model, tokenizer, device, args, current_epoch)
    
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
        # Handle both DataParallel and non-DataParallel cases
        if isinstance(model, DataParallel):
            generated_images = model.module.generate(
                token_ids,
                num_steps=args.generation_steps,
                temperature=args.temperature,
                seed=sample_seed
            )
        else:
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
                    # Handle both DataParallel and non-DataParallel cases
                    if isinstance(model, DataParallel):
                        noisy_image, _ = model.module.add_noise(image, noise_level, noise=initial_noise)
                        
                        # Process through model
                        text_features, image_features = model.module.input_module(token_ids[0:1], noisy_image)
                        combined_features = model.module.transformer_module(text_features, image_features)
                        decoded_image, _ = model.module.output_module(combined_features)
                    else:
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
    
    # Memory management parameters
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating weights")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="Enable memory-efficient training options")
    
    # Generation parameters
    parser.add_argument("--generation_steps", type=int, default=50,
                        help="Number of steps for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--save_diffusion_steps", action="store_true",
                        help="Save visualization of diffusion steps")
    
    args = parser.parse_args()
    
    train(args)
