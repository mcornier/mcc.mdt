import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def plot_diffusion_steps(images, titles=None, figsize=(15, 3)):
    """
    Plot a sequence of diffusion steps.
    
    Args:
        images: List of tensors or numpy arrays of shape [1, 32, 32] or [32, 32]
        titles: Optional list of titles for each image
        figsize: Figure size (width, height)
    """
    num_images = len(images)
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    
    # Plot each image
    for i, (img, ax) in enumerate(zip(images, axes)):
        # Convert tensor to numpy if needed
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
            
        # Reshape if needed
        if img.shape[0] == 1:  # [1, 32, 32]
            img = img.squeeze(0)
            
        # Plot
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # Add title if provided
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    
    plt.tight_layout()
    return fig


def plot_generated_shapes(images, prompts=None, nrow=4, figsize=(10, 10)):
    """
    Plot a grid of generated shapes.
    
    Args:
        images: Tensor of shape [batch_size, 1, 32, 32]
        prompts: Optional list of text prompts for each image
        nrow: Number of images per row
        figsize: Figure size (width, height)
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(images):
        # Create grid
        grid = make_grid(images, nrow=nrow, normalize=True, pad_value=1)
        # Convert to numpy and transpose
        grid = grid.detach().cpu().numpy().transpose(1, 2, 0)
        # Convert to grayscale if needed
        if grid.shape[2] == 1:
            grid = grid.squeeze(2)
    else:
        # Assume it's already a grid
        grid = images
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grid
    ax.imshow(grid, cmap='gray')
    ax.axis('off')
    
    # Add prompts as title if provided
    if prompts is not None:
        prompt_str = ' | '.join(prompts[:min(len(prompts), 5)])
        if len(prompts) > 5:
            prompt_str += ' | ...'
        ax.set_title(prompt_str)
    
    plt.tight_layout()
    return fig


def save_diffusion_animation(images, filename='diffusion_animation.gif', fps=5):
    """
    Save a sequence of diffusion steps as an animated GIF.
    
    Args:
        images: List of tensors or numpy arrays of shape [1, 32, 32] or [32, 32]
        filename: Output filename
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio")
        return
    
    # Convert tensors to numpy if needed
    frames = []
    for img in images:
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
            
        # Reshape if needed
        if img.shape[0] == 1:  # [1, 32, 32]
            img = img.squeeze(0)
            
        # Normalize to [0, 255]
        img = (img * 255).astype(np.uint8)
        
        frames.append(img)
    
    # Save as GIF
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Animation saved to {filename}")
