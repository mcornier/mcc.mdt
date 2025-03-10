import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .input_module import InputModule
from .transformer import TransformerModule
from .output_module import OutputModule


class DiffusionModel(nn.Module):
    """
    Complete diffusion model for generating geometric shapes from text prompts.
    Combines input, transformer, and output modules.
    """
    def __init__(
        self,
        text_embedding_dim=384,
        image_latent_dim=1024,
        transformer_dim=384,
        num_transformer_layers=3,
        num_heads=4,
        dropout=0.1,
        num_noise_levels=6
    ):
        super(DiffusionModel, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.image_latent_dim = image_latent_dim
        self.transformer_dim = transformer_dim
        self.num_noise_levels = num_noise_levels
        
        # Input module
        self.input_module = InputModule(
            text_embedding_dim=text_embedding_dim,
            image_latent_dim=image_latent_dim
        )
        
        # Transformer module
        self.transformer_module = TransformerModule(
            text_dim=text_embedding_dim,
            image_dim=image_latent_dim,
            num_layers=num_transformer_layers,
            heads=num_heads,
            dropout=dropout
        )
        
        # Output module
        self.output_module = OutputModule(latent_dim=transformer_dim)
        
        # Noise schedule (linear)
        self.register_buffer(
            'noise_schedule',
            torch.linspace(0.0, 1.0, num_noise_levels)
        )
        
    def add_noise(self, images, noise_level, noise=None, seed=None):
        """
        Add noise to images based on noise level.
        
        Args:
            images: Tensor of shape [batch_size, 1, 32, 32]
            noise_level: Float between 0 and 1
            noise: Optional pre-generated noise of same shape as images
            seed: Optional seed for noise generation. Can be either:
                 - A single integer: same seed used for all images
                 - A list/tensor of integers: one seed per image in the batch
        
        Returns:
            noisy_images: Tensor of shape [batch_size, 1, 32, 32]
            noise: The noise that was applied
        """
        batch_size = images.shape[0]
        
        # Generate noise if not provided
        if noise is None:
            # If seed is provided
            if seed is not None:
                # Check if seed is a list/tensor (per-image seeds) or a single value
                if isinstance(seed, (list, tuple)) or (isinstance(seed, torch.Tensor) and seed.numel() > 1):
                    # Per-image seeds
                    noise = torch.zeros_like(images)
                    for i in range(batch_size):
                        # Store current RNG state
                        rng_state = torch.get_rng_state()
                        torch.manual_seed(seed[i] if i < len(seed) else seed[0])
                        
                        # Generate noise for this image
                        noise[i] = torch.randn_like(images[i])
                        
                        # Restore RNG state
                        torch.set_rng_state(rng_state)
                else:
                    # Single seed for all images
                    rng_state = torch.get_rng_state()
                    torch.manual_seed(seed)
                    
                    # Generate random noise for all images
                    noise = torch.randn_like(images)
                    
                    # Restore RNG state
                    torch.set_rng_state(rng_state)
            else:
                # No seed provided, just generate random noise
                noise = torch.randn_like(images)
        
        # Interpolate between original image and noise
        noisy_images = (1 - noise_level) * images + noise_level * noise
        
        return noisy_images, noise
    
    def forward(self, token_ids, images, noise_level=None, seed=None, target_noise_offset=1):
        """
        Forward pass for training.
        
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
            images: Tensor of shape [batch_size, 1, 32, 32]
            noise_level: Optional float between 0 and 1.
                         If None, a random noise level is sampled.
            seed: Optional seed for noise generation
            target_noise_offset: Number of noise levels to reduce for target (default: 1)
                               0 means target is original image
        
        Returns:
            decoded_images: Tensor of shape [batch_size, 1, 32, 32]
            noise_pred: Tensor of shape [batch_size, num_noise_levels]
            noise_target: Tensor of shape [batch_size, num_noise_levels]
            applied_noise: The noise that was applied to the images
            target_images: The target images to aim for
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Sample noise level if not provided
        if noise_level is None:
            # Don't sample the lowest levels as they won't have a target
            max_idx = self.num_noise_levels - target_noise_offset
            if max_idx < 1:
                max_idx = 1
            noise_level_idx = torch.randint(0, max_idx, (1,)).item()
            noise_level = self.noise_schedule[noise_level_idx]
        else:
            # Find closest noise level
            noise_level_idx = torch.abs(self.noise_schedule - noise_level).argmin().item()
        
        # Create target for noise prediction
        noise_target = torch.zeros(batch_size, self.num_noise_levels, device=device)
        noise_target[:, noise_level_idx] = 1.0
        
        # Add noise to images for input
        noisy_images, applied_noise = self.add_noise(images, noise_level, seed=seed)
        
        # Create target images based on target_noise_offset
        if target_noise_offset == 0:
            target_images = images  # Original images
        else:
            target_noise_idx = max(0, noise_level_idx - target_noise_offset)
            target_noise_level = self.noise_schedule[target_noise_idx]
            target_images, _ = self.add_noise(images, target_noise_level, noise=applied_noise)
        
        # Process through input module
        text_features, image_features = self.input_module(token_ids, noisy_images)
        
        # Process through transformer module
        combined_features = self.transformer_module(text_features, image_features)
        
        # Process through output module
        decoded_images, noise_pred = self.output_module(combined_features)
        
        return decoded_images, noise_pred, noise_target, applied_noise, target_images
    
    def generate(self, token_ids, num_steps=50, temperature=1.0, seed=None):
        """
        Generate images from text prompts using the diffusion process.
        
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
            num_steps: Number of denoising steps
            temperature: Temperature for sampling
            seed: Optional seed for initial noise generation
        
        Returns:
            Generated images of shape [batch_size, 1, 32, 32]
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device
        
        # Set seed if provided for consistent initial noise
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            
        # Start with random noise
        initial_noise = torch.randn(batch_size, 1, 32, 32, device=device) * temperature
        images = initial_noise.clone()
        
        # Restore RNG state if seed was set
        if seed is not None:
            torch.set_rng_state(rng_state)
        
        # Gradually denoise the images
        for step in range(num_steps):
            # Calculate current noise level (from high to low)
            noise_level = 1.0 - step / (num_steps - 1)
            
            # Add noise to current images using the same initial noise pattern
            noisy_images, _ = self.add_noise(images, noise_level, noise=initial_noise)
            
            # Process through model
            with torch.no_grad():
                text_features, image_features = self.input_module(token_ids, noisy_images)
                combined_features = self.transformer_module(text_features, image_features)
                decoded_images, _ = self.output_module(combined_features)
            
            # Update images
            images = decoded_images
        
        return images
    
    def compute_loss(self, decoded_images, target_images, noise_pred, noise_target, applied_noise=None, target_noise_offset=1):
        """
        Compute loss for training.
        
        Args:
            decoded_images: Tensor of shape [batch_size, 1, 32, 32]
            target_images: Tensor of shape [batch_size, 1, 32, 32]
            noise_pred: Tensor of shape [batch_size, num_noise_levels]
            noise_target: Tensor of shape [batch_size, num_noise_levels]
            applied_noise: Optional tensor of shape [batch_size, 1, 32, 32]
                          representing the noise that was applied
        
        Returns:
            Total loss
        """
        # Image reconstruction loss (MSE)
        image_loss = F.mse_loss(decoded_images, target_images)
        
        # Noise prediction loss (Cross-entropy)
        noise_loss = F.cross_entropy(noise_pred, noise_target.argmax(dim=1))
        
        # Total loss (weighted sum)
        total_loss = image_loss + 0.1 * noise_loss
        
        return total_loss
