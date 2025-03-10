import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TextEncoder(nn.Module):
    """
    Encodes text prompts using embeddings from a pre-trained model.
    Limited to 12 tokens as specified in the architecture.
    """
    def __init__(self, embedding_dim=384):
        super(TextEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_tokens = 12
        
        # Placeholder for embeddings from MiniLM-v6
        # In practice, this would use the transformers library
        self.embedding = nn.Embedding(30000, embedding_dim)  # Vocabulary size of 30k
        
        # Position encoding
        self.position_encoding = nn.Parameter(
            torch.zeros(1, self.max_tokens, embedding_dim)
        )
        
    def forward(self, token_ids):
        """
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
                       where seq_len <= max_tokens
        
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Ensure we don't exceed max tokens
        if seq_len > self.max_tokens:
            token_ids = token_ids[:, :self.max_tokens]
            seq_len = self.max_tokens
            
        # Get embeddings
        embeddings = self.embedding(token_ids)
        
        # Add positional encoding
        embeddings = embeddings + self.position_encoding[:, :seq_len, :]
        
        return embeddings


class ImageEncoder(nn.Module):
    """
    Encodes 32x32 grayscale images into a latent vector of size 1024.
    Uses a small CNN architecture.
    """
    def __init__(self, latent_dim=1024):
        super(ImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 4x4
        
        # Fully connected layers
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Normalization
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, 1, 32, 32]
               representing grayscale images
        
        Returns:
            Tensor of shape [batch_size, latent_dim]
        """
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layer
        x = self.fc(x)
        
        # Apply normalization
        x = self.norm(x)
        
        return x


class InputModule(nn.Module):
    """
    Combined module for processing both text and image inputs.
    """
    def __init__(self, text_embedding_dim=384, image_latent_dim=1024):
        super(InputModule, self).__init__()
        self.text_encoder = TextEncoder(embedding_dim=text_embedding_dim)
        self.image_encoder = ImageEncoder(latent_dim=image_latent_dim)
        
    def forward(self, token_ids, images):
        """
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
            images: Tensor of shape [batch_size, 1, 32, 32]
        
        Returns:
            text_features: Tensor of shape [batch_size, seq_len, text_embedding_dim]
            image_features: Tensor of shape [batch_size, image_latent_dim]
        """
        text_features = self.text_encoder(token_ids)
        image_features = self.image_encoder(images)
        
        return text_features, image_features
