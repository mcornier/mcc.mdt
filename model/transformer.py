import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for the transformer.
    """
    def __init__(self, dim, heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, dim]
        
        Returns:
            Tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Apply output projection
        out = self.out_proj(out)
        
        return out


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for combining text and image features.
    """
    def __init__(self, text_dim, image_dim, heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.heads = heads
        self.head_dim = text_dim // heads
        
        # Project image features to match text dimension
        self.image_proj = nn.Linear(image_dim, text_dim)
        
        # Query, Key, Value projections
        self.query = nn.Linear(text_dim, text_dim)
        self.key = nn.Linear(text_dim, text_dim)
        self.value = nn.Linear(text_dim, text_dim)
        
        # Output projection
        self.out_proj = nn.Linear(text_dim, text_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: Tensor of shape [batch_size, text_seq_len, text_dim]
            image_features: Tensor of shape [batch_size, image_dim]
        
        Returns:
            Tensor of shape [batch_size, text_seq_len, text_dim]
        """
        batch_size, text_seq_len, _ = text_features.shape
        
        # Project image features to match text dimension
        # and expand to create a sequence of length 1
        image_features = self.image_proj(image_features).unsqueeze(1)  # [batch_size, 1, text_dim]
        
        # Concatenate text and image features
        # This creates a sequence where the last token is the image representation
        combined_features = torch.cat([text_features, image_features], dim=1)
        combined_seq_len = combined_features.shape[1]
        
        # Project queries, keys, values
        q = self.query(text_features).view(batch_size, text_seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(combined_features).view(batch_size, combined_seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(combined_features).view(batch_size, combined_seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, text_seq_len, self.text_dim)
        
        # Apply output projection
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """
    def __init__(self, dim, heads=4, ff_hidden_dim=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        ff_hidden_dim = ff_hidden_dim or dim * 4
        
        # Self-attention
        self.self_attn = SelfAttention(dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ff = FeedForward(dim, ff_hidden_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.self_attn(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        
        return x


class TransformerModule(nn.Module):
    """
    Transformer module that combines text and image features.
    Implements the architecture described with:
    1. First attention mechanism on text only
    2. Second attention mechanism mixing text and image
    """
    def __init__(self, text_dim=384, image_dim=1024, num_layers=3, heads=4, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        
        # Text-only transformer blocks
        self.text_blocks = nn.ModuleList([
            TransformerBlock(text_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention for text and image
        self.cross_attn = CrossAttention(text_dim, image_dim, heads=heads, dropout=dropout)
        self.norm_cross = nn.LayerNorm(text_dim)
        
        # Final transformer blocks after cross-attention
        self.final_blocks = nn.ModuleList([
            TransformerBlock(text_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Project image features for output
        self.image_proj = nn.Linear(image_dim, text_dim)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(text_dim)
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: Tensor of shape [batch_size, text_seq_len, text_dim]
            image_features: Tensor of shape [batch_size, image_dim]
        
        Returns:
            combined_features: Tensor of shape [batch_size, text_dim]
            representing the combined text and image features
        """
        # Process text features through text-only transformer blocks
        for block in self.text_blocks:
            text_features = block(text_features)
        
        # Apply cross-attention between text and image
        text_features = text_features + self.cross_attn(self.norm_cross(text_features), image_features)
        
        # Process through final transformer blocks
        for block in self.final_blocks:
            text_features = block(text_features)
        
        # Apply final normalization
        text_features = self.final_norm(text_features)
        
        # Average pooling over sequence dimension to get a single vector
        pooled_text = text_features.mean(dim=1)
        
        # Project image features
        projected_image = self.image_proj(image_features)
        
        # Combine text and image features
        combined_features = pooled_text + projected_image
        
        return combined_features
