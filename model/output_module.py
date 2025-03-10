import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoder(nn.Module):
    """
    Decodes a latent vector back to a 32x32 grayscale image.
    Uses a small CNN architecture with transposed convolutions.
    """
    def __init__(self, latent_dim=384):
        super(ImageDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Initial projection from latent to spatial representation
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 32x32
        self.deconv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)              # 32x32, 1 channel
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, latent_dim]
        
        Returns:
            Tensor of shape [batch_size, 1, 32, 32]
            representing grayscale images
        """
        batch_size = x.shape[0]
        
        # Project to spatial representation
        x = self.fc(x)
        x = x.view(batch_size, 256, 4, 4)
        
        # Apply transposed convolution layers
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))  # Sigmoid to get values in [0, 1]
        
        return x


class NoisePredictor(nn.Module):
    """
    Predicts noise levels for the diffusion process.
    Outputs a vector of 6 values representing noise levels
    (0%, 20%, 40%, 60%, 80%, 100%).
    """
    def __init__(self, latent_dim=384):
        super(NoisePredictor, self).__init__()
        self.latent_dim = latent_dim
        
        # MLP for noise prediction
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 noise levels
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, latent_dim]
        
        Returns:
            Tensor of shape [batch_size, 6]
            representing noise level predictions
        """
        # Apply MLP
        noise_pred = self.mlp(x)
        
        # Apply softmax to get a probability distribution
        noise_pred = F.softmax(noise_pred, dim=-1)
        
        return noise_pred


class OutputModule(nn.Module):
    """
    Combined module for image decoding and noise prediction.
    """
    def __init__(self, latent_dim=384):
        super(OutputModule, self).__init__()
        self.latent_dim = latent_dim
        
        # Image decoder
        self.image_decoder = ImageDecoder(latent_dim=latent_dim)
        
        # Noise predictor
        self.noise_predictor = NoisePredictor(latent_dim=latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, latent_dim]
        
        Returns:
            decoded_image: Tensor of shape [batch_size, 1, 32, 32]
            noise_pred: Tensor of shape [batch_size, 6]
        """
        # Decode image
        decoded_image = self.image_decoder(x)
        
        # Predict noise
        noise_pred = self.noise_predictor(x)
        
        return decoded_image, noise_pred
