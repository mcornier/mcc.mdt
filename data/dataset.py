import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class GeometricShapesDataset(Dataset):
    """
    Dataset for geometric shapes with text descriptions.
    """
    def __init__(self, data_dir, tokenizer, transform=None, split="train"):
        """
        Args:
            data_dir: Directory containing the dataset
            tokenizer: TextTokenizer instance for encoding prompts
            transform: Optional transform to apply to images
            split: Dataset split ("train", "val", or "test")
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.split = split
        
        # Load metadata
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Get image paths and prompts
        self.image_paths = []
        self.prompts = []
        
        for item in self.metadata:
            self.image_paths.append(os.path.join(data_dir, item["image_path"]))
            self.prompts.append(item["prompt"])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
        
        Returns:
            Dictionary containing:
                - token_ids: Tensor of shape [seq_len]
                - image: Tensor of shape [1, 32, 32]
                - prompt: Original text prompt
        """
        # Get image path and prompt
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        
        # Load image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and normalize
            image = np.array(image) / 255.0
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Tokenize prompt
        token_ids = self.tokenizer.encode(prompt)[0]  # Get first item (batch dimension)
        
        return {
            "token_ids": token_ids,
            "image": image,
            "prompt": prompt
        }


def create_dataloader(data_dir, tokenizer, batch_size=32, num_workers=4, split="train"):
    """
    Create a DataLoader for the geometric shapes dataset.
    
    Args:
        data_dir: Directory containing the dataset
        tokenizer: TextTokenizer instance for encoding prompts
        batch_size: Batch size
        num_workers: Number of workers for data loading
        split: Dataset split ("train", "val", or "test")
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = GeometricShapesDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        transform=transform,
        split=split
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
