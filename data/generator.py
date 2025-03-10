import os
import json
import random
import numpy as np
import cv2
from tqdm import tqdm


class ShapeGenerator:
    """
    Generator for geometric shapes and characters using OpenCV.
    """
    def __init__(self, output_dir, image_size=32, num_samples=10000):
        """
        Args:
            output_dir: Directory to save generated images and metadata
            image_size: Size of the generated images (square)
            num_samples: Number of samples to generate
        """
        self.output_dir = output_dir
        self.image_size = image_size
        self.num_samples = num_samples
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Define shape types
        self.shape_types = ["circle", "square", "triangle", "rectangle", "ellipse"]
        
        # Define characters (letters)
        self.characters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Define colors (grayscale values)
        self.colors = [50, 100, 150, 200, 250]  # Different gray levels
        
    def generate_shape(self, shape_type, image=None, size_desc=None, position_desc=None):
        """
        Generate a single shape on a blank or existing image.
        
        Args:
            shape_type: Type of shape to generate
            image: Optional existing image to draw on
            size_desc: Optional size description ("small", "medium", "large")
            position_desc: Optional position description ("centered", "top left", etc.)
        
        Returns:
            image: Generated image
            params: Dictionary of shape parameters
        """
        # Create blank image if not provided
        if image is None:
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Set size based on description or random
        if size_desc == "small":
            size = random.randint(3, self.image_size // 6)  # 1/6 of image size
        elif size_desc == "medium":
            size = random.randint(self.image_size // 6, self.image_size // 4)  # 1/6 to 1/4 of image size
        elif size_desc == "large":
            size = random.randint(self.image_size // 4, self.image_size // 3)  # 1/4 to 1/3 of image size
        else:
            size = random.randint(3, self.image_size // 3)  # Random size
        
        # Calculate safe margins (10% of shape size)
        margin = max(int(size * 0.1), 1)
        
        # Set position based on description or random
        if position_desc == "centered":
            x = self.image_size // 2
            y = self.image_size // 2
        elif position_desc == "top left":
            x = size + margin
            y = size + margin
        elif position_desc == "top right":
            x = self.image_size - size - margin
            y = size + margin
        elif position_desc == "bottom left":
            x = size + margin
            y = self.image_size - size - margin
        elif position_desc == "bottom right":
            x = self.image_size - size - margin
            y = self.image_size - size - margin
        else:
            # Random position with safe margins
            x = random.randint(size + margin, self.image_size - size - margin)
            y = random.randint(size + margin, self.image_size - size - margin)
        
        color = random.choice(self.colors)
        rotation = random.randint(0, 360)
        
        params = {
            "type": shape_type,
            "x": x,
            "y": y,
            "size": size,
            "color": color,
            "rotation": rotation
        }
        
        # Draw shape based on type
        if shape_type == "circle":
            cv2.circle(image, (x, y), size, color, -1)
        
        elif shape_type == "square":
            # Create a rotated square
            rect = ((x, y), (size, size), rotation)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(image, [box], 0, color, -1)
        
        elif shape_type == "triangle":
            # Create a triangle
            height = int(size * 1.732)  # sqrt(3) for equilateral triangle
            points = np.array([
                [x, y - height // 2],
                [x - size, y + height // 2],
                [x + size, y + height // 2]
            ], np.int32)
            
            # Rotate the triangle
            if rotation > 0:
                # Create rotation matrix
                M = cv2.getRotationMatrix2D((x, y), rotation, 1.0)
                # Apply rotation to each point
                for i in range(3):
                    px, py = points[i]
                    px, py = M[0, 0] * px + M[0, 1] * py + M[0, 2], M[1, 0] * px + M[1, 1] * py + M[1, 2]
                    points[i] = [int(px), int(py)]
            
            cv2.fillPoly(image, [points], color)
        
        elif shape_type == "rectangle":
            # Create a rotated rectangle
            width = size
            height = size // 2
            rect = ((x, y), (width, height), rotation)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(image, [box], 0, color, -1)
        
        elif shape_type == "ellipse":
            # Create an ellipse
            axes = (size, size // 2)
            cv2.ellipse(image, (x, y), axes, rotation, 0, 360, color, -1)
        
        return image, params
    
    def generate_character(self, char, image=None, size_desc=None, style_desc=None):
        """
        Generate a single character on a blank or existing image.
        
        Args:
            char: Character to generate
            image: Optional existing image to draw on
            size_desc: Optional size description ("small", "medium", "large")
            style_desc: Optional style description ("plain", "bold", "italic")
        
        Returns:
            image: Generated image
            params: Dictionary of character parameters
        """
        # Create blank image if not provided
        if image is None:
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Set size based on description or random
        if size_desc == "small":
            size = random.randint(8, 12)  # Font size
        elif size_desc == "medium":
            size = random.randint(12, 16)  # Font size
        elif size_desc == "large":
            size = random.randint(16, 20)  # Font size
        else:
            size = random.randint(8, 20)  # Random size
        
        # Set thickness based on style
        if style_desc == "bold":
            thickness = random.randint(2, 3)  # Font thickness
        else:
            thickness = random.randint(1, 2)  # Font thickness
        
        # Calculate approximate character dimensions
        char_width = int(size * 0.7)  # Approximate width based on font size
        char_height = size
        
        # Calculate safe margins (10% of character size)
        margin_x = max(int(char_width * 0.1), 2)
        margin_y = max(int(char_height * 0.1), 2)
        
        # Set position with safe margins
        x = random.randint(margin_x, self.image_size - char_width - margin_x)
        y = random.randint(char_height + margin_y, self.image_size - margin_y)
        
        color = random.choice(self.colors)
        rotation = random.randint(0, 360)
        
        params = {
            "type": "character",
            "char": char,
            "x": x,
            "y": y,
            "size": size,
            "thickness": thickness,
            "color": color,
            "rotation": rotation
        }
        
        # Create text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # If no rotation, simply put text
        if rotation == 0:
            cv2.putText(image, char, (x, y), font, size / 20, color, thickness)
        else:
            # Create a temporary image for the rotated text
            temp = np.zeros_like(image)
            cv2.putText(temp, char, (x, y), font, size / 20, color, thickness)
            
            # Rotate the image
            M = cv2.getRotationMatrix2D((self.image_size // 2, self.image_size // 2), rotation, 1.0)
            rotated = cv2.warpAffine(temp, M, (self.image_size, self.image_size))
            
            # Add the rotated text to the original image
            image = cv2.add(image, rotated)
        
        return image, params
    
    def generate_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Generate the complete dataset.
        
        Args:
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Calculate split sizes
        train_size = int(self.num_samples * train_ratio)
        val_size = int(self.num_samples * val_ratio)
        test_size = self.num_samples - train_size - val_size
        
        # Generate samples
        all_metadata = []
        
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            # Decide whether to generate a shape or character
            is_shape = random.random() < 0.7  # 70% shapes, 30% characters
            
            if is_shape:
                # Generate a shape
                shape_type = random.choice(self.shape_types)
                image, params = self.generate_shape(shape_type)
                
                # Create prompt and generate shape with specific size and position
                size_desc = random.choice(["small", "medium", "large"])
                position_desc = random.choice(["centered", "top left", "top right", "bottom left", "bottom right"])
                prompt = f"A {size_desc} {shape_type} {position_desc}"
                image, params = self.generate_shape(shape_type, size_desc=size_desc, position_desc=position_desc)
                
            else:
                # Generate a character with specific size and style
                char = random.choice(self.characters)
                size_desc = random.choice(["small", "medium", "large"])
                style_desc = random.choice(["plain", "bold", "italic"])
                prompt = f"The letter {char} in {size_desc} {style_desc} font"
                image, params = self.generate_character(char, size_desc=size_desc, style_desc=style_desc)
            
            # Save image
            image_path = f"images/sample_{i:05d}.png"
            cv2.imwrite(os.path.join(self.output_dir, image_path), image)
            
            # Add metadata
            metadata = {
                "image_path": image_path,
                "prompt": prompt,
                "params": params
            }
            
            all_metadata.append(metadata)
        
        # Shuffle metadata
        random.shuffle(all_metadata)
        
        # Split into train, val, test
        train_metadata = all_metadata[:train_size]
        val_metadata = all_metadata[train_size:train_size + val_size]
        test_metadata = all_metadata[train_size + val_size:]
        
        # Save metadata
        with open(os.path.join(self.output_dir, "train_metadata.json"), "w") as f:
            json.dump(train_metadata, f, indent=2)
        
        with open(os.path.join(self.output_dir, "val_metadata.json"), "w") as f:
            json.dump(val_metadata, f, indent=2)
        
        with open(os.path.join(self.output_dir, "test_metadata.json"), "w") as f:
            json.dump(test_metadata, f, indent=2)
        
        print(f"Generated {self.num_samples} samples:")
        print(f"  - Train: {len(train_metadata)}")
        print(f"  - Val: {len(val_metadata)}")
        print(f"  - Test: {len(test_metadata)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate geometric shapes dataset")
    parser.add_argument("--output_dir", type=str, default="data/geometric_shapes",
                        help="Output directory for the dataset")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Size of the generated images")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Create generator
    generator = ShapeGenerator(
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_samples=args.num_samples
    )
    
    # Generate dataset
    generator.generate_dataset()
