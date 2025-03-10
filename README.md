# Mini-Diffusion-Transformer

A mini transformer-based diffusion model for generating geometric shapes from text prompts.

## Architecture

This project implements a small-scale diffusion model using a transformer architecture to generate geometric shapes based on text descriptions. The model consists of three main modules:

1. **Input Module**:
   - Text input processing (limited to 12 tokens)
   - Image input processing (32x32 grayscale images)

2. **Transformer Module**:
   - Combines text and image representations
   - Uses attention mechanisms for text-only and text-image interactions

3. **Output Module**:
   - Image decoder to generate 32x32 images
   - Noise prediction for the diffusion process

## Dataset

The dataset consists of generated geometric shapes (squares, circles, triangles, etc.) and characters with various transformations (size, position, rotation). Each image is associated with corresponding text tokens.

## Requirements

See `requirements.txt` for the list of dependencies.

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate the dataset:
   ```
   python -m data.generator
   ```

3. Train the model:
   ```
   python train.py
   ```

4. Run inference:
   ```
   python inference.py --prompt "your text prompt"
