# MNIST Diffusion Model

## Overview
A PyTorch implementation of a conditional diffusion model for generating MNIST digits. This project demonstrates the application of denoising diffusion probabilistic models (DDPM) to generate handwritten digits with controllable output through conditioning.

## Features
- **Conditional Generation**: Generate specific digits (0-9) using class conditioning
- **Interactive Mode**: User-friendly interface for both training and inference
- **Visualization**: Real-time training progress visualization
- **Checkpointing**: Automatic save/resume functionality
- **Progress Tracking**: Regular sample generation during training

## Requirements
pip install torch torchvision matplotlib pillow

## Model Architecture
### ConditionalUNet
- U-Net architecture with skip connections
- Conditional embedding for digit classes
- Time embedding for diffusion steps
- Input: 28x28 grayscale images
- Output: Noise prediction for denoising process

### Diffusion Process
- Forward Process: Gradually adds noise to images over 500 timesteps
- Reverse Process: Learns to denoise images conditioned on digit class
- Beta Schedule: Linear schedule from 1e-4 to 0.02

## Usage
### Training
# Start training with command line arguments
python script.py --mode train --epochs 200 --batch_size 128

# Or run interactively
python script.py

Training Parameters:
- Batch Size: 128 (default)
- Learning Rate: 1e-4
- Optimizer: AdamW with weight decay 1e-5
- Device: Automatically uses GPU if available

### Inference
# Run inference with trained model
python script.py --mode inference

## Project Structure
├── mnistdiffuser_individual_Jan_21_25.py  # Main script
├── data/                                  # MNIST dataset (downloaded automatically)
├── samples/                               # Generated samples during training
│   └── epoch_*.png                        # Sample grids for each epoch
├── checkpoints/                           # Periodic model checkpoints
│   └── diffusion_checkpoint_epoch_*.pt    # Saved every 10 epochs
├── inference_samples/                     # Generated samples during inference
│   └── digit_*_*.png                      # Individual generated digits
├── diffusion_checkpoint.pt                # Latest checkpoint
└── conditional_diffusion_mnist.pth        # Final trained model

## Training Progress
The model typically progresses through several stages:
1. **Early Stage** (Loss > 0.3)
   - Mostly random noise
   - Basic structural patterns emerging

2. **Middle Stage** (Loss ~0.2-0.15)
   - More defined shapes
   - Digit-like structures appearing

3. **Later Stage** (Loss ~0.1-0.05)
   - Clear digit formations
   - Consistent positioning and scaling

4. **Final Stage** (Loss < 0.05)
   - Sharp, well-defined digits
   - Clear class conditioning

## Monitoring Progress
- Loss values printed every 100 batches
- Sample grid generated after each epoch
- Checkpoints saved every 10 epochs
- Training can be resumed from latest checkpoint

## Implementation Details
- **Data Scaling**: Images scaled to [-1, 1] range
- **Noise Schedule**: Linear beta schedule for stability
- **Gradient Clipping**: Max norm of 1.0
- **Batch Processing**: Individual image processing within batches
- **Model Size**: ~1.5M parameters

## Tips for Best Results
1. Train for at least 200 epochs
2. Monitor both loss values and visual samples
3. Use GPU for faster training
4. Keep default hyperparameters for stability
5. Save periodic checkpoints for comparison

## Troubleshooting
Common issues and solutions:
1. **CUDA Out of Memory**
   - Reduce batch size
   - Use fewer timesteps

2. **Slow Training**
   - Enable GPU acceleration
   - Reduce number of workers in DataLoader

3. **Poor Generation Quality**
   - Train for more epochs
   - Check input data scaling
   - Verify noise prediction ranges

## License
[MIT License](LICENSE)

## Acknowledgments
- Based on the DDPM paper by Ho et al.
- MNIST dataset from LeCun et al.
- Implementation inspired by various diffusion model repositories

## Author
Jonathan M. Rothberg

## Contributing
Contributions welcome! Please feel free to submit a Pull Request.
