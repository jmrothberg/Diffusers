# mnistdiffuser_individual_Jan_20_25.py
# Jonathan M. Rothberg
# January 20, 2025  
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import argparse

"""
MNIST Diffusion Model - Training and Inference

This script implements a conditional diffusion model for MNIST digits.
You can:
1. Train the model from scratch (--mode train)
2. Generate digits using a trained model (--mode inference)

Requirements:
pip install torch torchvision matplotlib pillow

Usage:
1. Training:   python script.py --mode train
2. Inference:  python script.py --mode inference

During training, you'll see:
- Loss values every 100 batches
- Sample images every 500 batches
- Full sample grid after each epoch
"""

class ConditionalUNet(nn.Module):
    """
    U-Net model that learns to predict and remove noise from images.
    
    Key Concepts:
    - Takes both noisy image and class label as input
    - Class label (0-9) is embedded into a vector
    - This embedding helps guide the denoising process
    - Uses a U-shaped architecture (encoder-decoder)
    
    Input Shapes:
    - Image: [batch_size, 1, 28, 28] (1 channel for grayscale)
    - Label: [batch_size] (integer labels 0-9)
    """
    def __init__(self, num_classes=10, emb_dim=10):
        super(ConditionalUNet, self).__init__()
        self.num_classes = num_classes  # 10 digits (0-9)
        self.emb_dim = emb_dim  # Size of label embedding
        
        # Convert digit labels (0-9) into learnable embeddings
        # This is like word embeddings in NLP, but for digits
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        
        # Add time embedding
        self.time_embedding = nn.Embedding(1000, emb_dim)  # 1000 timesteps
        
        # First layer combines image (1 channel) with embedded label and time (emb_dim channels each)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1 + emb_dim + emb_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Normalizes activations for stable training
            nn.ReLU()  # Non-linear activation
        )
        
        # Downsampling path: Reduce spatial dimensions, increase channels
        # This captures increasingly abstract features
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Halves the spatial dimensions
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Upsampling path: Increase spatial dimensions back to original size
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Doubles spatial dimensions
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Final layer to produce denoised image
        self.out_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # No Tanh activation - we want raw noise predictions

    def forward(self, x, labels, t):
        """
        Forward pass through the network.
        
        Args:
            x: Batch of noisy images [batch_size, 1, 28, 28]
            labels: Corresponding digit labels [batch_size]
            t: Timestep for time embedding [batch_size]
            
        Process:
        1. Convert labels to embeddings
        2. Convert timesteps to embeddings
        3. Combine embeddings with image
        4. Pass through U-Net architecture
        5. Output denoised image prediction
        """
        batch_size = x.size(0)
        
        # Convert labels to embeddings
        label_emb = self.class_embedding(labels).view(batch_size, self.emb_dim, 1, 1)
        label_emb = label_emb.expand(batch_size, self.emb_dim, x.size(2), x.size(3))
        
        # Convert timesteps to embeddings
        time_emb = self.time_embedding(t).view(batch_size, self.emb_dim, 1, 1)
        time_emb = time_emb.expand(batch_size, self.emb_dim, x.size(2), x.size(3))
        
        # Concatenate image with label and time embeddings
        x = torch.cat([x, label_emb, time_emb], dim=1)
        
        # Encoder path (downsampling)
        x1 = self.init_conv(x)         # [batch_size, 64, 28, 28]
        x2 = self.down1(x1)            # [batch_size, 128, 14, 14]
        x3 = self.down2(x2)            # [batch_size, 256, 7, 7]
        
        # Decoder path (upsampling)
        x = self.up1(x3)               # [batch_size, 128, 14, 14]
        x = self.up2(x)                # [batch_size, 64, 28, 28]
        x = self.out_conv(x)           # [batch_size, 1, 28, 28]
        return x

class DiffusionModel:
    """
    Implements the diffusion process: gradually adding and removing noise from images.
    
    Key Concepts:
    1. Forward Process (adding noise):
       - Gradually adds noise to images over T timesteps
       - Amount of noise at each step is controlled by beta schedule
    
    2. Reverse Process (removing noise):
       - Learns to gradually denoise images
       - Uses model to predict noise at each step
       
    The beta schedule controls how quickly noise is added:
    - beta_start: Small value (0.0001) for minimal initial noise
    - beta_end: Larger value (0.02) for more noise in later steps
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear beta schedule instead of cosine for better stability
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, x, t):
        """
        Add noise to images at specified timesteps.
        """
        noise = torch.randn_like(x)
        
        # Ensure proper broadcasting of alpha values
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        noisy_images = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha * noise
        return noisy_images, noise

    def sample(self, model, device, label, n_samples=1):
        """
        Generate new images using corrected DDPM sampling.
        """
        model.eval()
        with torch.no_grad():
            # Start with pure noise
            x = torch.randn(n_samples, 1, 28, 28).to(device)
            labels = torch.full((n_samples,), label, dtype=torch.long).to(device)
            
            # Gradually denoise
            for i in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), i, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x, labels, t)
                
                # Calculate denoising step parameters
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                # Calculate mean for posterior q(x_{t-1} | x_t, x_0)
                x_0_pred = (x - torch.sqrt(1. - alpha_cumprod) * predicted_noise) / \
                          torch.sqrt(alpha_cumprod)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)
                
                mean = (beta * x_0_pred + (1. - beta) * x) / torch.sqrt(alpha)
                
                # Only add noise for non-final steps
                if i > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta) * noise
                else:
                    x = mean
            
            # Scale from [-1,1] to [0,1]
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)
            return x

def save_samples(model, diffusion, device, epoch, batch_idx=None):
    """
    Saves and displays a grid of all digits (0-9)
    """
    os.makedirs('samples', exist_ok=True)
    
    # Generate all digits for the grid
    print("\nGenerating all digits...")
    samples = []
    for i in range(10):
        # Generate one sample at a time
        sample = diffusion.sample(model, device, i, n_samples=1)
        samples.append(sample)
    
    # Create and save the grid
    samples = torch.cat(samples, dim=0)
    grid = utils.make_grid(samples, nrow=5, normalize=True)
    
    # Save the grid
    filename = f'samples/epoch_{epoch}_batch_{batch_idx}.png' if batch_idx else f'samples/epoch_{epoch}.png'
    utils.save_image(grid, filename)
    print(f"\nSaved grid of all digits to {filename}")
    
    # Display the grid (non-blocking)
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f'Generated Digits (Epoch {epoch})')
    plt.draw()
    plt.pause(0.1)  # Short pause to render
    plt.close('all')  # Ensure all figures are closed

def train(model, diffusion, dataloader, optimizer, device, num_epochs):
    """
    Training loop with corrected noise prediction
    """
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Modify scheduler for higher learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * 2,  # Slower decay
        eta_min=3e-4  # Higher minimum, was 1e-4
    )
    
    # Initialize start_epoch
    start_epoch = 0
    
    # Load checkpoint if exists
    if os.path.exists('diffusion_checkpoint.pt'):
        print("Loading checkpoint...")
        checkpoint = torch.load('diffusion_checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Handle both old and new checkpoint formats
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Restored scheduler state from checkpoint")
        else:
            print("Old checkpoint format detected - scheduler initialized fresh")
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Modify loop to use start_epoch
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device) * 2 - 1  # Scale to [-1, 1]
            labels = labels.to(device)
            
            # Process each image individually but in parallel
            loss = 0
            for i in range(images.shape[0]):
                # Sample random timestep for this specific image
                t = torch.randint(0, diffusion.timesteps, (1,), device=device)
                
                # Get noisy version of this image
                noisy_image, target_noise = diffusion.add_noise(images[i:i+1], t)
                
                # Predict noise (not the denoised image)
                predicted_noise = model(noisy_image, labels[i:i+1], t)
                
                # Compare predicted noise with target noise
                loss += nn.MSELoss()(predicted_noise, target_noise)
            
            # Average loss over batch
            loss = loss / images.shape[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}')
                
                #if batch_idx % 500 == 0:
                #    save_samples(model, diffusion, device, epoch, batch_idx)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }
        
        # Always save latest checkpoint
        torch.save(checkpoint, 'diffusion_checkpoint.pt')
        
        # Also save as the final model file
        torch.save(model.state_dict(), 'conditional_diffusion_mnist.pth')
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_checkpoint_path = f'checkpoints/diffusion_checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_checkpoint_path)
            print(f"Saved periodic checkpoint at epoch {epoch}")
        else:
            print(f"Saved latest checkpoint at epoch {epoch}")
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        # Only save samples once per epoch, at the end
        if epoch % 1 == 0:  # Every epoch
            save_samples(model, diffusion, device, epoch)

def inference_mode(model_path, device):
    """
    Interactive mode where you can input digits and see generated images.
    
    Usage:
    1. Enter a digit (0-9)
    2. See the generated image
    3. Image is saved to 'inference_samples' directory
    4. Enter -1 to quit
    """
    # Load model
    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    diffusion = DiffusionModel()
    
    print("\nDiffusion Model Inference Mode")
    print("------------------------------")
    print("- Enter a digit (0-9) to generate an image")
    print("- Enter -1 to quit")
    print("- Generated images will be saved in 'inference_samples' directory")
    
    os.makedirs('inference_samples', exist_ok=True)
    
    while True:
        try:
            # Get digit from user
            digit = int(input("\nEnter digit (0-9) or -1 to quit: "))
            if digit == -1:
                print("Goodbye!")
                break
            if digit < 0 or digit > 9:
                print("Please enter a digit between 0 and 9")
                continue
            
            print(f"Generating digit {digit}...")
            
            # Generate image
            sample = diffusion.sample(model, device, digit)
            
            # Save image
            timestamp = torch.rand(1).item()
            filename = f'inference_samples/digit_{digit}_{timestamp:.3f}.png'
            utils.save_image(sample, filename, normalize=True)
            print(f"Saved to {filename}")
            
            # Display image
            plt.figure(figsize=(3, 3))
            plt.imshow(sample.squeeze().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f'Generated Digit: {digit}')
            plt.show()
            plt.close()
            
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 9")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def main():
    try:
        # Try command line arguments first
        parser = argparse.ArgumentParser(description='MNIST Diffusion Model')
        parser.add_argument('--mode', type=str, default=None, choices=['train', 'inference'],
                          help='train the model or run inference')
        parser.add_argument('--model_path', type=str, default='conditional_diffusion_mnist.pth',
                          help='path to save/load model')
        parser.add_argument('--epochs', type=int, default=None,
                          help='number of training epochs')
        parser.add_argument('--batch_size', type=int, default=256,
                          help='training batch size')
        args = parser.parse_args()
        
        # If mode not specified via command line, ask for input
        if args.mode is None:
            print("\nMNIST Diffusion Model")
            print("1. Train model")
            print("2. Run inference")
            while True:
                mode_choice = input("\nEnter your choice (1 or 2): ").strip()
                if mode_choice in ['1', '2']:
                    args.mode = 'train' if mode_choice == '1' else 'inference'
                    break
                print("Please enter 1 or 2")
        
        # If training and epochs not specified, ask for input
        if args.mode == 'train' and args.epochs is None:
            while True:
                try:
                    epochs = input("\nEnter number of epochs to train (e.g., 50): ").strip()
                    args.epochs = int(epochs)
                    if args.epochs > 0:
                        break
                    print("Please enter a positive number")
                except:
                    print("Invalid input. Please enter a number")
    
        # Set device and random seed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        print(f"Using device: {device}")
        
        if args.mode == 'train':
            print("\nStarting Training Mode")
            print("---------------------")
            
            # Initialize model and diffusion
            model = ConditionalUNet().to(device)
            # Reduce timesteps for faster training and better stability
            diffusion = DiffusionModel(timesteps=500)  # Changed from 1000
            
            # Prepare dataset with proper scaling
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            train_dataset = datasets.MNIST(
                root='./data', 
                train=True, 
                transform=transform,
                download=True
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=128,  # Reduced from 256 for better stability
                shuffle=True,
                num_workers=4 if torch.cuda.is_available() else 0
            )
            
            print(f"Training on {len(train_dataset)} images")
            print(f"Batch size: {args.batch_size}")
            print(f"Epochs: {args.epochs}")
            
            # Modified optimizer settings
            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,  # Reduced from 3e-4
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            
            # Modify the train call to start from the loaded epoch
            train(model, diffusion, train_loader, optimizer, device, args.epochs)
            
            # Save final model
            torch.save(model.state_dict(), args.model_path)
            print(f"\nTraining complete! Model saved to {args.model_path}")
            
        else:  # inference mode
            if not os.path.exists(args.model_path):
                print(f"Error: Model file {args.model_path} not found!")
                print("Please train the model first or provide a valid model path.")
                return
            
            inference_mode(args.model_path, device)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()