# mnistdiffuser_individual_Jan_20_25.py
# Jonathan M. Rothberg
# January 20, 2025  
# Larger model with skip connections
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
import tempfile
import shutil

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


CIFAR10 classes:
['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
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
    - Image: [batch_size, channels, height, width]  (noisy images)
    - Label: [batch_size] (integer labels 0-9)
    """
    def __init__(self, num_classes=10, emb_dim=16, timesteps=1000, in_channels=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.timesteps = timesteps
        
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        self.time_embedding = nn.Embedding(timesteps, emb_dim)
        
        # Wider initial features (128 instead of 64)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels + emb_dim + emb_dim, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),  # GroupNorm is more stable than BatchNorm
            nn.SiLU(),  # SiLU (Swish) activation often works better than ReLU
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        
        # Wider channels throughout
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        # Deeper middle block
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
        )
        
        # Matching wider channels in up path
        self.up1_conv = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        self.up2_conv = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        
        # Better final processing
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c):
        """
        Forward pass through the network.
        
        Args:
            x:      [batch_size, channels, height, width]  (noisy images)
            t:      [batch_size] (timesteps)
            c:      [batch_size] (digit labels 0-9)
        """
        t_emb = self.time_embedding(t)
        c_emb = self.class_embedding(c)
        
        # Reshape embeddings to match image dimensions
        t_emb = t_emb.view(-1, self.emb_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        c_emb = c_emb.view(-1, self.emb_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenate along channel dimension
        x = torch.cat([x, t_emb, c_emb], dim=1)  # This will have in_channels + emb_dim + emb_dim channels
        
        # Down path with residual connections
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Middle with residual connection
        identity = x3
        x3 = self.middle(x3)
        x3 = x3 + identity  # Residual connection
        
        # Up path with skip connections
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1_conv(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2_conv(x)
        
        x = self.final(x)
        
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
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Add name_suffix property for use in folder names
        self.name_suffix = f"ts{timesteps}_bs{beta_start:.0e}_be{beta_end:.0e}"
        
        # Linear or cosine beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).cuda()
        elif schedule_type == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps).cuda()
        else:
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Choose 'linear' or 'cosine'.")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

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
        Generate new images using the existing DDPM sampling with optional improvements.
        """
        model.eval()
        with torch.no_grad():
            # Get input shape from model's first conv layer, use dataset image size
            in_channels = model.init_conv[0].weight.shape[1] - 32  # Subtract 32 for embeddings
            image_size = 32 if in_channels == 3 else 28  # CIFAR=32x32, MNIST=28x28
            
            # Start with pure noise of correct size
            x = torch.randn(n_samples, in_channels, image_size, image_size).to(device)
            labels = torch.full((n_samples,), label, dtype=torch.long).to(device)
            
            # Gradually denoise with smaller steps
            for i in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), i, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x, t, labels)
                
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

def save_samples(model, diffusion, device, epoch, avg_loss, dataset_name, batch_idx=None):
    """
    Saves and displays a grid of all classes (0-9)
    """
    sample_dir = f'samples_{dataset_name}_{diffusion.schedule_type}_{diffusion.name_suffix}'  # Added dataset_name
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generate all digits for the grid
    print("\nGenerating all digits...")
    samples = []
    for i in range(10):
        # Generate one sample at a time with same model parameters
        sample = diffusion.sample(model, device, i, n_samples=1)
        samples.append(sample)
    
    # Create and save the grid
    samples = torch.cat(samples, dim=0)
    grid = utils.make_grid(samples, nrow=5, normalize=True)
    
    # Save the grid
    filename = f'{sample_dir}/epoch_{epoch}_loss_{avg_loss:.4f}.png' if batch_idx is None else f'{sample_dir}/epoch_{epoch}_batch_{batch_idx}_loss_{avg_loss:.4f}.png'
    utils.save_image(grid, filename)
    print(f"\nSaved grid of all digits to {filename}")
    
    # Display the grid (non-blocking)
    #plt.figure(figsize=(10, 4))
    #plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    #plt.axis('off')
    #plt.title(f'Generated Digits (Epoch {epoch})')
    #plt.draw()
    #plt.pause(0.1)  # Short pause to render
    #plt.close('all')  # Ensure all figures are closed

def train(model, diffusion, dataloader, optimizer, device, num_epochs, dataset_name, override_lr=None):
    """
    Training loop with corrected noise prediction and learning rate override option
    """
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    # Use a higher learning rate
    if override_lr is None:
        override_lr = 5e-3  # Increased from 2e-3 to 5e-3
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=override_lr,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.4,  # Longer warmup
        div_factor=1.0,
        final_div_factor=5,  # Keep LR higher at end
        three_phase=True
    )
    
    # Set initial learning rate explicitly
    for param_group in optimizer.param_groups:
        param_group['lr'] = override_lr
    
    # Rest of the existing train function remains exactly the same
    start_epoch = 0
    
    # Define checkpoint paths
    main_checkpoint = f'diffusion_checkpoint_{dataset_name}_ts{diffusion.timesteps}.pt'
    checkpoint_dir = f'checkpoints_{dataset_name}_{diffusion.name_suffix}'
    
    # Look for checkpoints in this order:
    # 1. Main checkpoint file with matching dataset
    # 2. Any matching checkpoint in root directory with matching dataset
    # 3. Most recent checkpoint in checkpoint directory with matching dataset
    checkpoint_path = None
    
    if os.path.exists(main_checkpoint):
        checkpoint = torch.load(main_checkpoint)
        if checkpoint.get('dataset_name') == dataset_name:  # Check dataset matches
            checkpoint_path = main_checkpoint
            print(f"Found main checkpoint: {main_checkpoint}")
    else:
        # Look for any checkpoint files in root directory
        root_checkpoints = [f for f in os.listdir('.') if f.startswith('diffusion_checkpoint_ts')]
        if root_checkpoints:
            checkpoint_path = root_checkpoints[0]
            print(f"Found checkpoint in root directory: {checkpoint_path}")
        # If no root checkpoints, look in checkpoint directory
        elif os.path.exists(checkpoint_dir):
            dir_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith('diffusion_checkpoint_ts')]
            if dir_checkpoints:
                # Sort by epoch number to get most recent
                dir_checkpoints.sort(key=lambda x: int(x.split('epoch_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, dir_checkpoints[-1])
                print(f"Found checkpoint in directory: {checkpoint_path}")
    
    # Load checkpoint if found
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Only load if timesteps match
        if checkpoint.get('timesteps') == diffusion.timesteps:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Only load optimizer/scheduler state if not overriding learning rate
            if override_lr is None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print(f"Overriding learning rate to {override_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = override_lr
                
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Timesteps don't match (checkpoint: {checkpoint.get('timesteps')}, "
                  f"current: {diffusion.timesteps}). Starting fresh training.")
    else:
        print("No checkpoint found, starting fresh training.")
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = f'checkpoints_{dataset_name}_{diffusion.name_suffix}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add this debug code at the start
    for images, _ in dataloader:
        print("Input image range:", images.min().item(), images.max().item())
        break
    
    # Modify loop to use start_epoch
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Sample random timesteps for the entire batch
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)

            # Get noisy version of the entire batch
            noisy_images, target_noise = diffusion.add_noise(images, t)

            # Predict noise for the entire batch
            predicted_noise = model(noisy_images, t, labels)

            # Calculate loss for the entire batch at once
            loss = nn.MSELoss(reduction='mean')(predicted_noise, target_noise) # No loop needed

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}')
                
                #if batch_idx % 500 == 0:
                #    save_samples(model, diffusion, device, epoch, batch_idx)
            
            # Inside training loop, add this debug print
            if batch_idx == 0 and epoch == 0:
                print("Predicted noise range:", predicted_noise.min().item(), predicted_noise.max().item())
                print("Target noise range:", target_noise.min().item(), target_noise.max().item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        # Save latest checkpoint with diffusion parameters
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'timesteps': diffusion.timesteps,
            'schedule_type': diffusion.schedule_type,
            'dataset_name': dataset_name  # Add dataset name to checkpoint
        }
        
        # Save main checkpoint - FIXED to include timesteps in filename
        torch.save(checkpoint, f'diffusion_checkpoint_{dataset_name}_ts{diffusion.timesteps}.pt')
        print(f"Saved latest checkpoint at epoch {epoch}")
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = f'{checkpoint_dir}/diffusion_checkpoint_ts{diffusion.timesteps}_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved periodic checkpoint at epoch {epoch}")
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        # Only save samples once per epoch, at the end
        if epoch % 1 == 0:  # Every epoch
            save_samples(model, diffusion, device, epoch, avg_loss, dataset_name)

def inference_mode(model_path, device, dataset_name):
    """
    Interactive mode where you can input digits and see generated images.
    """
    # Look for any checkpoint file matching the dataset
    checkpoint_files = [f for f in os.listdir('.') if f.startswith(f'diffusion_checkpoint_{dataset_name}_ts')]
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found for {dataset_name}!")
        print("Please train the model first.")
        return
    
    # If multiple files exist, let user choose
    if len(checkpoint_files) > 1:
        print("\nAvailable checkpoints:")
        for idx, file in enumerate(checkpoint_files, 1):
            print(f"{idx}. {file}")
        while True:
            choice = input("\nSelect checkpoint number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(checkpoint_files):
                checkpoint_file = checkpoint_files[int(choice) - 1]
                break
            print("Invalid choice. Please try again.")
    else:
        checkpoint_file = checkpoint_files[0]
    
    print(f"\nLoading checkpoint: {checkpoint_file}")
    
    # Load checkpoint first to get parameters
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Create model with matching parameters from checkpoint
    in_channels = 3 if dataset_name == 'cifar10' else 1
    model = ConditionalUNet(
        timesteps=checkpoint['timesteps'],  # Use timesteps from checkpoint
        in_channels=in_channels
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion model with matching parameters
    diffusion = DiffusionModel(
        timesteps=checkpoint['timesteps'],
        schedule_type=checkpoint.get('schedule_type', 'linear')  # Default to linear if not found
    )
    
    print("\nDiffusion Model Inference Mode")
    print("------------------------------")
    print("- Enter a digit (0-9) to generate an image")
    print("- Enter -1 to quit")
    print("- Generated images will be saved in 'inference_samples' directory")
    
    os.makedirs('inference_samples', exist_ok=True)
    
    while True:
        try:
            if dataset_name == 'cifar10':
                print("\nCIFAR10 classes:")
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
                for i, name in enumerate(classes):
                    print(f"{i}. {name}")
                digit = int(input("\nEnter class number (0-9) or -1 to quit: "))
            else:
                digit = int(input("\nEnter digit (0-9) or -1 to quit: "))
                
            if digit == -1:
                print("Goodbye!")
                break
            if digit < 0 or digit > 9:
                print("Please enter a number between 0 and 9")
                continue
            
            class_name = classes[digit] if dataset_name == 'cifar10' else str(digit)
            print(f"\nGenerating {class_name}...")
            
            # Generate image
            sample = diffusion.sample(model, device, digit)
            
            # Save image with appropriate name
            timestamp = torch.rand(1).item()
            filename = f'inference_samples/{class_name}_{timestamp:.3f}.png'
            utils.save_image(sample, filename, normalize=True)
            print(f"Saved to {filename}")
            
            # Display image
            plt.figure(figsize=(3, 3))
            img = sample.squeeze().cpu()
            if dataset_name == 'cifar10':
                img = img.permute(1, 2, 0)
            plt.imshow(img, cmap=None if dataset_name == 'cifar10' else 'gray')
            plt.axis('off')
            plt.title(f'Generated {class_name}')
            plt.show()
            #plt.pause(10)
            plt.close()
            
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 9")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def get_input_with_default(prompt, default, convert_type=float):
    """Helper function to get input with default value"""
    response = input(f"{prompt} (default={default}): ").strip()
    if response == "":
        return default
    try:
        return convert_type(response)
    except ValueError:
        if convert_type == str: # Handle string type directly
            return response if response else default
        print(f"Invalid input, using default: {default}")
        return default

def main():
    # Clear temp files at start
    temp_dir = tempfile.gettempdir()
    try:
        shutil.rmtree(os.path.join(temp_dir, 'torch_extensions'), ignore_errors=True)
    except:
        pass
    
    # Set device and random seed right at the start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    print(f"Using device: {device}")
    
    try:
        # Set proper defaults in argument parser
        parser = argparse.ArgumentParser(description='MNIST Diffusion Model')
        parser.add_argument('--mode', type=str, default=None, choices=['train', 'inference'])
        parser.add_argument('--model_path', type=str, default='conditional_diffusion_mnist.pth')
        parser.add_argument('--timesteps', type=int, default=None)
        parser.add_argument('--beta_start', type=float, default=None)
        parser.add_argument('--beta_end', type=float, default=None)
        parser.add_argument('--epochs', type=int, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--noise_scale', type=float, default=None)
        parser.add_argument('--use_noise_scaling', action='store_true')
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--schedule_type', type=str, default=None, choices=['linear', 'cosine'])
        args = parser.parse_args()
        
        # Add dataset configuration right after device setup
        DATASETS = {
            'mnist': {
                'class': datasets.MNIST,
                'in_channels': 1,
                'image_size': 28,
                'normalize': ([0.5], [0.5]),
            },
            'cifar10': {
                'class': datasets.CIFAR10,
                'in_channels': 3,
                'image_size': 32,
                'normalize': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            }
        }

        if args.mode is None:
            print("\nDiffusion Model Training")
            print("1. Train model")
            print("2. Run inference")
            while True:
                mode_choice = input("\nEnter your choice (1 or 2): ").strip()
                if mode_choice in ['1', '2']:
                    args.mode = 'train' if mode_choice == '1' else 'inference'
                    break
                print("Please enter 1 or 2")

        # Add dataset selection
        print("\nSelect dataset:")
        for idx, (name, info) in enumerate(DATASETS.items(), 1):
            print(f"{idx}. {name.upper()} ({info['image_size']}x{info['image_size']}, {info['in_channels']} channels)")
        
        while True:
            dataset_choice = input("\nEnter your choice (1-{}): ".format(len(DATASETS))).strip()
            if dataset_choice.isdigit() and 1 <= int(dataset_choice) <= len(DATASETS):
                dataset_name = list(DATASETS.keys())[int(dataset_choice) - 1]
                dataset_config = DATASETS[dataset_name]
                break
            print(f"Please enter a number between 1 and {len(DATASETS)}")

        if args.mode == 'train':
            print("\n=== MNIST Diffusion Model Parameter Guide ===")
            print("\nCore Parameters:")
            print(f"• Timesteps: {args.timesteps}")
            print("  - Higher values (1000+) = smoother transitions but slower training")
            print("  - Lower values (500-) = faster training but potentially lower quality")
            print("  - Recommended: 1000 for quality, 500 for quick experiments")
            
            print(f"\n• Beta Schedule (start={args.beta_start}, end={args.beta_end})")
            print("  - Controls noise addition rate")
            print("  - Lower beta_start (1e-5) = clearer early steps")
            print("  - Higher beta_end (0.02-0.05) = more diverse samples")
            print("  - Current values optimal for MNIST, adjust for more complex datasets")
            
            print(f"\n• Batch Size: {args.batch_size}")
            print("  - Larger = faster training but more memory")
            print("  - Smaller = more stable but slower")
            print("  - Adjust based on your GPU memory")
            
            print(f"\n• Learning Rate: {args.learning_rate}")
            print("  - Higher (1e-3) = faster learning but potential instability")
            print("  - Lower (1e-5) = more stable but slower progress")
            print("  - Current default good for MNIST, reduce for complex datasets")
            
            print("\nAdvanced Parameters:")
            print(f"• Noise Scale: {args.noise_scale}")
            print("  - Higher (1.2+) = more diverse but potentially noisier samples")
            print("  - Lower (0.8-) = cleaner but less diverse samples")
            print("  - Adjust if samples are too noisy or too similar")
            
            print(f"\n• Noise Scaling: {'Enabled' if args.use_noise_scaling else 'Disabled'}")
            print("  - Enable for gradual noise reduction during sampling")
            print("  - Can help with stability but might reduce diversity")
            print("  - Experiment with this for different datasets")

            # Added Noise Schedule explanation
            schedule_type_display = args.schedule_type if args.schedule_type else 'Cosine (default)'
            print(f"\n• Noise Schedule: {schedule_type_display}")
            print("  - Cosine (default): Gradual noise addition, often better quality")
            print("  - Linear: Uniform noise addition, simpler but can be less stable")
            print("  - Cosine recommended for most cases, use Linear for experiments")
            
            print("\nFor Other Datasets:")
            print("• Complex images (faces, natural scenes):")
            print("  - Increase timesteps (2000+)")
            print("  - Reduce learning rate (5e-5)")
            print("  - Consider larger model capacity")
            
            print("\n• Larger images:")
            print("  - Reduce batch size")
            print("  - Increase model capacity")
            print("  - Consider progressive training")
            
            print("\nMonitoring Tips:")
            print("• Watch for:")
            print("  - Loss not decreasing = try lower learning rate")
            print("  - Blurry samples = increase timesteps or model capacity")
            print("  - Training collapse = reduce learning rate or increase batch size")
            
            user_continue = input("\nPress Enter to continue with these settings, or 'q' to quit: ")
            if user_continue.lower() == 'q':
                print("Exiting...")
                return

            print("\nEnter parameters (press return to use defaults):")
            
            # When getting interactive input, use the same standard defaults
            if args.timesteps is None:
                args.timesteps = get_input_with_default("Timesteps", 500, int)
            if args.beta_start is None:
                args.beta_start = get_input_with_default("Beta start", 5e-5)
            if args.beta_end is None:
                args.beta_end = get_input_with_default("Beta end", 0.01)
            if args.noise_scale is None:
                args.noise_scale = get_input_with_default("Noise scale", 0.8)
            if args.schedule_type is None:
                args.schedule_type = get_input_with_default("Noise Schedule (linear/cosine)", 'linear', str)
            if args.epochs is None:
                args.epochs = get_input_with_default("Epochs", 500, int)
            if args.batch_size is None:
                args.batch_size = get_input_with_default("Batch size", 128, int)
            if args.learning_rate is None:
                args.learning_rate = get_input_with_default("Learning rate", 5e-3)

            print(f"\nUsing parameters:")
            print(f"Timesteps: {args.timesteps}")
            print(f"Beta start: {args.beta_start}")
            print(f"Beta end: {args.beta_end}")
            print(f"Epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Noise Schedule: {args.schedule_type}")
            print(f"Learning rate: {args.learning_rate}")

            # Modify dataset loading
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*dataset_config['normalize'])
            ])
            
            train_dataset = dataset_config['class'](
                root='./data',
                train=True,
                transform=transform,
                download=True
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=32 if torch.cuda.is_available() else 0
            )

            # Create diffusion model instance
            diffusion = DiffusionModel(
                timesteps=args.timesteps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                schedule_type=args.schedule_type
            )

            # Create model with correct channels
            model = ConditionalUNet(
                num_classes=10,
                emb_dim=16,
                timesteps=args.timesteps,
                in_channels=dataset_config['in_channels']
            ).to(device)

            # Update checkpoint and model paths to include dataset name
            checkpoint_file = f'diffusion_checkpoint_{dataset_name}_ts{args.timesteps}.pt'
            args.model_path = f'conditional_diffusion_{dataset_name}.pth'

            print("\nStarting Training Mode")
            print("---------------------")
            
            print(f"Training on {len(train_dataset)} images")
            print(f"Batch size: {args.batch_size}")
            print(f"Epochs: {args.epochs}")
            
            # Modified optimizer settings
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=1e-6,
                betas=(0.9, 0.999)
            )
            
            # Add override learning rate option when loading from checkpoint
            override_lr = None
            if os.path.exists(checkpoint_file):
                print("\nFound existing checkpoint.")
                override_choice = input("Would you like to override the learning rate? (y/n): ").lower()
                if override_choice == 'y':
                    override_lr = get_input_with_default("Enter new learning rate", args.learning_rate)
                    print(f"Will override learning rate to: {override_lr}")
                else:
                    print("Will continue with checkpoint's original learning rate")

            # Train the model
            train(model, diffusion, train_loader, optimizer, device, args.epochs, dataset_name, override_lr)
            
            # Save final model
            torch.save(model.state_dict(), args.model_path)
            print(f"\nTraining complete! Model saved to {args.model_path}")
            
        else:  # inference mode
            inference_mode(args.model_path, device, dataset_name)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()