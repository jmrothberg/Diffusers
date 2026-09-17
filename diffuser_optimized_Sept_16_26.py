# diffuser_optimized_Sept_16_26.py   <- THIS IS THE SCRIPT TO RUN (MNIST and CIFAR-10)
# Jonathan M. Rothberg
# January 20, 2025  
# September 16-17, 2026: renamed from diffuser_optimized_Oct_25_25.py. Fixed DDPM sampler (posterior
#   mean/variance), corrected CIFAR beta schedule, EMA weights, horizontal flips for CIFAR,
#   bf16 + torch.compile speed path, labeled 10x5 sample grids. See README.
# Larger model with skip connections
# Embedding dimension variable
# Expanded attention layer
# February 14, 2025 new features extra self-attention, capacity multiplier
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
import sys
import copy  # EMA (Sep 2026): deepcopy of the model for the exponential moving average

"""
October 25, 2025 JMR
## ONLY single GPU is working properly, just could not get the multi-GPU working without NAN and slow.
MNIST and CIFAR-10 Diffusion Model – Training and Inference

This script implements a conditional diffusion model that can learn to generate:
1. MNIST digits (0–9)
2. CIFAR-10 objects (
3. CIFAR-10 optimized (smaller model for faster training)

)

You can:
• Train the model from scratch (--mode train)
• Generate images using a trained model (--mode inference)

Requirements:
pip install torch torchvision matplotlib pillow

Usage:
1. Training:   python script.py --mode train
2. Inference:  python script.py --mode inference

During training, the script provides:
• Loss values every 100 batches
• Sample images every 500 batches
• A full grid of samples at the end of each epoch


CIFAR10 classes:
['airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck']
"""

class ConditionalUNet(nn.Module):
    """
    U-Net model that learns to predict and remove noise from images.
    
    Key Concepts:
    - Takes both a noisy image and a class label (MNIST: digits 0–9, CIFAR-10: classes 0–9) as inputs.
    - Class label is embedded into a vector (embedding dimension specified by emb_dim).
    - Time steps are also embedded (via time_embedding) and appended as channels.
    - The U-Net follows an encoder-decoder (U-shaped) architecture:
      • Encoder progressively downsamples feature maps, learning higher-level abstractions.
      • Decoder upscales these feature maps, refines them, and integrates information from earlier layers via skip connections.
    - Skip connections preserve fine-grained spatial information from earlier layers, helping the model reconstruct sharper details.
    - The model is trained to predict the noise present in the input images, so it can remove that noise (denoise) in subsequent sampling steps.
    
    Input Shapes:
    - Image: [batch_size, channels, height, width]
      • For MNIST: channels = 1, height = width = 28.
      • For CIFAR-10: channels = 3, height = width = 32.
    - Label: [batch_size] (digit or class labels 0–9).
      • This label is embedded into emb_dim channels and concatenated.
    - Timestep: [batch_size] (int in range [0, timesteps-1]).
      • Embedded into emb_dim channels and concatenated.
    
    Output:
    - Predicted noise map: [batch_size, channels, height, width] matching the input image size.
    """
    def __init__(self, num_classes=10, emb_dim=32, timesteps=1000, in_channels=3, use_attention=False, use_optimized_cifar10=False, use_optimized_mnist=False):
        super().__init__()

        # Step 3: For CIFAR (color images), enforce a higher embedding dimension if too low.
        if in_channels == 3 and emb_dim < 128:
            emb_dim = 128

        # Step 2: Set capacity multiplier for CIFAR (in_channels==3) vs MNIST.
        # For optimized CIFAR-10, use smaller base channels (384→256→512 instead of 768→1536→3072)
        # For optimized MNIST, use much smaller base channels (128→192→256 instead of 512→1024→2048)
        if use_optimized_cifar10 and in_channels == 3:
            # Optimized CIFAR-10: smaller model for faster training
            capacity_mult = 1.0  # No scaling needed
            base_init = 384  # Fixed: Was 128 causing 259→128 bottleneck, now 259→384 (reasonable)
            base_down1 = 256
            base_down2 = 512
        elif use_optimized_mnist and in_channels == 1:
            # Optimized MNIST: much smaller model for faster training
            capacity_mult = 1.0  # No scaling needed
            base_init = 128  # Much smaller: 512 → 128
            base_down1 = 192  # Much smaller: 1024 → 192
            base_down2 = 256  # Much smaller: 2048 → 256
        else:
            # Original configuration
            capacity_mult = 1.5 if in_channels == 3 else 1.0
            base_init = 512
            base_down1 = 1024
            base_down2 = 2048

        # Define the new channel sizes based on the multiplier.
        init_channels   = int(base_init * capacity_mult)   # 128 or 512 originally
        down1_channels  = int(base_down1 * capacity_mult)  # 256 or 1024
        down2_channels  = int(base_down2 * capacity_mult)  # 512 or 2048
        up1_channels    = int(base_down1 * capacity_mult)  # 256 or 1024 in up path
        up2_channels    = int(base_init * capacity_mult)   # 128 or 512 in up path
        final_ch1       = int(128 * capacity_mult)   # scale the final block too
        final_ch2       = int(64 * capacity_mult)
        
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.use_attention = use_attention  # Store this as model parameter
        self.use_optimized_cifar10 = use_optimized_cifar10  # Store optimized flag
        self.use_optimized_mnist = use_optimized_mnist  # Store optimized flag
        
        # Calculate total input channels including embeddings
        total_input_channels = in_channels + (2 * emb_dim)  # Original channels + time emb + class emb
        
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        self.time_embedding = nn.Embedding(timesteps, emb_dim)
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, init_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, init_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, init_channels),
            nn.SiLU(),
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(init_channels, down1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, down1_channels),
            nn.SiLU(),
            nn.Conv2d(down1_channels, down1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, down1_channels),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(down1_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        # Determine attention heads based on optimized model settings
        attention_heads = 4 if (use_optimized_cifar10 or use_optimized_mnist) else 8

        middle_layers = []
        # First convolution block of the middle
        middle_layers.extend([
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
        ])
        if use_attention:
            middle_layers.append(SelfAttention(down2_channels, num_heads=attention_heads))
        middle_layers.extend([
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
        ])
        if use_attention:
            middle_layers.append(SelfAttention(down2_channels, num_heads=attention_heads))
        middle_layers.extend([
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
        ])
        self.middle = nn.Sequential(*middle_layers)
        
        up1_layers = [
            nn.Conv2d(down2_channels + down1_channels, up1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up1_channels),
            nn.SiLU(),
        ]
        if use_attention:
            up1_layers.append(SelfAttention(up1_channels, num_heads=attention_heads))
        up1_layers.extend([
            nn.Conv2d(up1_channels, up1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up1_channels),
            nn.SiLU(),
        ])
        self.up1_conv = nn.Sequential(*up1_layers)

        up2_layers = [
            nn.Conv2d(up1_channels + init_channels, up2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up2_channels),
            nn.SiLU(),
        ]
        if use_attention:
            up2_layers.append(SelfAttention(up2_channels, num_heads=attention_heads))
        up2_layers.extend([
            nn.Conv2d(up2_channels, up2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up2_channels),
            nn.SiLU(),
        ])
        self.up2_conv = nn.Sequential(*up2_layers)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        
        self.final = nn.Sequential(
            nn.Conv2d(up2_channels, final_ch1, kernel_size=3, padding=1),
            nn.GroupNorm(16, final_ch1),
            nn.SiLU(),
            nn.Conv2d(final_ch1, final_ch2, kernel_size=3, padding=1),
            nn.GroupNorm(16, final_ch2),
            nn.SiLU(),
            nn.Conv2d(final_ch2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor):
                Noisy images of shape [batch_size, channels, height, width].
                - For MNIST:  channels=1, height=28, width=28
                - For CIFAR:  channels=3, height=32, width=32
            t (torch.Tensor):
                Timestep integers of shape [batch_size], in the range [0, timesteps-1].
            c (torch.Tensor):
                Class labels of shape [batch_size].
                - MNIST: digits 0–9
                - CIFAR: class labels 0–9 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        """
                
        # Clamp indices to prevent out-of-bounds access (DataParallel can cause issues)
        t = torch.clamp(t, 0, self.timesteps - 1)
        c = torch.clamp(c, 0, self.num_classes - 1)

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


class SelfAttention(nn.Module):
    """
    Self-attention layer for capturing global dependencies in images.
    Significantly improves the model's ability to understand and generate coherent structures.

    Args:
        channels: Number of input channels
        num_heads: Number of attention heads (default: 8 for regular model, 4 for optimized CIFAR-10)
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, 4 * channels),  # Wider FF network
            nn.GELU(),
            nn.Linear(4 * channels, channels),  # Back to original channels
        )

    def forward(self, x):
        size = x.shape[-2:]  # Store original spatial dimensions
        
        # Reshape to sequence format and apply layer norm
        x = x.view(x.shape[0], self.channels, -1).swapaxes(1, 2)  # [B, HW, C]
        x_ln = self.ln(x)
        
        # Apply attention with residual connection
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        
        # Apply feedforward with residual
        attention_value = self.ff_self(attention_value) + attention_value
        
        # Reshape back to image format
        return attention_value.swapaxes(2, 1).view(x.shape[0], self.channels, *size)


class DiffusionModel:
    """
    Implements the diffusion process: gradually adding and removing noise from images.
    
    Key Concepts:
    1. Forward Process (adding noise):
       - Gradually adds Gaussian noise to images over T timesteps
       - Each timestep t adds a controlled amount of noise based on the beta schedule
       - By timestep T, the image becomes pure noise (no visible structure)
       
    2. Reverse Process (removing noise):
       - Starting from pure noise, gradually denoises the image
       - Uses neural network to predict and remove noise at each step
       - Network is conditioned on:
         • Current noisy image
         • Timestep t (how noisy the image is)
         • Class label (what we want to generate)
    
    Beta Schedule Controls Noise Addition:
    - beta_start (e.g., 1e-4): 
      • Very small initial noise for subtle early changes
      • Helps preserve important image structure
    
    - beta_end (e.g., 0.02):
      • Larger final noise for complete randomization
      • Ensures full coverage of image space
    
    Schedule Types:
    - Linear: Simple uniform increase from beta_start to beta_end
    - Cosine: Smoother noise schedule that often gives better results
    
    Example Process:
    1. Start with clean image
    2. Forward: Add noise gradually until image is random
    3. Training: Learn to reverse this process
    4. Inference: Generate new images by starting with pure noise
       and gradually denoising with class guidance
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear', 
                 noise_scale=1.0, use_noise_scaling=False, cosine_s=0.008, emb_dim=128):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.noise_scale = noise_scale
        self.use_noise_scaling = use_noise_scaling
        self.cosine_s = cosine_s
        self.emb_dim = emb_dim
        
        # Modify name_suffix to include embedding dimension
        self.name_suffix = f"ts{timesteps}_bs{beta_start:.0e}_be{beta_end:.0e}_emb{emb_dim}"
        
        # Linear or cosine beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).cuda()
        elif schedule_type == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps, s=self.cosine_s).cuda()
        else:
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Choose 'linear' or 'cosine'.")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Creates a cosine schedule for the noise variance (beta) over time.
        Paper: "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672)
        
        Why Cosine Schedule?
        - Provides smoother transitions between noise levels compared to linear
        - Often works better for complex images (like CIFAR-10)
        - Can help prevent early timestep collapse
        
        Parameters:
        - timesteps: Total number of denoising steps (e.g., 1000)
        - s: Offset parameter (default=0.008)
          • Controls schedule behavior at t=0
          • Larger s = more rapid initial noise addition
          • Smaller s = gentler start
        
        How it works:
        1. Creates timestep range from 0 to T
        2. Transforms this range using cosine function
        3. Computes cumulative alpha products (noise retention)
        4. Derives beta values (noise addition per step)
        
        For CIFAR-10:
        - Works well with timesteps=10001
        - May need larger s (e.g., 0.01-0.02) for more complex images
        - Consider increasing beta_end to 0.02-0.05
        
        For MNIST:
        - If failing, try:
          • Reduce timesteps (500-800)
          • Decrease s to 0.005
          • Keep beta_end lower (0.01-0.02)
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        # Transform timesteps using cosine function
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        # Normalize to ensure alpha_1 = 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Calculate beta values from alphas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clip betas to prevent numerical instability
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
        # Fix: Properly access emb_dim when model is wrapped in DataParallel
        model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
        emb_dim = model_unwrapped.emb_dim  # read from unwrapped model
        
        with torch.no_grad():
            # The only line changed: subtract 2 * emb_dim instead of a fixed "32"
            in_channels = model_unwrapped.init_conv[0].weight.shape[1] - (2 * emb_dim)
            
            # The rest of your sample code is unchanged:
            image_size = 32 if in_channels == 3 else 28
            x = torch.randn(n_samples, in_channels, image_size, image_size).to(device)
            labels = torch.full((n_samples,), label, dtype=torch.long).to(device)
            
            for i in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), i, device=device, dtype=torch.long)
                
                predicted_noise = model(x, t, labels)
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                # FIX (Sep 2026): alpha_bar_{t-1}; defined as 1 at t=0 so the final step returns x_0_pred exactly
                alpha_cumprod_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.ones_like(alpha_cumprod)
                
                x_0_pred = (x - torch.sqrt(1. - alpha_cumprod) * predicted_noise) / \
                           torch.sqrt(alpha_cumprod)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)
                
                # FIX (Sep 2026): use the true DDPM posterior mean of q(x_{t-1} | x_t, x_0) (Ho et al. 2020, Eq. 7).
                # Old line was  mean = (beta*x_0_pred + (1-beta)*x)/sqrt(alpha)  which only removes a beta-sized
                # (~1%) fraction of the noise per step instead of beta/(1-alpha_bar), so samples stayed noisy even
                # with a perfect noise predictor. MNIST hid this (+-1 pixels clamp clean); CIFAR mid-tones cannot.
                mean = (torch.sqrt(alpha_cumprod_prev) * beta / (1. - alpha_cumprod)) * x_0_pred + \
                       (torch.sqrt(alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)) * x
                # FIX (Sep 2026): matching posterior variance beta_tilde_t (was beta_t)
                posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
                
                if i > 0:
                    noise = torch.randn_like(x)
                    # Use use_noise_scaling parameter
                    if self.use_noise_scaling:
                        # Scale noise down over time
                        current_scale = self.noise_scale * (i / self.timesteps)
                        x = mean + torch.sqrt(posterior_variance) * noise * current_scale
                    else:
                        x = mean + torch.sqrt(posterior_variance) * noise * self.noise_scale
                else:
                    x = mean
            
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)
            return x

    def sample_batch(self, model, device, labels):
        """
        Generate multiple samples with different class labels in a single batch.
        Much faster than calling sample() multiple times.

        Args:
            model: The diffusion model
            device: CUDA device
            labels: List of class labels (e.g., [0,1,2,3,4,5,6,7,8,9])

        Returns:
            List of generated samples, one for each label
        """
        model.eval()
        # Fix: Properly access emb_dim when model is wrapped in DataParallel
        model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
        emb_dim = model_unwrapped.emb_dim  # read from unwrapped model

        num_samples = len(labels)

        with torch.no_grad():
            # The only line changed: subtract 2 * emb_dim instead of a fixed "32"
            in_channels = model_unwrapped.init_conv[0].weight.shape[1] - (2 * emb_dim)

            # The rest of your sample code is unchanged:
            image_size = 32 if in_channels == 3 else 28

            # Generate noise for all samples at once
            x = torch.randn(num_samples, in_channels, image_size, image_size).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

            for i in reversed(range(self.timesteps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)

                predicted_noise = model(x, t, labels_tensor)
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                # FIX (Sep 2026): alpha_bar_{t-1}; defined as 1 at t=0 (same fix as in sample())
                alpha_cumprod_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.ones_like(alpha_cumprod)

                x_0_pred = (x - torch.sqrt(1. - alpha_cumprod) * predicted_noise) / \
                           torch.sqrt(alpha_cumprod)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)

                # FIX (Sep 2026): true DDPM posterior mean q(x_{t-1}|x_t,x_0) (Ho et al. 2020, Eq. 7); see sample()
                # Old: mean = (beta * x_0_pred + (1. - beta) * x) / torch.sqrt(alpha)
                mean = (torch.sqrt(alpha_cumprod_prev) * beta / (1. - alpha_cumprod)) * x_0_pred + \
                       (torch.sqrt(alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)) * x
                # FIX (Sep 2026): matching posterior variance beta_tilde_t (was beta_t)
                posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

                if i > 0:
                    noise = torch.randn_like(x)
                    # Use use_noise_scaling parameter
                    if self.use_noise_scaling:
                        # Scale noise down over time
                        current_scale = self.noise_scale * (i / self.timesteps)
                        x = mean + torch.sqrt(posterior_variance) * noise * current_scale
                    else:
                        x = mean + torch.sqrt(posterior_variance) * noise * self.noise_scale
                else:
                    x = mean

            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)

            # Return list of individual samples
            return [x[i:i+1] for i in range(num_samples)]

def save_samples(model, diffusion, device, epoch, avg_loss, dataset_name, batch_idx=None, use_batch_inference=True,
                 samples_per_class=5):
    """
    Saves and displays a grid of all classes (0-9)

    Args:
        use_batch_inference: If True, use batch generation. If False, use sequential generation.
                           This preference is set once at the beginning of training.
        samples_per_class: GRID (Sep 2026) rows in the grid; one column per class, class name printed on top.
    """
    # GRID (Sep 2026): human-readable column headers
    CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_names = CIFAR_CLASSES if dataset_name.startswith('cifar10') else [str(i) for i in range(10)]
    # Check for DataParallel and access wrapped model properly
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model

    # Add attention to sample directory name
    attention_suffix = '_attention' if model_unwrapped.use_attention else ''
    sample_dir = f'samples_{dataset_name}_{diffusion.schedule_type}_{diffusion.name_suffix}{attention_suffix}'
    os.makedirs(sample_dir, exist_ok=True)

    # Generate all digits for the grid
    print("\nGenerating all digits...")
    samples = []

    # GRID (Sep 2026): row-major label order -> row r of the grid is the r-th sample of every class
    labels = list(range(10)) * samples_per_class

    if use_batch_inference:
        # NEW: Batch generation - generate all classes at once
        try:
            print("Using batch generation (10x faster)...")
            # Generate all 10 classes (x samples_per_class) in one batch
            batch_samples = diffusion.sample_batch(model, device, labels)
            samples = [batch_samples[i] for i in range(len(labels))]
            print("Batch generation successful!")
        except Exception as e:
            print(f"Batch generation failed ({e}), falling back to sequential...")
            use_batch_inference = False

    if not use_batch_inference:
        # ORIGINAL: Sequential generation (fallback)
        print("Using sequential generation...")
        for i in labels:
            # Generate one sample at a time with same model parameters
            sample = diffusion.sample(model, device, i, n_samples=1)
            samples.append(sample)

    # Create and save the grid
    samples = torch.cat(samples, dim=0)
    # GRID (Sep 2026): 10 columns = classes; no normalize=True any more - samples are already in [0,1] and
    # per-grid renormalisation used to hide brightness/contrast problems (it is what made the old speckled
    # MNIST grids look clean). Upscaled 3x nearest-neighbour and given a class-name header.
    grid = utils.make_grid(samples, nrow=10, padding=2, pad_value=1.0)
    img = Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
    scale = 3
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    header, footer = 22, 18
    canvas = Image.new('RGB', (img.width, img.height + header + footer), 'white')
    canvas.paste(img, (0, header))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    cell = (samples.shape[-1] + 2) * scale  # image width + padding, in canvas pixels
    for i, name in enumerate(class_names):
        draw.text((2 * scale + i * cell + 4, 5), name, fill='black')
    draw.text((4, canvas.height - footer + 3), f'epoch {epoch}   loss {avg_loss:.4f}   (EMA weights)', fill='gray')

    # Save the grid
    filename = f'{sample_dir}/epoch_{epoch}_loss_{avg_loss:.4f}.png' if batch_idx is None else f'{sample_dir}/epoch_{epoch}_batch_{batch_idx}_loss_{avg_loss:.4f}.png'
    canvas.save(filename)  # GRID (Sep 2026): was utils.save_image(grid, filename)
    method = "batch" if use_batch_inference else "sequential"
    print(f"\nSaved grid of all digits to {filename} (using {method} generation)")
    

def train(model, diffusion, dataloader, optimizer, device, num_epochs, dataset_name, override_lr=None, use_batch_inference=True):
    """
    Training loop with corrected noise prediction and learning rate override option
    """
    model.train()
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=override_lr if override_lr is not None else optimizer.param_groups[0]['lr'],
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.4,
        div_factor=1.0,
        final_div_factor=5,  # Keep LR higher at end
        three_phase=True
    )
    
    # Only set initial learning rate if explicitly overriding
    if override_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = override_lr
    
    # Rest of the existing train function remains exactly the same
    start_epoch = 0
    
    # Add attention to checkpoint directory name
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    attention_suffix = '_attention' if model_unwrapped.use_attention else ''
    checkpoint_dir = f'checkpoints_{dataset_name}_{diffusion.name_suffix}{attention_suffix}'
    checkpoint_path = None
    
    # Add attention to main checkpoint name
    main_checkpoint = f'diffusion_checkpoint_{dataset_name}_ts{diffusion.timesteps}_emb{diffusion.emb_dim}{attention_suffix}.pt'
    
    if os.path.exists(main_checkpoint):
        checkpoint = torch.load(main_checkpoint)
        if (checkpoint.get('dataset_name') == dataset_name and 
            checkpoint.get('emb_dim') == diffusion.emb_dim):
            checkpoint_path = main_checkpoint
            print(f"Found main checkpoint: {main_checkpoint}")
    else:
        # Modified pattern to include attention
        root_checkpoints = [f for f in os.listdir('.') 
                          if f.startswith(f'diffusion_checkpoint_{dataset_name}_ts')]
        if root_checkpoints:
            for ckpt_file in root_checkpoints:
                ckpt = torch.load(ckpt_file)
                if (ckpt.get('dataset_name') == dataset_name and
                    ckpt.get('emb_dim') == diffusion.emb_dim and
                    ckpt.get('use_attention', False) == model_unwrapped.use_attention and  # Check attention matches
                    ckpt.get('use_optimized_cifar10', False) == model_unwrapped.use_optimized_cifar10 and  # Check optimized flag matches
                    ckpt.get('use_optimized_mnist', False) == model_unwrapped.use_optimized_mnist):  # Check optimized flag matches
                    checkpoint_path = ckpt_file
                    print(f"Found compatible checkpoint in root directory: {ckpt_file}")
                    break
        
        elif os.path.exists(checkpoint_dir):
            dir_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith(f'diffusion_checkpoint_{dataset_name}_ts')]
            if dir_checkpoints:
                dir_checkpoints.sort(key=lambda x: int(x.split('epoch_')[-1].split('.')[0]))
                latest_ckpt = torch.load(os.path.join(checkpoint_dir, dir_checkpoints[-1]))
                if (latest_ckpt.get('dataset_name') == dataset_name and
                    latest_ckpt.get('emb_dim') == diffusion.emb_dim and
                    latest_ckpt.get('use_attention', False) == model_unwrapped.use_attention and  # Check attention matches
                    latest_ckpt.get('use_optimized_cifar10', False) == model_unwrapped.use_optimized_cifar10 and  # Check optimized flag matches
                    latest_ckpt.get('use_optimized_mnist', False) == model_unwrapped.use_optimized_mnist):  # Check optimized flag matches
                    checkpoint_path = os.path.join(checkpoint_dir, dir_checkpoints[-1])
                    print(f"Found compatible checkpoint in directory: {checkpoint_path}")
    
    # Modified checkpoint loading logic
    if checkpoint_path:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Additional verification message
        print(f"Checkpoint details:")
        print(f"- Dataset: {checkpoint.get('dataset_name')}")
        print(f"- Embedding dim: {checkpoint.get('emb_dim')}")
        print(f"- Timesteps: {checkpoint.get('timesteps')}")
        print(f"- Using attention: {checkpoint.get('use_attention', False)}")
        
        # Check compatibility
        is_compatible = (
            checkpoint.get('timesteps') == diffusion.timesteps and
            checkpoint.get('emb_dim') == model_unwrapped.emb_dim and
            checkpoint.get('use_attention', False) == model_unwrapped.use_attention and
            checkpoint.get('use_optimized_cifar10', False) == model_unwrapped.use_optimized_cifar10 and
            checkpoint.get('use_optimized_mnist', False) == model_unwrapped.use_optimized_mnist
        )

        if not is_compatible:
            print("\nWARNING: Checkpoint parameters don't match current settings:")
            print(f"Checkpoint: timesteps={checkpoint.get('timesteps')}, "
                  f"emb_dim={checkpoint.get('emb_dim')}, "
                  f"attention={checkpoint.get('use_attention', False)}, "
                  f"optimized_cifar10={checkpoint.get('use_optimized_cifar10', False)}, "
                  f"optimized_mnist={checkpoint.get('use_optimized_mnist', False)}")
            print(f"Current: timesteps={diffusion.timesteps}, "
                  f"emb_dim={model_unwrapped.emb_dim}, "
                  f"attention={model_unwrapped.use_attention}, "
                  f"optimized_cifar10={model_unwrapped.use_optimized_cifar10}, "
                  f"optimized_mnist={model_unwrapped.use_optimized_mnist}")
            print("\nCannot continue from this checkpoint due to architecture mismatch.")
            user_input = input("Start fresh training? (y/n): ")
            if user_input.lower() != 'y':
                print("Exiting...")
                sys.exit(0)
            checkpoint_path = None
        else:
            # Handle loading state dict for both DataParallel and non-DataParallel models
            state_dict = checkpoint['model_state_dict']
            
            # Check if we're loading a non-DataParallel checkpoint into a DataParallel model
            is_loading_non_parallel_to_parallel = (
                isinstance(model, nn.DataParallel) and
                not any(k.startswith('module.') for k in state_dict.keys())
            )
            
            if is_loading_non_parallel_to_parallel:
                print("Converting non-DataParallel checkpoint to DataParallel format...")
                # Create new state dict with 'module.' prefix for each key
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[f'module.{k}'] = v
                state_dict = new_state_dict
            
            # Check if we're loading a DataParallel checkpoint into a non-DataParallel model
            is_loading_parallel_to_non_parallel = (
                not isinstance(model, nn.DataParallel) and
                any(k.startswith('module.') for k in state_dict.keys())
            )
            
            if is_loading_parallel_to_non_parallel:
                print("Converting DataParallel checkpoint to non-DataParallel format...")
                # Create new state dict removing 'module.' prefix from each key
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                state_dict = new_state_dict
            
            # Now load the properly formatted state dict
            model.load_state_dict(state_dict)
            
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

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = f'checkpoints_{dataset_name}_{diffusion.name_suffix}{attention_suffix}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # EMA (Sep 2026): exponential moving average of the weights, used ONLY for sampling (Ho et al. 2020 use
    # decay 0.9999). The raw weights jump around step to step; averaging them gives much cleaner samples at
    # the same epoch. Trained weights are untouched. Decay warms up from 0 so early epochs are not stuck at
    # the initial weights: decay = min(0.9999, (1+step)/(10+step)).
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    ema_model = copy.deepcopy(model_unwrapped).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if checkpoint_path and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        print("Restored EMA weights from checkpoint")
    else:
        print("EMA initialised from current model weights")
    ema_step = start_epoch * len(dataloader)
    ema_params = list(ema_model.parameters())
    live_params = list(model_unwrapped.parameters())
    
    # Add this debug code at the start
    for images, _ in dataloader:
        print("Input image range:", images.min().item(), images.max().item())
        break
    
    # SPEED (Sep 2026): measured on GB10, batch 128, this model: fp32 eager 376 ms/step -> bf16 + torch.compile
    # 204 ms/step (~1.8x; ~3.8x vs the old cu128 PyTorch wheel). We compile a *view* of the model used only
    # for the training forward pass; `model` itself is unchanged, so state_dict keys, checkpoints,
    # DataParallel handling and save_samples() all behave exactly as before.
    torch.backends.cudnn.benchmark = True  # let cuDNN pick the fastest conv kernels for our fixed shapes
    use_bf16 = (device.type == 'cuda')
    model_fwd = model
    if device.type == 'cuda' and not isinstance(model, nn.DataParallel):
        # Triton's bundled ptxas does not know the GB10 (sm_121a); point it at the system CUDA-13 ptxas.
        if os.path.exists('/usr/local/cuda/bin/ptxas'):
            os.environ.setdefault('TRITON_PTXAS_PATH', '/usr/local/cuda/bin/ptxas')
        try:
            compiled = torch.compile(model)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                compiled(images[:2].to(device),
                         torch.zeros(2, dtype=torch.long, device=device),
                         torch.zeros(2, dtype=torch.long, device=device))  # warm-up; raises here if compile is broken
            model_fwd = compiled
            print("torch.compile enabled for the training forward pass")
        except Exception as e:
            print(f"torch.compile unavailable ({type(e).__name__}); training uncompiled")
    
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
            # SPEED (Sep 2026): bf16 autocast for the forward/backward (GroupNorm and the loss stay fp32).
            # Was: predicted_noise = model(noisy_images, t, labels)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                predicted_noise = model_fwd(noisy_images, t, labels)
            predicted_noise = predicted_noise.float()

            # Calculate loss for the entire batch at once
            loss = nn.MSELoss(reduction='mean')(predicted_noise, target_noise) # No loop needed

            optimizer.zero_grad()
            loss.backward()
            # Use higher gradient clipping for optimized models (smaller models need less clipping)
            model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            max_norm = 1.0 if (getattr(model_unwrapped, 'use_optimized_cifar10', False) or getattr(model_unwrapped, 'use_optimized_mnist', False)) else 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            # EMA (Sep 2026): ema = decay*ema + (1-decay)*live, after every optimizer step
            with torch.no_grad():
                ema_step += 1
                decay = min(0.9999, (1 + ema_step) / (10 + ema_step))
                torch._foreach_lerp_(ema_params, live_params, 1.0 - decay)
            
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
            'dataset_name': dataset_name,  # Add dataset name to checkpoint
            'emb_dim': diffusion.emb_dim,  # Add emb_dim to checkpoint
            'use_attention': model_unwrapped.use_attention,  # Use unwrapped model here
            'use_optimized_cifar10': model_unwrapped.use_optimized_cifar10,  # Add optimized flag
            'use_optimized_mnist': model_unwrapped.use_optimized_mnist,  # Add optimized flag
            'ema_state_dict': ema_model.state_dict()  # EMA (Sep 2026): averaged weights used for sampling
        }
        
        # Save main checkpoint - FIXED to include timesteps in filename
        torch.save(checkpoint, f'diffusion_checkpoint_{dataset_name}_ts{diffusion.timesteps}_emb{diffusion.emb_dim}{attention_suffix}.pt')
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
            # EMA (Sep 2026): sample from the averaged weights (was: model)
            save_samples(ema_model, diffusion, device, epoch, avg_loss, dataset_name, use_batch_inference=use_batch_inference)

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
    
    # Extract emb_dim from checkpoint, default to 32 if not found
    emb_dim_ckpt = checkpoint.get('emb_dim', 32)
    print(f"Loaded emb_dim from checkpoint: {emb_dim_ckpt}") # Debug print

    # Create model with matching parameters from checkpoint
    in_channels = 3 if dataset_name.startswith('cifar10') else 1  # Handle both cifar10 and cifar10_optimized
    model = ConditionalUNet(
        timesteps=checkpoint['timesteps'],  # Use timesteps from checkpoint
        in_channels=in_channels,
        emb_dim=emb_dim_ckpt, # Use loaded emb_dim here
        use_attention=checkpoint.get('use_attention', False), # Load use_attention from checkpoint, default to False if not found
        use_optimized_cifar10=checkpoint.get('use_optimized_cifar10', False), # Load optimized flag from checkpoint
        use_optimized_mnist=checkpoint.get('use_optimized_mnist', False) # Load optimized flag from checkpoint
    ).to(device)
    
    # EMA (Sep 2026): prefer the averaged weights for generation when the checkpoint has them
    if 'ema_state_dict' in checkpoint:
        print("Using EMA weights for inference")
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion model with matching parameters
    diffusion = DiffusionModel(
        timesteps=checkpoint['timesteps'],
        schedule_type=checkpoint.get('schedule_type', 'linear'),  # Default to linear if not found
        noise_scale=checkpoint.get('noise_scale', 1.0),
        use_noise_scaling=checkpoint.get('use_noise_scaling', False),
        cosine_s=checkpoint.get('cosine_s', 0.008)
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
        parser = argparse.ArgumentParser(description='MNIST/CIFAR Diffusion Model')
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
        parser.add_argument('--cosine_s', type=float, default=None)
        parser.add_argument('--emb_dim', type=int, default=None, 
                           help='Embedding dimension (default: 128 for CIFAR, 32 for MNIST)')
        parser.add_argument('--use_attention', type=bool, default=None,
                           help='Use self-attention in the model')
        parser.add_argument('--num_gpus', type=int, default=None,
                           help='Number of GPUs to use (default: all available)')
        args = parser.parse_args()
        
        # Add dataset configuration right after device setup
        DATASETS = {
            'mnist': {
                'class': datasets.MNIST,
                'in_channels': 1,
                'image_size': 28,
                'normalize': ([0.5], [0.5]),
                # Add recommended defaults
                'defaults': {
                    'timesteps': 500,
                    'beta_start': 1e-5,
                    'beta_end': 0.01,
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'schedule_type': 'linear',
                    'cosine_s': 0.005,
                    'noise_scale': 0.8,
                    'emb_dim': 32  # Default for MNIST
                }
            },
            'cifar10': {
                'class': datasets.CIFAR10,
                'in_channels': 3,
                'image_size': 32,
                'normalize': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                'defaults': {
                    'timesteps': 500,           # Back to 500 as you suggested
                    # FIX (Sep 2026): with (1e-5, 0.012) at T=500, alpha_bar_T=0.049 so x_T still held 22% of the
                    # clean image, yet sampling starts from pure N(0,I) the net never saw. (2e-4, 0.04) is the
                    # DDPM (1e-4, 0.02 @ T=1000) schedule rescaled to T=500 -> alpha_bar_T=3.8e-5.
                    'beta_start': 2e-4,         # was 1e-5
                    'beta_end': 0.04,           # was 0.012
                    'batch_size': 32,           # Keep 32 for stability
                    'learning_rate': 1e-4,      # Reduced to 1e-4 for more stable training
                    'schedule_type': 'linear',  # Back to linear as you suggested
                    'cosine_s': 0.008,         # (not used with linear schedule)
                    'noise_scale': 0.9,         # Slightly reduced noise scale
                    'emb_dim': 128
                }
            },
            'cifar10_optimized': {
                'class': datasets.CIFAR10,
                'in_channels': 3,
                'image_size': 32,
                'normalize': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                'defaults': {
                    'timesteps': 500,           # Same as regular CIFAR-10
                    'beta_start': 2e-4,         # FIX (Sep 2026): was 1e-5, see 'cifar10' comment above
                    'beta_end': 0.04,           # FIX (Sep 2026): was 0.012, see 'cifar10' comment above
                    'batch_size': 128,          # Increased for better stability with smaller model
                    'learning_rate': 1e-4,      # Keep same as original to avoid NaN issues
                    'schedule_type': 'linear',  # Linear schedule (not cosine as requested)
                    'cosine_s': 0.008,         # Not used with linear
                    'noise_scale': 0.9,         # Same noise scale
                    'emb_dim': 128,             # Same embedding dimension
                    'use_optimized_cifar10': True  # Enable optimized model architecture
                }
            },
            'mnist_optimized': {
                'class': datasets.MNIST,
                'in_channels': 1,
                'image_size': 28,
                'normalize': ([0.5], [0.5]),
                'defaults': {
                    'timesteps': 500,           # Same as regular MNIST
                    'beta_start': 1e-5,         # Same gentler start
                    'beta_end': 0.01,           # Same as regular MNIST
                    'batch_size': 256,          # Larger batch size works with smaller model
                    'learning_rate': 1e-3,      # Keep same as regular MNIST
                    'schedule_type': 'linear',  # Linear schedule
                    'cosine_s': 0.005,         # Same as regular MNIST
                    'noise_scale': 0.8,         # Same as regular MNIST
                    'emb_dim': 32,              # Same embedding dimension
                    'use_optimized_mnist': True  # Enable optimized model architecture
                }
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
            if name in ['cifar10_optimized', 'mnist_optimized']:
                print(f"{idx}. {name.upper()} ({info['image_size']}x{info['image_size']}, {info['in_channels']} channels) - FASTER/BETTER")
            else:
                print(f"{idx}. {name.upper()} ({info['image_size']}x{info['image_size']}, {info['in_channels']} channels)")

        while True:
            dataset_choice = input("\nEnter your choice (1-{}): ".format(len(DATASETS))).strip()
            if dataset_choice.isdigit() and 1 <= int(dataset_choice) <= len(DATASETS):
                dataset_name = list(DATASETS.keys())[int(dataset_choice) - 1]
                dataset_config = DATASETS[dataset_name]
                break
            print(f"Please enter a number between 1 and {len(DATASETS)}")

        if args.mode == 'train':
            print("\n=== MNIST/CIFAR-10 Diffusion Model Parameter Guide ===")
            print("\nCore Parameters:")
            print(f"• Timesteps: {args.timesteps}")
            print("  - Higher values (1000+) = smoother transitions but slower training")
            print("  - Lower values (500-) = faster training but potentially lower quality")
            print("  - Recommended: 500 for CIFAR-10, 500-800 for MNIST")
            
            print(f"\n• Beta Schedule (start={args.beta_start}, end={args.beta_end})")
            print("  - Controls noise addition rate")
            print("  - Lower beta_start (1e-5) = clearer early steps")
            print("  - Higher beta_end (0.02-0.05) = more diverse samples")
            print("  - For MNIST: try beta_end=0.01-0.02")
            print("  - For CIFAR: try beta_end=0.02-0.05")
            
            print(f"\n• Noise Schedule Type: {args.schedule_type}")
            print("  - Linear: Simple uniform noise increase, good for starting out")
            print("  - Cosine: Often better quality but needs careful tuning")
            print("    • For MNIST: Use s=0.005 for gentler noise")
            print("    • For CIFAR: Use s=0.008-0.01 for complex images")

            # Get schedule type first
            if args.schedule_type is None:
                args.schedule_type = get_input_with_default("Noise Schedule (linear/cosine)", 'linear', str)
                
            # Only ask for cosine_s if cosine schedule is selected
            if args.schedule_type.lower() == 'cosine' and args.cosine_s is None:
                args.cosine_s = get_input_with_default("Cosine schedule offset (s)", 
                                                     0.008, 
                                                     float)
                print("  Note: Lower s (0.005) for simpler images like MNIST")
                print("        Higher s (0.008-0.01) for complex images like CIFAR")

            print(f"\n• Batch Size: {args.batch_size}")
            print("  - Larger = faster training but more memory")
            print("  - Smaller = more stable but slower")
            print("  - For MNIST: 64-128 usually works well")
            print("  - For CIFAR: 32-64 recommended")
            
            print(f"\n• Learning Rate: {args.learning_rate}")
            print("  - Higher (1e-3) = faster learning but potential instability")
            print("  - Lower (1e-5) = more stable but slower progress")
            print("  - With cosine schedule: try 1e-4 to 5e-4")
            print("  - With linear schedule: try 1e-3 to 5e-3")
            
            print("\nAdvanced Parameters:")
            print(f"• Noise Scale: {args.noise_scale}")
            print("  - Controls the amount of noise added during sampling")
            print("  - Higher (1.2+) = more diverse but potentially noisier samples")
            print("  - Lower (0.8-) = cleaner but less diverse samples")
            print(f"  - Default: {dataset_config['defaults']['noise_scale']} for {dataset_name.upper()}")
            print("  - Adjust if samples are too noisy or too similar")
            
            print(f"\n• Dynamic Noise Scaling: {'Enabled' if args.use_noise_scaling else 'Disabled'}")
            print("  - When enabled: noise scale decreases linearly from full value to 0")
            print("    • Starts at your chosen noise_scale value")
            print("    • Gradually reduces to 0 as sampling progresses")
            print("    • Formula: current_scale = noise_scale * (timestep / total_timesteps)")
            print("  - When disabled: noise scale stays constant at your chosen value")
            print("  - Default: Disabled")
            print("  - Enable for:")
            print("    • Potentially cleaner final results")
            print("    • More controlled noise reduction")
            print("  - Disable for:")
            print("    • Consistent noise level throughout")
            print("    • Traditional diffusion behavior")

            # Also update the input prompt to be clearer
            if args.use_noise_scaling is None:
                print("\nDynamic Noise Scaling:")
                print("- If enabled: noise scale will decrease linearly during sampling")
                print("- If disabled: noise scale will remain constant")
                use_scaling = input("Enable dynamic noise scaling? (y/n, default=y): ").lower().strip() or "y"
                args.use_noise_scaling = use_scaling == 'y'

            print(f"\n• Learning Rate: {args.learning_rate}")
            print("  - Higher (1e-3) = faster learning but potential instability")
            print("  - Lower (1e-5) = more stable but slower progress")
            print("  - With cosine schedule: try 1e-4 to 5e-4")
            print("  - With linear schedule: try 1e-3 to 5e-3")
            
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
                args.timesteps = get_input_with_default("Timesteps", dataset_config['defaults']['timesteps'], int)
            if args.beta_start is None:
                args.beta_start = get_input_with_default("Beta start", dataset_config['defaults']['beta_start'])
            if args.beta_end is None:
                args.beta_end = get_input_with_default("Beta end", dataset_config['defaults']['beta_end'])
            if args.noise_scale is None:
                args.noise_scale = get_input_with_default("Noise scale", dataset_config['defaults']['noise_scale'])
                # Add clear noise scaling prompt right after noise scale input
                print("\nDynamic Noise Scaling:")
                print("  - When enabled: noise decreases gradually during sampling")
                print("  - When disabled: noise stays constant")
                use_scaling = input("Enable dynamic noise scaling? (y/n, default=y): ").lower().strip() or "y"
                args.use_noise_scaling = use_scaling == 'y'

            if args.schedule_type is None:
                args.schedule_type = get_input_with_default("Noise Schedule (linear/cosine)", dataset_config['defaults']['schedule_type'], str)
            if args.schedule_type.lower() == 'cosine' and args.cosine_s is None:
                args.cosine_s = get_input_with_default("Cosine schedule offset (s)", dataset_config['defaults']['cosine_s'])
            if args.epochs is None:
                args.epochs = get_input_with_default("Epochs", 500, int)
            if args.batch_size is None:
                args.batch_size = get_input_with_default("Batch size", dataset_config['defaults']['batch_size'], int)
            if args.learning_rate is None:
                args.learning_rate = get_input_with_default("Learning rate", dataset_config['defaults']['learning_rate'])
            if args.emb_dim is None:
                args.emb_dim = get_input_with_default("Embedding dimension", 
                                                    dataset_config['defaults'].get('emb_dim', 128), 
                                                    int)
                print(f"\nUsing embedding dimension: {args.emb_dim}")

            # Now check if attention is None before asking
            if args.use_attention is None:
                print("\nSelf-attention can improve image quality but increases training time and memory usage.")
                use_attention = input("Use self-attention in the model? (y/n, default=y): ").lower().strip() or "y"
                args.use_attention = use_attention == 'y'

            print(f"\nUsing parameters:")
            print(f"Timesteps: {args.timesteps}")
            print(f"Beta start: {args.beta_start}")
            print(f"Beta end: {args.beta_end}")
            print(f"Epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Noise Schedule: {args.schedule_type}")
            print(f"Learning rate: {args.learning_rate}")

            # Add GPU selection option
            if args.num_gpus is None:
                total_gpus = torch.cuda.device_count()
                if total_gpus > 1:
                    print(f"\nMulti-GPU Training:")
                    print(f"Detected {total_gpus} GPUs available")
                    gpu_choices = [f"{i+1} GPU{'s' if i > 0 else ''}" for i in range(total_gpus)]
                    print("Options:")
                    for i, choice in enumerate(gpu_choices, 1):
                        print(f"{i}. Use {choice}")
                    
                    while True:
                        gpu_choice = input(f"\nHow many GPUs to use? (1-{total_gpus}, default={total_gpus}): ").strip()
                        if gpu_choice == "":
                            args.num_gpus = total_gpus
                            break
                        elif gpu_choice.isdigit() and 1 <= int(gpu_choice) <= total_gpus:
                            args.num_gpus = int(gpu_choice)
                            break
                        print(f"Please enter a number between 1 and {total_gpus}")
                else:
                    args.num_gpus = 1
                    print("\nSingle GPU training (only 1 GPU detected)")
            else:
                # Ensure num_gpus doesn't exceed available GPUs
                args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
                if args.num_gpus <= 0:
                    args.num_gpus = 1
            
            print(f"Using {args.num_gpus} GPU{'s' if args.num_gpus > 1 else ''}")

            # Ask about batch inference preference once
            print("\nSample Generation Options:")
            print("1. Batch generation (faster) - Generate all 10 classes at once")
            print("2. Sequential generation (slower, more stable) - Generate one class at a time")
            while True:
                choice = input("\nChoose sample generation method (1 or 2, default=1): ").strip()
                if choice == "" or choice == "1":
                    use_batch_inference = True
                    print("Using batch generation (faster)")
                    break
                elif choice == "2":
                    use_batch_inference = False
                    print("Using sequential generation (slower but more stable)")
                    break
                else:
                    print("Please enter 1 or 2")

            # Modify dataset loading
            # AUGMENT (Sep 2026): random horizontal flip for CIFAR only (standard for DDPM on CIFAR-10;
            # doubles effective data). Not applied to MNIST - a mirrored digit is a different digit.
            transform = transforms.Compose(
                ([transforms.RandomHorizontalFlip()] if dataset_config['in_channels'] == 3 else []) + [
                transforms.ToTensor(),
                transforms.Normalize(*dataset_config['normalize'])
            ])
            
            train_dataset = dataset_config['class'](
                root='./data',
                train=True,
                transform=transform,
                download=True
            )
            
            # Fix for DataParallel freezing: Reduce num_workers significantly
            # Too many workers can cause synchronization issues with DataParallel
            num_workers = min(4, torch.cuda.device_count() * 2) if torch.cuda.is_available() else 0
            print(f"Using {num_workers} DataLoader workers")

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                prefetch_factor=2 if num_workers > 0 else None  # Reduce prefetch to avoid memory issues
            )

            # Create diffusion model instance
            diffusion = DiffusionModel(
                timesteps=args.timesteps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                schedule_type=args.schedule_type,
                noise_scale=args.noise_scale,
                use_noise_scaling=args.use_noise_scaling,
                cosine_s=args.cosine_s,
                emb_dim=args.emb_dim
            )

            # Create model with correct channels
            model = ConditionalUNet(
                num_classes=10,
                emb_dim=args.emb_dim,
                timesteps=args.timesteps,
                in_channels=dataset_config['in_channels'],
                use_attention=args.use_attention,
                use_optimized_cifar10=dataset_config['defaults'].get('use_optimized_cifar10', False),
                use_optimized_mnist=dataset_config['defaults'].get('use_optimized_mnist', False)
            ).to(device)
            
            # Multi-GPU setup with specific device selection
            if args.num_gpus > 1:
                # Check if this is an optimized model - use single GPU for stability
                model_unwrapped = model
                if getattr(model_unwrapped, 'use_optimized_cifar10', False) or getattr(model_unwrapped, 'use_optimized_mnist', False):
                    optimized_type = "CIFAR-10" if getattr(model_unwrapped, 'use_optimized_cifar10', False) else "MNIST"
                    print(f"Optimized {optimized_type} model detected - using single GPU for stability")
                    print("Consider using --num_gpus 1 with optimized model to avoid NaN issues")
                    args.num_gpus = 1

                if args.num_gpus > 1:
                    print(f"Setting up DataParallel with {args.num_gpus} GPUs")
                    # Select specific devices
                    device_ids = list(range(args.num_gpus))
                    model = nn.DataParallel(model, device_ids=device_ids)
                    print(f"Model parallelized across GPUs: {device_ids}")
                    # Calculate and display effective batch size
                    effective_batch_size = args.batch_size * args.num_gpus
                    print(f"Effective batch size: {effective_batch_size}")

                    # Add DataParallel-specific fixes for freezing issues
                    print("Applied DataParallel fixes:")
                    print("- Reduced DataLoader num_workers to prevent synchronization issues")
                    print("- Use persistent workers to avoid worker respawning overhead")
                    print("- Reduced prefetch factor to minimize memory pressure")
                    print("- If still freezing, try reducing batch_size or num_gpus")
            
            # Update optimizer settings - add learning rate scaling for multi-GPU
            base_lr = args.learning_rate
            if args.num_gpus > 1:
                # Optional: Linear scaling rule for multi-GPU
                # args.learning_rate = base_lr * args.num_gpus
                print(f"Note: Consider scaling learning rate for multi-GPU training")
                print(f"Current learning rate: {args.learning_rate}")
                scale_lr = input("Scale learning rate by number of GPUs? (y/n, default=n): ").lower().strip()
                if scale_lr == 'y':
                    args.learning_rate = base_lr * args.num_gpus
                    print(f"Scaled learning rate to: {args.learning_rate}")
            
            # Create optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.005,
                betas=(0.9, 0.999)
            )
            
            # Update checkpoint and model paths to include dataset name
            checkpoint_path = None
            # Fix: Access use_attention through module if using DataParallel
            model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            attention_suffix = '_attention' if model_unwrapped.use_attention else ''
            main_checkpoint = f'diffusion_checkpoint_{dataset_name}_ts{args.timesteps}_emb{args.emb_dim}{attention_suffix}.pt'

            print("\nStarting Training Mode")
            print("---------------------")
            
            print(f"Training on {len(train_dataset)} images")
            print(f"Batch size: {args.batch_size}")
            print(f"Epochs: {args.epochs}")
            
            # Modified optimizer settings
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.005,              # Reduced from 0.01 to 0.005
                betas=(0.9, 0.999)              # Back to standard Adam betas
            )
            
            # Add override learning rate option when loading from checkpoint
            override_lr = None
            if os.path.exists(main_checkpoint):
                print("\nFound existing checkpoint.")
                override_choice = input("Would you like to override the learning rate? (y/n): ").lower()
                if override_choice == 'y':
                    override_lr = get_input_with_default("Enter new learning rate", args.learning_rate)
                    print(f"Will override learning rate to: {override_lr}")
                else:
                    print("Will continue with checkpoint's original learning rate")

            # When loading checkpoint, check embedding dimension match
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path)
                if checkpoint.get('emb_dim') != args.emb_dim:
                    print(f"\nWARNING: Checkpoint embedding dimension ({checkpoint.get('emb_dim')}) "
                          f"doesn't match requested dimension ({args.emb_dim})")
                    print("You'll need to start fresh training with the new embedding dimension.")
                    user_continue = input("Continue with fresh training? (y/n): ")
                    if user_continue.lower() != 'y':
                        print("Exiting...")
                        return

            # Set environment variables to help debug DataParallel issues
            if args.num_gpus > 1:
                # Only set CUDA_LAUNCH_BLOCKING if explicitly requested for debugging
                # TORCH_USE_CUDA_DSA=1 causes too many false positive warnings with complex models
                print("Using DataParallel with improved settings for stability")

            # Train the model
            try:
                train(model, diffusion, train_loader, optimizer, device, args.epochs, dataset_name, override_lr, use_batch_inference)
            except Exception as e:
                if args.num_gpus > 1:
                    print(f"\nTraining failed with DataParallel. Try these fixes:")
                    print(f"1. Reduce num_gpus from {args.num_gpus} to {max(1, args.num_gpus-1)}")
                    print(f"2. Reduce batch_size from {args.batch_size} to {max(1, args.batch_size//2)}")
                    print(f"3. Set CUDA_LAUNCH_BLOCKING=1 to debug exactly where it hangs")
                    print(f"4. Try single GPU training first to isolate the issue")
                raise e
            
            # Save final model
            torch.save(model.state_dict(), args.model_path)
            print(f"\nTraining complete! Model saved to {args.model_path}")
            
        else:  # inference mode
            inference_mode(args.model_path, device, dataset_name)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()