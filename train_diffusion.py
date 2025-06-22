import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from tqdm import tqdm
from diffusion_model import MarioFrameDataset
from torch.utils.data import DataLoader
from diffusion_model import MarioDiffusion
import torch.nn.functional as F

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionTrainer:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move all beta schedule tensors to GPU
        self.betas = linear_beta_schedule(timesteps).to(self.device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, action, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, action)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def train(self, dataloader, epochs=100, lr=2e-4):
        optimizer = Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch data to GPU
                current_frame, action, next_frame = [b.to(self.device, non_blocking=True) for b in batch]
                
                # Get the difference frame
                diff_frame = next_frame - current_frame
                
                optimizer.zero_grad(set_to_none=True)  # Slightly more efficient than zero_grad()
                
                # Mixed precision training
                with autocast():
                    batch_size = current_frame.shape[0]
                    t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                    loss = self.p_losses(diff_frame, t, action)
                
                # Scale loss and call backward
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

# Usage example:
if __name__ == "__main__":
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Initialize dataset and dataloader with larger batch size
    dataset = MarioFrameDataset("mario_frame_dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=64,  # Increased batch size
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU cores
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
    )
    
    # Initialize model and move to GPU
    model = MarioDiffusion().to(device)
    
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Create trainer and start training
    trainer = DiffusionTrainer(model)
    
    try:
        trainer.train(dataloader, epochs=100)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        torch.cuda.empty_cache() 