import torch
from torch.utils.cpp_extension import CUDA_HOME

print(f"CUDA_HOME: {CUDA_HOME}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Built With CUDA: {torch.version.cuda}")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Create a test tensor on GPU
    x = torch.randn(1000, 1000).to(device)
    y = torch.matmul(x, x)
    print("Successfully ran a GPU computation!")