import torch
import pytorch3d
import subprocess

print(torch.__version__)
print('gpu:', torch.cuda.is_available())

# torch.cuda.empty_cache()
print(torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
print(f"Reservrd memory: {torch.cuda.memory_reserved()} bytes")
print(f"allocated memory: {torch.cuda.memory_allocated()} bytes")
print(f"Free memory: {torch.cuda.memory_reserved() - torch.cuda.memory_allocated()} bytes")