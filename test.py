import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
    print("Device:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")