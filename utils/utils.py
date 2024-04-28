import torch
import os
import psutil

# Count how many trainable weights the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count how much large memory this model uses
def count_memory_size(model):
    memory_dict = {torch.float32:4, torch.float64:8}
    return sum(p.numel()*memory_dict[p.data.dtype] for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    # Save model to a temporary file
    torch.save(model.state_dict(), "../models/temp.pth")
    # Get the size of the temporary file
    model_size = os.path.getsize("../models/temp.pth")
    # Delete the temporary file
    os.remove("../models/temp.pth")
    return model_size

def estimate_runtime_memory(model, input_shape):
    # Estimate runtime memory using psutil to monitor memory usage during inference
    process = psutil.Process(os.getpid())
    before_memory = process.memory_info().rss / 1024 / 1024  # Memory usage before inference
    
    # Perform inference
    with torch.no_grad():
        model.eval()
        input_tensor = torch.randn(input_shape).to(next(model.parameters()).device)
        _ = model(input_tensor)
    after_memory = process.memory_info().rss / 1024 / 1024  # Memory usage after inference
    runtime_memory = after_memory - before_memory  # Estimated runtime memory
    return runtime_memory