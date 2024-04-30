import torch
import torch.autograd.profiler as profiler
from torch_api import *

# Count how many trainable weights the model has
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# byte to MB conversion
def bytes_to_MB(bytes):
    MB = bytes / (1024 * 1024)
    return MB

# Count how large memory this model uses
def count_memory_size(model, mtype=None, mname=None):
    model.initialize(model, mtype, mname)
    total_memory = sum(
        p.numel() * (get_default_dtype(model, mname, p)) for p in model.parameters() if p.requires_grad
    )
    total_memory = round(bytes_to_MB(total_memory), 1)
    print(f"{mname}: ", total_memory, "M")

# Count run time usage
def get_run_time(model, inputs, masks, mtype=None, mname=None):
    model.initialize(model, mtype, mname)
    model(inputs, masks)
    with profiler.profile(with_stack=False,
                          profile_memory=True,
                          use_cuda=torch.cuda.is_available(),
                          with_flops=True) as prof:
        out_prob = model(inputs, masks)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
