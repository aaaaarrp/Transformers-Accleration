import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AutoTokenizer
from utils.constants import *
import torch.autograd.profiler as profiler

from quantization.transformer import Transformer
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = Transformer(4,
                        tokenizer.vocab_size,
                        BASELINE_MODEL_NUMBER_OF_LAYERS,
                        BASELINE_MODEL_NUMBER_OF_HEADS,
                        BASELINE_MODEL_DIM).to(device)
    
    inputs = torch.rand(1, 512, device=device).int()  # assume one sentence only?
    masks = torch.rand(1, 512, device=device).int()
    

    # Load baseline mode
    baseline = torch.load("models/baseline_best.pth", map_location=torch.device(device))
    # Load Quantize model
    quantized = torch.load("models/quantized_best.pth", map_location=torch.device(device))
    # Load Binarize model
    binarized = torch.load("models/binarized_best.pth", map_location=torch.device(device))
    # Load optimized model
    optimized = torch.load("models/optimized_best.pth", map_location=torch.device(device))



    # get_run_time(model, inputs, masks, baseline, "baseline")
    # get_run_time(model, inputs, masks, quantized, "quantized")
    # get_run_time(model, inputs, masks, binarized, "binarized")
    # get_run_time(model, inputs, masks, optimized, "optmized")

    # Get model size
    print("\nModel size - ")
    count_memory_size(model, baseline, "baseline")
    count_memory_size(model, quantized, "quantized")
    count_memory_size(model, binarized, "binarized")
    count_memory_size(model, optimized, "optimized")








