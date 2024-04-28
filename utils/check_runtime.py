# check runtime model size
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AutoTokenizer
from constants import *
import torch.autograd.profiler as profiler

from quantization.transformer import Transformer
from quantization.quantize import quantizer
from quantization.binarize import binarize
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = Transformer(n_layers=BASELINE_MODEL_NUMBER_OF_LAYERS, n_class=4, vocab=tokenizer.vocab_size)
    optimized_model = model
    
    # Load baseline mode
    baseline_model = model
    baseline_model.load_state_dict(torch.load("../models/baseline_best.pth"))

    # Load Quantize model
    quantized_model = quantizer(model, 4, True)
    quantized_model.load_state_dict(torch.load("../model/quantized_best.pth"))

    # Load Binarize model
    binarized_model = model
    binarize(binarized_model, 'ALL', 'basic', skip_final=True, kv_only=True)
    binarized_model.load_state_dict(torch.load("../models/binarized_best.pth"))

    # Load optimized model
    optimized_model = model
    binarize(optimized_model, 'ALL', 'optmized', skip_final=True, kv_only=True)
    optimized_model.load_state_dict(torch.load("../models/optimized_best.pth"))


    # Print model sizes
    print("Model Sizes:")
    print("Baseline Model Size: ", get_model_size(baseline_model), "bytes")
    print("Quantized Model Size: ", get_model_size(quantized_model), "bytes")
    print("Binarized Model Size: ", get_model_size(binarized_model), "bytes")
    print("Optimized Model Size: ", get_model_size(optimized_model), "bytes")

    # Estimate runtime memory sizes (using input shape [batch_size, seq_len])
    input_shape = (BATCH_SIZE, MAX_SEQ_LEN)
    print("\nRuntime Memory Sizes (Estimated):")
    print("Baseline Runtime Memory: ", estimate_runtime_memory(baseline_model, input_shape))
    print("Quantized Runtime Memory: ", estimate_runtime_memory(quantized_model, input_shape))
    print("Binarized Basic Runtime Memory: ", estimate_runtime_memory(binarized_model, input_shape))
    print("Binarized Optimized Runtime Memory:", estimate_runtime_memory(optimized_model, input_shape))



    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = Transformer(4,
    #                     tokenizer.vocab_size,
    #                     BASELINE_MODEL_NUMBER_OF_LAYERS,
    #                     BASELINE_MODEL_NUMBER_OF_HEADS,
    #                     BASELINE_MODEL_DIM).to(device)
    # inputs = torch.rand(1, 512, device=device).int()  # assume one sentence only?
    # masks = torch.rand(1, 512, device=device).int()
    # # warm up
    # model(inputs, masks)
    # with profiler.profile(with_stack=False,
    #                       profile_memory=True,
    #                       use_cuda=torch.cuda.is_available(),
    #                       with_flops=True) as prof:
    #     out_prob = model(inputs, masks)
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
