import math
import torch
import torch.nn as nn
from torch.autograd import Function
import pdb

class BinaryLinearFunction(Function):
    """
    Implements binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # binarize weights by sign function
        weight_mask = (weight > 1) | (weight < -1)
        weight = torch.sign(weight)
        
        # save for grad computing
        ctx.save_for_backward(input, weight, weight_mask, bias)
        
        # linear layer
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight, weight_mask, bias = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # if weights' absolute value larger than 1, no grads
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            grad_weight.masked_fill_(weight_mask, 0.0)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class BinarizedLinear(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return BinaryLinearFunction.apply(input, self.weight, self.bias)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"

class OptimizedBinarizationFunction(Function):
    """
    Implements optimized binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # binarize weights by sign function
        weight_mask = (weight > 1) | (weight < -1)
        in_dim = list(input.size())[-1]
        weight = torch.sign(weight) / (2 * math.sqrt(in_dim))
        
        # save for grad computing
        ctx.save_for_backward(input, weight, weight_mask, bias)
        
        # linear layer
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight, weight_mask, bias = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # if weights' absolute value larger than 1, no grads
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            grad_weight.masked_fill_(weight_mask, 0.0)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
    

class OptimizedBinarization(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    """

    def __init__(self, in_features, out_features, bias=True):
        super(OptimizedBinarization, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return OptimizedBinarizationFunction.apply(input, self.weight, self.bias)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"
   

def binarize(model, pattern, binarize_layer='basic', skip_final=False, qk_only=False, qv_only=False, kv_only=False):
    """
    Recursively replace linear layers with binary layers
    ---------
    Arguments:
    model           - Model to be binarized
    pattern         - Binarization pattern
    binarize_layer  - either 'basic' or 'optmized'
    skip_final      - whether to leave final layer unbinarized
    qk_only         - whether to only binarize Q net and K net, leaving V net
    ---------
    No return, the original model is binarized
    """
    
#     pdb.set_trace()
    
    patterns = ['MHA', 'FFN', 'CLS', 'ALL']
    
    if pattern not in patterns:
        raise Exception(f'Unimplemented pattern, pattern should be in {patterns}, got {pattern}!')
    
    if pattern == 'MHA':
        model = model.__dict__["_modules"]['sublayer_attention']
    elif pattern == 'FFN':
        model = model.__dict__["_modules"]['sublayer_ffn']
    elif pattern == 'CLS':
        model = model.__dict__["_modules"]['classifier']
    elif pattern == 'ALL':
        model = model
    
    for name, layer in model.named_children():
        # Binarization
        if type(layer) == nn.Linear:
            if (skip_final == True) & (layer.out_features == 4):
                continue
            if (kv_only == True) & (name == '0'):
                continue
            if (qv_only == True) & (name == '1'):
                continue
            if (qk_only == True) & (name == '2'):
                continue
                
            if binarize_layer == 'basic':
                b = BinarizedLinear(layer.in_features, layer.out_features)
            elif binarize_layer == 'optimized':
                b = OptimizedBinarization(layer.in_features, layer.out_features)  
            model.__dict__["_modules"][name] = b
        else:
            layer_types = [type(layer) for layer in layer.modules()]

            if nn.Linear in layer_types:
                binarize(layer, 'ALL', binarize_layer, skip_final, qk_only, qv_only, kv_only)
    return model