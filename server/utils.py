import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(predictions, references):
    f1_scores = []
    for prediction, reference in zip(predictions, references):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(reference).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0)
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1_scores.append((2 * precision * recall) / (precision + recall))

    return np.mean(f1_scores)

class FakeQuantLinear(nn.Linear):
    def __init__(self, original_layer, bits=4):
        super().__init__(
            original_layer.in_features, 
            original_layer.out_features, 
            bias=original_layer.bias is not None
        )
        self.weight.data = original_layer.weight.data
        if original_layer.bias is not None:
            self.bias.data = original_layer.bias.data
        
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

    def pseudo_quantize(self, w):
        max_val = w.abs().max()
        scale = max_val / (self.qmax)
        
        w_int = (w / scale).round().clamp(self.qmin, self.qmax)
        w_float_simulated = w_int * scale
        
        return w_float_simulated

    def forward(self, input):
        quantized_weight = self.pseudo_quantize(self.weight)
        return F.linear(input, quantized_weight, self.bias)

def replace_with_fake_quantize(model, bits=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quant_layer = FakeQuantLinear(module, bits=bits)
            setattr(model, name, quant_layer)
        else:
            replace_with_fake_quantize(module, bits)
