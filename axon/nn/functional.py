from typing import *
import math

import axon as ax


def layer_norm(x, weight, bias, eps):
    # Compute the mean of the input along the last axis
    mean = ax.mean(x, axes=-1)

    # Compute the variance of the input along the last axis
    variance = ax.var(x, axes=-1)

    # Compute the standard deviation with numerical stability
    std = ax.sqrt(variance + eps)

    # Normalize the input
    normalized = (x - mean) / std

    # Apply the learnable parameters (weight and bias)
    output = normalized * weight + bias

    return output
