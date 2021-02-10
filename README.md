# Parametric LIF

Implementation of [Incorporating Learnable Membrane Time Constants to Enhance Learning of Spiking Neural Networks](http://arxiv.org/abs/2007.05785) by Fang et al. (2020).

Design choices in this paper:
- Feedforward
- Learnable time constants shared per layer (adjusted for numerical stability through sigmoid)
- LIF with hard, detached reset
- Fixed thresholds of 1
- Batch normalization (not per-timestep?)
- Max pooling
- Learnable encoding
- Population spike count decoding
- SG: derivative of `1/pi * arctan(pi * x) + 0.5`
- Param init: ...
- Small batches (16)
- Time steps: 8 for MNIST
- Adam with cosine annealing
