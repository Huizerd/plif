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

Possible discrepancies between this implementation and the paper:
- Larger batch size (128 vs 16) to speed up training
- Initialization of weights: not described in paper
- Batch normalization: this paper and [Ledinauskas 2020](http://arxiv.org/abs/2006.04436) seem to describe shared statistics for all timesteps, but some works ([Cooijmans 2017](http://arxiv.org/abs/1603.09025), [Kim 2020](http://arxiv.org/abs/2010.01729)) use separate statistics/parameters
- Output voting layer: implemented and described as average pooling spatially, but what about the temporal dimension? Sum over time? Average? Take last step only?