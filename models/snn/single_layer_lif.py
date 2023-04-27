import torch
from spikingjelly.activation_based import layer, neuron, surrogate
from torch import nn


class SingleLayerLIF(nn.Module):
    def __init__(self, n_input, n_output, tau=1):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.tau = tau

        self.layers = nn.Sequential(
            layer.Flatten(),
            layer.Linear(n_input, n_output, bias=False),
            neuron.ParametricLIFNode(
                init_tau=tau,
                surrogate_function=surrogate.ATan()
            ),
        )
        # self.layers = nn.Sequential(
        #     layer.Flatten(),
        #     layer.Linear(n_input, 5, bias=False),
        #     neuron.LIFNode(
        #         tau=tau,
        #         surrogate_function=surrogate.ATan()
        #     ),
        #     # layer.Flatten(),
        #     layer.Linear(5, n_output, bias=False),
        #     neuron.LIFNode(
        #         tau=tau,
        #         surrogate_function=surrogate.ATan()
        #     ),
        # )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
