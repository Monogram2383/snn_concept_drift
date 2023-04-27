import numpy as np
import torch
from skmultiflow.data import SEAGenerator

from models.snn.single_layer_lif import SingleLayerLIF
from train import train_snn_model
# torch.autograd.set_detect_anomaly(True)
stream = SEAGenerator(
    random_state=145,
    balance_classes=False,
    # noise_percentage=0.33
)

n_input = stream.n_features
n_output = stream.n_classes
model = SingleLayerLIF(n_input=n_input, n_output=n_output, tau=2.0)

_, accuracy = train_snn_model(model, stream, num_of_iterations=200, enable_drift=True)
print(f'last acc: {accuracy[-1]: 0.2f}, avg acc: {np.mean(accuracy): 0.2f}, max acc: {np.max(accuracy): 0.2f}')
