import numpy as np
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier

from train import train_model

stream = SEAGenerator(
    random_state=145,
    balance_classes=False,
    noise_percentage=0.33
)

model = HoeffdingTreeClassifier()
_, accuracy = train_model(model, stream, enable_drift=True)
print(f'last acc: {accuracy[-1]: 0.2f}, avg acc: {np.mean(accuracy): 0.2f}, max acc: {np.max(accuracy): 0.2f}')
