import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skmultiflow.data import SEAGenerator
from tqdm import tqdm

random.seed(0)
torch.manual_seed(0)

NUM_OF_ITERATIONS = 200
DRIFT_STEP = 20
LEARNING_RATE = 1e-4
TAU = 5.0

stream = SEAGenerator(
    random_state=0,
    balance_classes=False,
    noise_percentage=0
)

from models.snn.single_layer_lif import SingleLayerLIF

n_input = stream.n_features + 1
n_output = 2
model = SingleLayerLIF(n_input=n_input, n_output=n_output, tau=TAU)


def train_concept_drift_detection(model, stream, num_of_iterations=NUM_OF_ITERATIONS, drift_step=DRIFT_STEP, T=10,
                                  to_plot=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metric_history = []

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    drifts = []
    drift_detected = []
    false_drift_detected = []

    is_stream_drifted = False
    last_drift_detection_i = 0
    last_drift_i = 0
    drift_detection_time = []
    l1_loss = []
    for i in tqdm(range(num_of_iterations)):
        optimizer.zero_grad()
        x, stream_y_true = stream.next_sample()
        X = np.concatenate((x, [stream_y_true]), axis=1)

        X = torch.from_numpy(X).float()

        out_fr = 0.
        for t in range(T):
            out_fr += model(X)
        out_fr = out_fr / T

        y_pred_drift = bool(out_fr.argmax(1)[0])
        metric_history.append(int(is_stream_drifted == y_pred_drift))  # update the metric history

        # loss = F.mse_loss(out_fr, torch.Tensor([int(is_stream_drifted)]))
        y_true = torch.Tensor([0, 1]) if is_stream_drifted else  torch.Tensor([1, 0]) # TODO: remake it to one hot
        # loss = F.mse_loss(out_fr, y_true)
        loss = F.l1_loss(out_fr, y_true)
        l1_loss.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()

        # stream will be drifted only after previous drift was detected and drift_step steps passed
        if (i - last_drift_detection_i + 1) >= drift_step and is_stream_drifted is False:
            drifts.append(i)
            stream.generate_drift()  # y is drifted
            is_stream_drifted = True
            last_drift_i = i
        if is_stream_drifted is False and y_pred_drift is True:
            false_drift_detected.append(i)
            false_positives += 1
        elif is_stream_drifted is True and y_pred_drift is True:
            drift_detection_time.append(i - last_drift_i)
            last_drift_detection_i = i
            drift_detected.append(i)
            is_stream_drifted = False
            true_positives += 1
        elif is_stream_drifted is True and y_pred_drift is False:
            false_negatives += 1

    accuracy = [sum(metric_history[:i]) / len(metric_history[:i]) for i in range(1, num_of_iterations)]

    if to_plot:
        num_of_iterations = len(metric_history)
        time = [i for i in range(1, num_of_iterations)]
        # time = [i for i in range(len(l1_loss))]
        plt.plot(time, accuracy, label="Accuracy")
        plt.vlines(x=drifts, color='r', ymin=0., ymax=1., label='Drift', linestyle='--')
        plt.vlines(x=drift_detected, color='g', ymin=0., ymax=1., label='Drift Detected', linestyle='-.')
        plt.vlines(x=false_drift_detected, color='b', ymin=0., ymax=1., label='False Drift Detection', linestyle='-.')
        plt.title(f"lr: {LEARNING_RATE}; tau: {TAU}; T: {T}")
        plt.legend()
        plt.show()

    print(f'last acc: {accuracy[-1]: 0.2f}, avg acc: {np.mean(accuracy): 0.2f}, max acc: {np.max(accuracy): 0.2f}')
    # print(f'last acc: {accuracy: 0.2f}')
    print(
        f"Precision: {true_positives / (true_positives + false_positives)}; recall: {true_positives / (true_positives + false_negatives)}")
    print(f"Average time needed to detect drift: {sum(drift_detection_time) / len(drift_detection_time)}")


train_concept_drift_detection(model, stream, num_of_iterations=200, to_plot=True)

# TODO
#   - move to colab and train for more iterations on GPU
#   - play with tau, T and learning rate
#   - check STDP
#   - try multilayer again
#   - train regular small RNN/LSTM network to see its metrics
