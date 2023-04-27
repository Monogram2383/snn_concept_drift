import torch
from spikingjelly.activation_based import functional
from tqdm import tqdm

from models.snn.single_layer_lif import SingleLayerLIF
import torch.nn.functional as F
from utils.visualization_utils import plot_train_history

NUM_OF_ITERATIONS = 200
DRIFT_STEP = 20


def train_model(model, stream,
                num_of_iterations=NUM_OF_ITERATIONS,
                enable_drift=True,
                drift_step=DRIFT_STEP,
                plot_results=True):
    metric_history = []
    drifts = []

    for i in tqdm(range(num_of_iterations)):
        x, y_true = stream.next_sample()
        y_pred = model.predict(x)

        metric_history.append(int(y_true == y_pred))
        model.partial_fit(x, y_true)

        if enable_drift and i % drift_step == 0:
            drifts.append(i)
            stream.generate_drift()

    accuracy = [sum(metric_history[:i]) / len(metric_history[:i]) for i in range(1, num_of_iterations)]
    if plot_results:
        plot_train_history(accuracy, "Accuracy", plot_drift=True, drifts=drifts)

    return model, accuracy


def train_snn_model(model, stream, num_of_iterations=NUM_OF_ITERATIONS, T=100, enable_drift=True, drift_step=DRIFT_STEP,
                    plot_results=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metric_history = []
    drifts = []

    for i in tqdm(range(num_of_iterations)):
        optimizer.zero_grad()
        x, y_true = stream.next_sample()
        x = torch.from_numpy(x).float()
        y_true = torch.from_numpy(y_true)

        y_true_onehot = F.one_hot(y_true, stream.n_classes).float()
        out_fr = 0.
        for t in range(T):
            out_fr += model(x)
        out_fr = out_fr / T
        loss = F.mse_loss(out_fr, y_true_onehot)
        loss.backward(retain_graph=True)
        optimizer.step()

        y_pred = out_fr.argmax(1)

        # y_pred = model.predict(x)

        metric_history.append(int(y_true == y_pred))
        # model.partial_fit(x, y_true)

        # After optimizing the parameters, the state of the network should be reset because the neurons of the SNN have “memory”.
        functional.reset_net(model)

        if enable_drift and i % drift_step == 0:
            drifts.append(i)
            stream.generate_drift()

    accuracy = [sum(metric_history[:i]) / len(metric_history[:i]) for i in range(1, num_of_iterations)]
    if plot_results:
        plot_train_history(accuracy, "Accuracy", plot_drift=True, drifts=drifts)

    return model, accuracy
