from typing import List

import matplotlib.pyplot as plt


def plot_train_history(metric_history: List[float], metric_name: str = None, plot_drift: bool = False,
                       drifts: List[int] = []):
    num_of_iterations = len(metric_history)
    if metric_name is None or metric_name == '':
        metric_name = "metric"

    time = [i for i in range(1, num_of_iterations + 1)]
    plt.plot(time, metric_history, label=metric_name)

    if plot_drift:
        plt.vlines(x=drifts, color='r', ymin=0., ymax=1., label='Drift', linestyle='--')

    plt.legend()
    plt.show()
