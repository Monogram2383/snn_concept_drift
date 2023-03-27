from tqdm import tqdm

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
