import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(17)
t = pd.Series(0.2 + np.random.rand(10)).cumsum()
t.name = 'time / s'
h = t.apply(lambda x: -x - np.random.randn())
h.name = 'height / m'

np.random.seed(42)
x = pd.Series(40 * np.random.rand(50))
x.name = 'hours studied'
y = x / 40 + 0.1 * np.random.randn(50) > 0.5
y.name = 'pass/fail'


def plot_model(
    axis,
    model: callable,
    x_min,
    x_max,
    samples: int = 1000,
    label='prediction',
    color='orange',
):
    """Plots the predictions of a 1D model over a range on an axis"""
    prediction_range = pd.Series(np.linspace(x_min, x_max, samples))
    predictions = prediction_range.transform(model)
    axis.plot(prediction_range, predictions, color=color, label=label)


def plot_errorbars(axis, x, y, yerr):
    """Plots errorbars between yerr and y onto an axis"""
    yerrbars = np.array([[abs(min(0, err)), max(0, err)] for err in yerr]).T
    axis.errorbar(
        x, y, yerr=yerrbars, fmt='none', ecolor='red', label='errors'
    )


def calculate_metrics(yerr, logistic_regression=False):
    """Calculates mean error, MAE, and MSE from a series of residuals."""
    if logistic_regression:
        metrics = {'Average cross-entropy': -np.log(1 - abs(yerr)).mean()}
    else:
        metrics = {
            'Mean Error': yerr.mean(),
            'Mean Absolute Error': abs(yerr).mean(),
        }
    metrics['Mean Squared Error'] = (yerr ** 2).mean()
    return metrics


def visualize_model(
    x: pd.Series,
    y: pd.Series,
    model: callable = None,
    samples: int = 1000,
    errorbars: bool = False,
    plot_residual: bool = False,
    residual_histogram: bool = False,
    bins=50,
):
    """Plots data from two series, a model, and its residual"""

    figure = plt.figure(figsize=(12, 12))

    gridspec = figure.add_gridspec(
        2, 2, width_ratios=(7, 1), height_ratios=(4, 4)
    )

    data_axis = figure.add_subplot(gridspec[0, 0])

    data_axis.set_xlabel(x.name)
    data_axis.set_ylabel(y.name)
    data_axis.scatter(x, y, label='data')

    metrics = None

    if model is not None:
        plot_model(data_axis, model, x.min(), x.max(), samples)

        predictions = x.transform(model)
        residual = y - predictions

        metrics = calculate_metrics(residual, np.allclose(abs(y - 0.5), 0.5))

        if errorbars:
            plot_errorbars(data_axis, x, predictions, residual)

        data_axis.legend()

        if plot_residual:
            residual_axis = figure.add_subplot(
                gridspec[1, 0], sharex=data_axis
            )
            residual_axis.set_ylabel('Residual')
            residual_axis.scatter(x, residual, label='residual')

            if residual_histogram:
                residual_histogram_axis = figure.add_subplot(
                    gridspec[1, 1], sharey=residual_axis
                )
                residual_histogram_axis.hist(
                    residual, orientation='horizontal', bins=bins
                )

    return figure, metrics
