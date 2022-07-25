from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_contributions(fig_name: str,
                       title: str,
                       contributions: np.ndarray,
                       n_to_plot: int = 5,
                       index_labels: Optional[List[str]] = None,
                       show: bool = False,
                       save: bool = True) -> None:
    """ Plot the top n_to_plot contributing variables to a faulty sample

    Parameters
    ----------
    fig_name: str
        The name of the figure used for the file name if saving
    title: str
        The title on the figure
    contributions: numpy.ndarray
        The numpy array of shape (n, ) containing the variable
        fault contributions of a sample
    n_to_plot: int
        The number of contributions to plot
    index_labels: list
        List of n strings describing each variable
    show: bool
        Show the plot
    save: bool
        Save the plot to the current directory
    """
    _f, ax = plt.subplots()
    _f.set_size_inches(8, 6)

    # Order contributions by descending order
    order = np.argsort(-1 * contributions)
    ordered_cont = contributions[order]
    cum_percent = np.cumsum(ordered_cont) / np.sum(contributions)
    bar_labels = [str(x) for x in order]

    ax.bar(bar_labels[:n_to_plot], ordered_cont[:n_to_plot])
    ax2 = ax.twinx()
    ax2.plot(cum_percent[:n_to_plot], 'r')

    ax.set_title(title)

    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Fault Contribution")
    ax2.set_ylabel("Cumulative Contribution")
    ax2.set_ylim([0, 1])

    if index_labels is not None:
        text_box = 'Variable Index Descriptions'
        for i in range(n_to_plot):
            index = order[i]
            label = index_labels[index]
            text_box += f'\nIndex {index} | {label}'

        bbox = dict(boxstyle="square", ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),
                    alpha=0.7)
        ax2.text(x=0.1, y=0.1, s=text_box, bbox=bbox, transform=ax.transAxes)

    if show:
        plt.show()
    if save:
        plt.savefig(fig_name, dpi=350)
    plt.close(fig=_f)
    _f = None
