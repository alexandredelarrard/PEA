import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def var_vs_target(data, y_label, variable, bins=30, normalize=False):

    if isinstance(y_label, str):
        y_label = [y_label]

    data = data.copy()

    if variable not in data.columns:
        return "variable not in database"

    if len(data[variable].value_counts()) > bins:
        if data[variable].dtype != "O":
            data[variable] = pd.qcut(data[variable], bins, precision=1, duplicates="drop")
        else:
            modalities = data[variable].value_counts().index[:bins]
            data.loc[~data[variable].isin(modalities), variable] = "other"

    avg_target = data[y_label].mean()
    if normalize:
        target_values = data[[variable] + list(y_label)].groupby(variable).mean() / data[list(y_label)].mean()

    else:
        target_values = data[[variable] + list(y_label)].groupby(variable).mean()

    grouped = data[[variable] + list(y_label)].groupby(variable).agg([np.size, np.std])

    ### add confidence_interval
    plt.figure(figsize=(12, 8))

    ax1 = grouped[y_label[0]]["size"].plot(kind="bar", alpha=0.42, grid=True)
    ax2 = ax1.twinx()

    if normalize:
        ax2.set_ylim([np.min(np.min(target_values)) * 0.95, np.max(np.max(target_values)) * 1.05])

    s = ax2.plot(ax1.get_xticks(), target_values[y_label], linestyle="-", label=[y_label])

    ax1.set_ylabel(f"{str(variable)} Volume")
    ax2.set_ylabel(f"{str(y_label)}")
    ax1.set_xlabel(f"{str(variable)}")

    plt.title(f"Evolution of {variable} vs {y_label}")
    ax2.legend(tuple(y_label), loc=1, borderaxespad=0.0)

    if not normalize:
        for i, value in enumerate(avg_target):
            plt.axhline(
                y=value,
                xmin=0,
                xmax=3,
                linewidth=0.5,
                linestyle="--",
                color=s[i].get_color(),
            )
            plt.errorbar(
                ax1.get_xticks(),
                target_values[y_label[i]],
                yerr=1.96 * grouped[y_label[i]]["std"] / np.sqrt(grouped[y_label[0]]["size"]),
                alpha=0.65,
                color=s[i].get_color(),
            )

    plt.setp(ax1.xaxis.get_ticklabels(), rotation=78)
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
