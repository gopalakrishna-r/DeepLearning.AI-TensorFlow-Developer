import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-whitegrid")


def plot_mi_scores(scores):
    plt.figure(dpi=100, figsize=(8, 5))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("mutual information scores ")
    plt.show()
