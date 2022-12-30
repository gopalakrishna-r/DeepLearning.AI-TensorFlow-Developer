import matplotlib.pyplot as plt


def plot_graphs(history, history_param):
    plt.plot(history.history[history_param])
    plt.plot(history.history["val_" + history_param])
    plt.xlabel("Epochs")
    plt.ylabel(history_param)
    plt.legend([history_param, "val_" + history_param])
    plt.show()
