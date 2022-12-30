import matplotlib.pyplot as plt


def plot_series(
    time,
    series,
    format="-",
    label=None,
    start=0,
    end=None,
    wandb_run=None,
    wandb_chart_name=None,
):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Series")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    if wandb_run and wandb_chart_name:
        wandb_run.log({wandb_chart_name: plt})
