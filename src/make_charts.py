import pandas as pd
import matplotlib.pyplot as plt


def line_chart(df, x, y, title, xlabel, ylabel, filename, err=None, zero_line=False):
    plt.figure(figsize=(7, 4.5))
    for name, g in df.groupby("model"):
        if err:
            plt.errorbar(g[x], g[y], yerr=g[err], marker="o", capsize=3, label=name)
        else:
            plt.plot(g[x], g[y], marker="o", label=name)
    if zero_line:
        plt.axvline(0, color="grey", linestyle="--", alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"charts/{filename}", dpi=200)
    plt.close()


noise = pd.read_csv("results/noise_results.csv")
missing = pd.read_csv("results/missing_results.csv")
shift = pd.read_csv("results/shift_results.csv")

line_chart(noise, "noise_level", "mean_accuracy",
           "Accuracy under Gaussian noise", "Noise level (x feature std)",
           "Accuracy", "noise_accuracy.png", err="std_accuracy")

line_chart(missing, missing.columns[1], "mean_accuracy",
           "Accuracy under missing values", "Fraction of values missing",
           "Accuracy", "missing_accuracy.png", err="std_accuracy")

line_chart(shift, "shift", "accuracy",
           "Accuracy under distribution shift", "Shift (x feature std)",
           "Accuracy", "shift_accuracy.png", zero_line=True)

line_chart(shift, "shift", "malignant_recall",
           "Malignant recall under distribution shift", "Shift (x feature std)",
           "Malignant recall", "shift_recall.png", zero_line=True)

print("Saved 4 charts to charts/")
