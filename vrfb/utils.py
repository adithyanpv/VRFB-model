# utils.py

import numpy as np
import matplotlib.pyplot as plt
import os


def compute_metrics(true_soc, est_soc):
    error = true_soc - est_soc

    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    final_error = error[-1]

    return rmse, mae, max_error, final_error


def save_plots(folder, time, true_soc, est_soc, voltage, current, temperature, flow):

    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(time, true_soc, label="True SOC")
    plt.plot(time, est_soc, label="CC SOC")
    plt.legend()
    plt.title("SOC Comparison")
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(time, true_soc - est_soc)
    plt.title("SOC Error")
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(time, voltage)
    plt.title("Voltage")
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(time, current)
    plt.title("Current")
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(time, temperature)
    plt.title("Temperature")
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(time, flow)
    plt.title("Flow")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{folder}/results.png")
    plt.close()
def save_metrics(folder, rmse, mae, max_error, final_error):

    os.makedirs(folder, exist_ok=True)

    with open(f"{folder}/metrics.txt", "w") as f:
        f.write("=== SOC Estimation Performance ===\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"Max Absolute Error: {max_error:.6f}\n")
        f.write(f"Final Drift Error: {final_error:.6f}\n")