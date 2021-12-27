import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def acf(data, k):
    data = np.array(data).reshape(-1)
    mean = data.mean()

    numerator = np.sum((data[:len(data)-k] - mean) * (data[k:] - mean))
    denominator = np.sum(np.square(data - mean))

    acf_val = numerator / denominator

    return acf_val

def pacf(data, k):
    if k == 0:
        pacf_val = 1
    else:
        gamma_array = np.array([acf(data, k) for k in range(1, k + 1)])

        gamma_matrix = []
        for i in range(k):
            temp = [0] * k
            temp[i:] = [acf(data, j) for j in range(k - i)]     # making diagonal
            gamma_matrix.append(temp)

        gamma_matrix = np.array(gamma_matrix)
        gamma_matrix = gamma_matrix + gamma_matrix.T - np.diag(gamma_matrix.diagonal())     # making symmetric matrix
        pacf_val = np.linalg.inv(gamma_matrix).dot(gamma_array)[-1]
    return pacf_val

# create a difference series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# using plot_acf, plot_pacf
def draw_plot(target, data, lag):
    # mean variance comparision
    print("---" + target + "---")
    print("mean of left group, right group: {}, {}".format(
        data[:len(data) // 2].mean(), data[len(data) // 2:].mean()))
    print("std of left group, right group: {}, {}\n\n".format(
        data[:len(data) // 2].std(), data[len(data) // 2:].std()))

    # draw original data
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plt.suptitle("Stationary " + target, fontsize=24)
    axes[0].plot(data)  # draw original data
    sns.distplot(data, ax=axes[1])  # draw histogram to check to follow gaussian dist.
    plt.show()

    ## ACF
    print(sm.tsa.stattools.acf(data, nlags=lag, fft=False))
    plot_acf(data, lags=lag, use_vlines=True)
    plt.suptitle(target + " ACF", fontsize=18)
    plt.xlabel("lag")
    plt.show()

    ## PACF
    print(sm.tsa.stattools.pacf(data, nlags=lag, method='ywm'))
    plot_pacf(data, lags=lag, use_vlines=True)
    plt.suptitle(target + " PACF", fontsize=18)
    plt.xlabel("lag")
    plt.show()

if __name__ == '__main__':
    target = "CO2"
    data = pd.read_csv("co2_time_series.csv")[target][:500]
    data = difference(data)
    lag = 15

    acf_result = [acf(data, k) for k in range(0, lag+1)]
    print(acf_result)

    pacf_result = [pacf(data, k) for k in range(lag+1)]
    print(pacf_result)

    draw_plot(target=target, data=data, lag=lag)