import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def draw_plot(data, data_diff):
    plt.plot(data, label="Original")
    plt.plot(data_diff, label="diff_1")
    plt.suptitle("CO2", size=24)
    plt.xlabel("Time")
    plt.legend()
    plt.show()

## ADF
def ADF(data):

    result = adfuller(data, autolag="AIC")

    print("---- Adfuller ----")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %1.10f' % result[1])
    print('Lag: %d' % result[2])
    print('observation: %d' % result[3])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

if __name__ == '__main__':
    data = pd.DataFrame(pd.read_csv("co2_time_series.csv")["CO2"][:500])
    data_diff = data.diff(axis=0).dropna()

    draw_plot(data=data, data_diff=data_diff)
    ADF(data_diff)