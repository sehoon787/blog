import numpy as np
import pandas as pd

DATASET_PATH = 'data/0326_0927/co2_time_series.csv'

## load data
target = 'CO2'
df = pd.read_csv(DATASET_PATH, parse_dates=['Date'], index_col='Date')

def outlier_iqr(data):
    q25, q75 = np.quantile(data, 0.25), np.quantile(data, 0.75)
    iqr = q75 - q25

    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    print('IQR은', iqr, '이다.')
    print('lower bound 값은', lower, '이다.')
    print('upper bound 값은', upper, '이다.')

    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기
    data1 = data[data > upper]
    data2 = data[data < lower]

    # 이상치 총 개수 구하기
    print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.\n')
    return lower, upper

outlier_iqr(df[target])

## quantile 이용
Q1_quantile = df[target].quantile(.25)
Q2_quantile = df[target].quantile(.50)
Q3_quantile = df[target].quantile(.75)

## describe 이용
des = df[target].describe()
Q1_describe = des["25%"]
Q2_describe = des["50%"]
Q3_describe = des["75%"]

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)
data = difference(df[target])
outlier_iqr(data)