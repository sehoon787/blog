# white noise
from random import gauss
from random import seed
from pandas import Series
import matplotlib.pyplot as plt

# seed random number generator
seed(1)
# create white noise series
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)

# summary stats
print(series.describe())

# line plot
series.plot()
plt.show()

import pandas as pd
df = pd.read_csv("co2_time_series.csv")["CO2"]
plt.plot(df)
plt.show()