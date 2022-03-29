import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("co2_time_series.csv")["CO2"]
y = np.log(df)

plt.plot(df, label="Original")
plt.plot(y, label="log")
plt.suptitle("Log Transform", size=24)
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("co2_time_series.csv")["CO2"]
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)
diff_1 = difference(df)

plt.plot(df, label="Original")
plt.plot(diff_1, label="diff_1")
plt.suptitle("Differencing", size=24)
plt.legend()
plt.show()