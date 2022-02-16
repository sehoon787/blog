import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

a = [66, 74, 82, 75, 73, 97, 87, 78]
b = [72, 51, 59, 62, 74, 64, 78, 63]
c = [61, 60, 57, 60, 81, 55, 70, 71]
print("a 평균 : ", np.mean(a))
print("b 평균 : ", np.mean(b))
print("c 평균 : ", np.mean(c))

plot_data = [a, b, c]
plt.boxplot(plot_data)
plt.xticks([1, 2, 3],['a', 'b', 'c'])
plt.grid(True)
plt.show()

F_statistic, pVal = stats.f_oneway(a, b, c)
print('F={0:.1f}, p-value={1:.3f}'.format(F_statistic, pVal))

