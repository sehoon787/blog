import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame([
        ['A01', 2, 1, 60, 139, 'country', 0, 'fail'],
        ['A02', 3, 2, 80, 148, 'country', 0, 'fail'],
        ['A03', 3, 4, 50, 149, 'country', 0, 'fail'],
        ['A04', 5, 5, 40, 151, 'country', 0, 'pass'],
        ['A05', 7, 5, 35, 154, 'city', 0, 'pass'],
        ['A06', 2, 5, 45, 149, 'country', 0, 'fail'],
        ['A07',8, 9, 40, 155, 'city', 1, 'pass'],
        ['A08', 9, 10, 70, 155, 'city', 3, 'pass'],
        ['A09', 6, 12, 55, 154, 'city', 0, 'pass'],
        ['A10', 9, 2, 40, 156, 'city', 1, 'pass'],
        ['A11', 6, 10, 60, 153, 'city', 0, 'pass'],
        ['A12', 2, 4, 75, 151, 'country', 0, 'fail']
    ], columns=['ID', 'hour', 'attendance', 'weight', 'iq', 'region', 'library', 'pass'])

corr = df.corr()
print(corr)

sns.heatmap(corr, cmap='viridis')
plt.show()