import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('CrabAgePrediction.csv')

sns.pairplot(df[['Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']])
plt.show()

X_train = df[['Height', 'Shell Weight']]
def feature_engineering_XbyVIF(X_train):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X_train.values, i)
                         for i in range(X_train.shape[1])]
    vif['Feature'] = X_train.columns
    return vif
vif = feature_engineering_XbyVIF(X_train)
print(vif)

df['intercept'] = 1
model = sm.OLS(df['Age'], df[['intercept', 'Height', 'Shell Weight']])
results = model.fit()
print(results.summary())