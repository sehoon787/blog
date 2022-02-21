import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def make_data(w=0.5, b=0.8, size=50, noise=1.0):
    x = np.arange(size)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise  # 노이즈 추가

    return x, yy

def make_linear(w=0.5, b=0.8, size=50, noise=1.0):
    x, yy = make_data(w=w, b=b, size=size, noise=noise)
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='r')
    plt.scatter(x, yy, label='data')
    plt.legend(fontsize=20)
    plt.title(label=f'y = {w}*x + {b}')
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy

# a=기울기, b=절편
beta = 0.8
alpha = 2

x, y = make_linear(size=100, w=beta, b=alpha, noise=6)
# y[5]=60     # noise
# y[10]=60    # noise
x_bar = x.mean()
y_bar = y.mean()
calculated_weight = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar)**2).sum()
print('w: {:.2f}'.format(calculated_weight))
calculated_bias = y_bar - calculated_weight*x_bar
print('b: {:.2f}'.format(calculated_bias))

def OLS(x, y):
    df = pd.DataFrame(x, columns=["x"])
    df['y'] = y
    df['intercept'] = 1
    model = sm.OLS(df['y'], df[['intercept', 'x']])
    results = model.fit()
    print(results.summary())

    res = results.params['x']*x + results.params['intercept']
    plt.figure(figsize=(10, 7))
    plt.plot(res, color='red')
    plt.scatter(x, y)
    plt.suptitle("OLS")
    plt.title('y = ' + str(round(results.params['x'], 3))+'*x+' + str(round(results.params['intercept'], 3)))
    plt.show()

    return results.params['x'], results.params['intercept']

def LR(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    res = model.coef_[0]*x + model.intercept_
    print('w: {:.2f}, b: {:.2f}'.format(model.coef_[0], model.intercept_))

    plt.figure(figsize=(10, 7))
    plt.plot(res, color='red')
    plt.scatter(x, y)
    plt.suptitle("LR")
    plt.title('y = ' + str(round(model.coef_[0], 3))+'*x+' + str(round(model.intercept_, 3)))
    plt.show()
    return model.coef_[0], model.intercept_

x, y = make_data(size=100, w=beta, b=alpha, noise=6)
OLS(x, y)
LR(x, y)
