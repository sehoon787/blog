import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def make_data(size=100, noise=1):
    x = np.linspace(-5, 11, size).reshape(100, 1)
    y = 3*x**2 + 3*x

    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise  # 노이즈 추가

    plt.scatter(x, y)
    plt.suptitle("Sample Data", size=24)
    plt.show()

    return x, yy

def poly(x):
    model = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = model.fit_transform(x)
    return x, x_poly

def LR(poly_x, x, y):
    SLR_model = LinearRegression()
    model = LinearRegression()

    SLR_model.fit(x, y)
    model.fit(poly_x, y)

    print("w1: ", model.coef_[0][0])
    print("w2: ", model.coef_[0][1])
    print("b: ",  model.intercept_[0])

    SLR_result = SLR_model.predict(x)
    result = model.predict(poly_x)

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y)
    plt.plot(x, result, color='red', label='Polynomial Regression')
    plt.plot(x, SLR_result, color='green', label='Simple Linear Regression')
    plt.suptitle("LR function", size=24)
    plt.legend()
    plt.show()

    return result

x, y = make_data(size=100, noise=6)
x, x_poly = poly(x)
result = LR(x_poly, x, y)
data = np.concatenate((x, y, result), axis=1)
df = pd.DataFrame(data, columns=['x', 'y', 'predict'])

print("결정계수: ", r2_score(y, result))
print("상관계수: \n", df.corr())
print("MSE: ", mean_squared_error(y, result))
