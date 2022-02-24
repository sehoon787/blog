import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def make_data(size=100, noise=1):
    x = 2*np.random.rand(size, 1)
    y = 3*x+np.random.randn(size, 1)

    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise  # 노이즈 추가

    plt.scatter(x, yy)
    plt.suptitle("Sample Data", size=24)
    plt.show()

    return x, yy

def SLR(x, y, epochs=5000, learning_rate=0.01):
    w = 0.0
    b = 0.0

    n_data = len(x)

    for i in range(epochs):
        hypothesis = w*x+b
        cost = np.sum((hypothesis-y)**2)/n_data
        gradient_w = np.sum((w*x-y+b)* 2*x)/n_data
        gradient_b = np.sum((w*x-y+b)*2)/n_data

        w -= learning_rate * gradient_w
        b -= learning_rate * gradient_b

        if i % 100 == 0:
            print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, w, b))


    plt.figure(figsize=(10, 7))
    plt.scatter(x, y)
    plt.plot(x, w*x+b, color='red')
    plt.suptitle("SLR", size=24)
    plt.title('w=' + str(round(w, 3))+', b=' + str(round(b, 3)))
    plt.show()

    return w, b, w*x+b

def LR(x, y):
    model = LinearRegression()
    model.fit(x, y)
    res = model.coef_[0]*x + model.intercept_[0]
    print('w: ', model.coef_[0][0], ", b:", model.intercept_[0])

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y)
    plt.plot(x, res, color='red')
    plt.suptitle("LR", size=24)
    plt.title('w=' + str(np.round(model.coef_[0][0], 3))+', b=' + str(np.round(model.intercept_[0], 3)))
    plt.show()
    return model.coef_[0][0], model.intercept_[0], res

x, y = make_data(size=100, noise=6)
_, _, res = LR(x, y)
# w, b, res = SLR(x, y)
df = pd.DataFrame(x, columns=['x'])
df['y'] = y
df['predict'] = res

print("결정계수: ", r2_score(y, res))
print("상관계수: ", df.y.corr(df.predict))
print("MSE: ", mean_squared_error(y, res))
