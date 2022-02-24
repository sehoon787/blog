import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def make_data(size=100, noise=1):
    x = 2*np.random.rand(size, 1)
    x2 = 3*np.random.rand(size, 1)
    y = 3*x+x2+np.random.randn(size, 1)

    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise  # 노이즈 추가

    return x, x2, yy

def MLR(x, y, epochs=5000, learning_rate=0.00001):
    x1 = x[:, 0]
    x2 = x[:, 1]

    w1 = 0.0
    w2 = 0.0
    b = 0.0

    n = len(x)

    for i in range(epochs):
        hypothesis = w1*x1+w2*x2+b
        cost = np.sum((hypothesis-y)**2)/n

        gradient_w1 = np.sum((w1*x1+w2*x2-y+b)*2*x1)/n
        gradient_w2 = np.sum((w1*x1+w2*x2-y+b)*2*x2)/n
        gradient_b = np.sum((w1*x1+w2*x2-y+b)*2)/n

        w1 -= learning_rate * gradient_w1
        w2 -= learning_rate * gradient_w2
        b -= learning_rate * gradient_b

        if i % 100 == 0:
            print('Epoch ({:10d}/{:10d}) cost: {:10f}, W1: {:10f}, W2: {:10f}, b:{:10f}'.
                  format(i, epochs, cost, w1, w2, b))

    result = (w1*x1+w2*x2+b).reshape(-1, 1)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y)
    ax.plot_surface(x1, x2, result, color="red")
    plt.suptitle("MLR", size=24)
    plt.title('w1=' + str(round(w1, 3))+'w2=' + str(round(w2, 3))+', b=' + str(round(b, 3)))
    plt.show()

    return w1, w2, b, result


def LR(x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]

    model = LinearRegression()
    model.fit(x, y)

    w1 = model.coef_[0][0]
    w2 = model.coef_[0][1]
    b = model.intercept_[0]
    result = (w1 * x1 + w2 * x2 + b).reshape(-1, 1)

    print('w1: ', model.coef_[0][0], ", w2: ", model.coef_[0][1], ", b:", model.intercept_[0])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y)
    ax.plot_surface(x1, x2, result, color="red")
    plt.suptitle("LR function", size=24)
    plt.title('w1=' + str(round(w1, 3)) + 'w2=' + str(round(w2, 3)) + ', b=' + str(round(b, 3)))
    plt.show()

    return w1, w2, b, res

x, x2, y = make_data(size=100, noise=6)
data = np.concatenate((x, x2), axis=1)
w1, w2, b, res = MLR(data, y)
# _, _, _, res = LR(data, y)
df = pd.DataFrame(x, columns=['x'])
df['y'] = y
df['predict'] = res

print("결정계수: ", r2_score(y, res))
print("상관계수: ", df.y.corr(df.predict))
print("MSE: ", mean_squared_error(y, res))
