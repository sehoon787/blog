import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def logistic_regression(data, learning_rate=0.01, epochs=1000):
    w = 0.0
    b = 0.0
    for i in range(1, epochs+1):
        for x, y in data:
            w_difference = x*(sigmoid(w*x+b)-y)
            b_difference = sigmoid(w*x+b)-y

            w -= learning_rate*w_difference
            b -= learning_rate*b_difference

        if i%10==0:
            print('epoch =', i, ', w =', round(w, 3), ', b =', round(b, 3))
    print('epoch =', i, ', w =', round(w, 3), ', b =', round(b, 3))

    return w, b

n = 20
x_data = np.linspace(0, n, n, dtype=int).reshape(-1, 1)
y_data = np.array([1 for i in range(n)]).reshape(-1, 1)
y_data[:int(n/2)] = 0
data = np.append(x_data, y_data, axis=1)

w, b = logistic_regression(data=data)
plt.scatter(x_data, y_data)
plt.plot(x_data, sigmoid(w*x_data+b))
plt.show()

################################################################################################

import pandas as pd
from sklearn import datasets    # https://scikit-learn.org/stable/datasets/toy_dataset.html
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine_data = datasets.load_wine()
print(wine_data.feature_names)

# train, test 셋 분리
x = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = wine_data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# 로지스틱 회귀 모델 학습
model = LogisticRegression(penalty='l2', max_iter=100)
model.fit(x_train, y_train)

# 로지스틱 모델 학습 성능 비교
y_pred = model.predict(x_test)  # 예측 결과 라벨

# 정확도 측정
print(round(accuracy_score(y_pred, y_test), 3)*100, "%")
