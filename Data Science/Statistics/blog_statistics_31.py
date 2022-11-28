from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터셋 로드
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris.data])


def normalize(df):
    # 데이터셋 정규화
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)

    # type casting with setting target
    df_scaled = pd.DataFrame(df_scaled, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    df_scaled['target'] = iris.target

    print(df_scaled)

    return df_scaled


def draw_plot(df_scaled):
    # 2차원으로 차원 축소, target 정보는 제외
    pca = PCA(n_components=2)
    pca.fit(df_scaled.iloc[:, :-1])

    # 데이터 프레임으로 자료형 변환 및 target class 정보 추가
    df_pca = pca.transform(df_scaled.iloc[:, :-1])
    print(df_pca)
    print(pca.explained_variance_ratio_)

    df_pca = pd.DataFrame(df_pca, columns=['component 0', 'component 1'])

    # class target 정보 불러오기
    df_pca['target'] = df_scaled['target']

    # target 별 분리
    df_pca_0 = df_pca[df_pca['target'] == 0]
    df_pca_1 = df_pca[df_pca['target'] == 1]
    df_pca_2 = df_pca[df_pca['target'] == 2]

    # target 별 시각화
    plt.scatter(df_pca_0['component 0'], df_pca_0['component 1'], color='orange', alpha=0.7, label='setosa')
    plt.scatter(df_pca_1['component 0'], df_pca_1['component 1'], color='red', alpha=0.7, label='versicolor')
    plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color='green', alpha=0.7, label='virginica')

    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()
    plt.show()


def pca_test(n=4):
    X = iris.data
    y = iris.target

    # 2차원으로 차원 축소, target 정보는 제외
    pca = PCA(n_components=n)
    pca.fit(X)

    # 데이터 프레임으로 자료형 변환 및 target class 정보 추가
    df_pca = pca.transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=0, multi_class='multinomial')

    # origin
    if X.shape[1] > n:
        X = X[:, :n]
    clf.fit(X, y)
    pred = clf.predict(X)
    print(confusion_matrix(y, pred))

    # pca
    clf.fit(df_pca, y)
    pred = clf.predict(df_pca)
    print(confusion_matrix(y, pred))


draw_plot(normalize(df))
pca_test(n=4)
pca_test(n=2)
