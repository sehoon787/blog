import pandas as pd
from math import log

def tf(t, d):
    return d.count(t)

def df(t):
    df = 0
    for doc in docs:
        df += t in doc
    return df

def idf(t):
    return log(n/(df(t)+1))

def tf_idf(t, d):
    return tf(t, d) * idf(t)

docs = [
    '맛있는 빨간 딸기',
    '비싼 빨간 딸기',
    '동그랗고 빨간 체리',
    '맛있고 비싸고 빨간 딸기 딸기',
    '과일은 너무 비싸다']

voca = list(set(w for doc in docs for w in doc.split()))
voca.sort()
print(voca)
# ['과일은', '너무', '동그랗고', '딸기', '맛있고', '맛있는', '비싸고', '비싸다', '비싼', '빨간', '체리']

n = len(docs)  # 문서 개수

# TF
tf_list = [[tf(voca[j], docs[i]) for j in range(len(voca))] for i in range(n)]
tf_res = pd.DataFrame(tf_list, columns=voca)

# IDF
idf_list = [idf(voca[j]) for j in range(len(voca))]
idf_res = pd.DataFrame(idf_list, index=voca, columns=["IDF"])

# TF-IDF
tf_idf_list = [[tf_idf(voca[j], docs[i]) for j in range(len(voca))] for i in range(n)]
tfidf_res = pd.DataFrame(tf_idf_list, columns=voca)