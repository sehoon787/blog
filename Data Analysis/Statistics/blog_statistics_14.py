import numpy as np

def cos_sim(A, B):
  return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

sentence1 = np.array([1, 1, 1, 0, 1])
sentence2 = np.array([1, 1, 0, 1, 1])
sentence3 = np.array([1, 2, 2, 0, 1])

print('문장1-문장2 코사인 유사도 :', cos_sim(sentence1, sentence2))
print('문장1-문장3 코사인 유사도 :', cos_sim(sentence1, sentence3))
print('문장2-문장3 코사인 유사도 :', cos_sim(sentence2, sentence3))

# 문장1-문장2 코사인 유사도 : 0.75
# 문장1-문장3 코사인 유사도 : 0.9486832980505138
# 문장2-문장3 코사인 유사도 : 0.6324555320336759