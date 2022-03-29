import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

k = 8       # 성공 횟수
n = 10      # 시행 횟수

def pmf(n, k, p):
    return comb(n, k)*(p**k)*((1-p)**(n-k))

def likelihood(n, k, p):
    return p**k*((1-p)**(n-k))

# result = [pmf(n, k, p/10) for p in np.arange(0.1, 1, 0.1)]
result = [likelihood(n, k, p) for p in np.arange(0.1, 1, 0.1)]

sum = 0
print("n="+str(n)+', k='+str(k))
for i, val in enumerate(result):
    sum += val
    print(str((i+1)/10)+' : '+str(val))
print("sum : ", sum)

plt.bar(range(1, 10), result)
plt.title("n="+str(n)+', k='+str(k))
plt.xlabel("x/10")
plt.show()
