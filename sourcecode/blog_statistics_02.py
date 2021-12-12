from scipy import stats

data1 = [408, 432, 438, 495, 530]      # 관측치
data2 = [300, 320, 340, 380, 450]      # 기대치
result1 = stats.chisquare(data1, data2)
print("statistic =", round(result1[0], 3), "p-value =", round(result1[1], 3))

data3 = [40, 43, 43, 49, 53]      # 관측치
data4 = [40, 40, 40, 40, 40]      # 기대치
result2 = stats.chisquare(data3, data4)
print("statistic =", round(result2[0], 3), "p-value =", round(result2[1], 3))
