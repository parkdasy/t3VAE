import pandas as pd
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt

spy = fdr.DataReader('SPY', start='2014-01-01', end='2024-01-01')
# print(spy.head())
# print(type(spy))

# matrix = spy.to_numpy()
# open = matrix[:,1]

close_pct_change = spy['Close'].pct_change().to_numpy()
open_pct_change = spy['Open'].pct_change().to_numpy()

# heatmap_data = np.column_stack((open_pct_change, close_pct_change))

# plt.hist(close, bins=70)
# plt.savefig("histogram0.png")

# plt.hist(open, bins=70)
# plt.savefig("histogram2.png")



############### 2차원 heatmap 확인 ###############

# spy['Open_pct_change'] = spy['Open'].pct_change()
# spy['Close_pct_change'] = spy['Close'].pct_change()

# # Open과 Close의 백분율 변화로 2D 히스토그램 데이터 생성
# x = spy['Open_pct_change'].dropna()
# y = spy['Close_pct_change'].dropna()

# # 히트맵 생성
# plt.figure(figsize=(10, 8))
# hist, xedges, yedges, im = plt.hist2d(x, y, bins=50, cmap='coolwarm')

# # 컬러바 추가
# plt.colorbar(im, label="Frequency")

# # 축 레이블 및 제목 추가
# plt.xlabel("Open Percentage Change")
# plt.ylabel("Close Percentage Change")
# plt.title("2D Heatmap of Open vs Close Percentage Changes")

# plt.savefig("heatmap.png")


############### 2차원 heatmap 가운데 zoom in ###############

# spy['Open_pct_change'] = spy['Open'].pct_change()
# spy['Close_pct_change'] = spy['Close'].pct_change()

# # 필터링: 절댓값이 0.04 이내인 데이터만 사용
# filtered_data = spy[(spy['Open_pct_change'].abs() <= 0.02) & (spy['Close_pct_change'].abs() <= 0.02)]

# x = filtered_data['Open_pct_change'].dropna()
# y = filtered_data['Close_pct_change'].dropna()

# # 히트맵 생성
# plt.figure(figsize=(10, 8))
# hist, xedges, yedges, im = plt.hist2d(x, y, bins=50, cmap='coolwarm')

# # 컬러바 추가
# plt.colorbar(im, label="Frequency")

# # 축 범위 설정 (Zoom-In)
# plt.xlim(-0.02, 0.02)
# plt.ylim(-0.02, 0.02)

# # 축 레이블 및 제목 추가
# plt.xlabel("Open Percentage Change")
# plt.ylabel("Close Percentage Change")
# plt.title("2D Heatmap (|Change| <= 0.02)")

# plt.savefig("heatmap0.png")





############### hill ###############
data = close_pct_change[close_pct_change>0]
data = np.log(np.sort(data)[::-1])

n = len(data)
alpha = 1 / (sum(data[0:n-1])/(n-1) - data[n-1])
print(alpha)



k = n - 1
alpha = 1 / (np.cumsum(data[:k]) / np.arange(1, k + 1) - data[1:k + 1])
print(alpha)
