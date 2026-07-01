import numpy as np
np.set_printoptions(threshold=np.inf)

fruits = np.load('fruits_300.npy')
print(len(fruits))  # 총 300개 샘플 : 3차원배열 ( 심플개수, 너비, 높이 )
# 한개 과일 샘플에 너비 * 높이 pixel = 100 * 100
print(len(fruits[:1][0]))
print(len(fruits[:1][0][0]))

# k-평균 모델훈련을 위해 2차원배열로 변환
fruits_2D = fruits.reshape(-1,100*100)
print(fruits_2D.shape)

# k-평균 모델 훈련
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

inertia = []
for k in range(2,7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2D)
    inertia.append(km.inertia_) # 이너셔 : 클러스터 중심과 클러스터에 속한 샘플 사이의 거리의 제곱 합

# 이너셔가 작을수록 중심점에 잘 군집됬다고 판단하며
# 클러스터의 개수를 늘려도 밀집된 정도가 개선되지 않은 시점(엘보우 방법)을
# 최적의 k 로  판단
plt.plot(range(2,7), inertia)
plt.show()