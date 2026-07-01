import pandas as pd
import numpy as np

df = pd.DataFrame([ {'height':185,'weight':60},
{'height':180,'weight':60},
{'height':185,'weight':70},
{'height':165,'weight':63},
{'height':155,'weight':68},
{'height':170,'weight':75},
{'height':175,'weight':80}]
                  )
print(df)
from sklearn.cluster import KMeans
kmm = KMeans(n_clusters=3).fit(df.values)

print(kmm.n_iter_)  # 수행된 이동 횟수
print(kmm.labels_)   # 각 데이터 포인트가 속한 군집 중심점 레이블
print(kmm.cluster_centers_)  # 각 클러스터 중심점 좌표
df['cluster_id'] = kmm.labels_
#print(df)
# 클러스터 중심좌표
cluster_df = pd.DataFrame(kmm.cluster_centers_, columns=['height','weight'])
cluster_df['label'] = [0,1,2]
print(cluster_df)

import matplotlib.pyplot as plt
import seaborn as sns


sns.lmplot(x='height',y='weight', data=df, fit_reg=False,
           scatter_kws={'s':150},
           hue='cluster_id')

#fig, ax = plt.subplots(figsize=(15,7))
# sns.scatterplot(x='height',y='weight', data=df, ax=ax)
# sns.scatterplot(x='height',y='weight', data=cluster_df,ax=ax)

plt.show()