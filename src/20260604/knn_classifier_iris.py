from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
# print(iris['data'][:5])
# print(iris['feature_names'])
# print(iris['target'][:5])
iris_data = np.column_stack( (iris['data'], iris['target']) )
print(iris_data)
# 문제
# 위 데이터를 바탕으로 Dataframe을 설계(구현)
# 컬럼명 ==> sepal_len , sepal_wid , petal_len, petal_wid , target
iris_df = pd.DataFrame(iris_data, 
                       columns=['sepal_len','sepal_wid','petal_len','petal_wid','target'])
print(iris_df.head(10))

# plt.scatter(iris_df['sepal_len'], iris_df['sepal_wid'])
# plt.savefig('iris.jpeg')

# iris 붓꽃 KNN 분류 모델에서  모델입력 특성데이터(train_x) 는
# 'petal_len, 'petal_wid' 2개만 사용
# 모델 타깃데이터(train_y) ==> 'target' 사용
iris_train_x = iris_df[['petal_len','petal_wid','target']].copy()#.values
print(iris_train_x)

# for i in range(3):
#     plt.scatter( iris_train_x.loc[iris_train_x['target'] == i , :]['petal_len'],
#                 iris_train_x.loc[iris_train_x['target'] == i , :]['petal_wid'] )

# plt.savefig('iris.jpeg')
# KNN  모델 준비 ( k= 5 디폴트 사용 )
knnmodel = KNeighborsClassifier( n_neighbors= 5 ) # 객체화가 KNN 모델 준비 , k = 5 디폴트

# knn 학습
knnmodel.fit( iris_train_x[['petal_len','petal_wid']].values,
              iris_train_x['target'].values)

# knn 성능 평가
print('acc : ', knnmodel.score( iris_train_x[['petal_len','petal_wid']].values,
              iris_train_x['target'].values) )

# 새로운 데이터  붓꽃 분류 ( 예측 )
pred = knnmodel.predict([[5.9,2.3],[3.4, 1.8],[5.4, 2.2]])
print(pred) #  [2 1 2] ==> ['virginica' , 'versicolor', 'virginica']
#print(iris.keys())
print(iris['target_names'])
for x in pred:
    print(iris['target_names'][int(x)])


new_pred = knnmodel.predict([[5.0,1.7]])
print(new_pred)
# petal_len ==> 5.9,  petal_wid ==> 2.3 인 위치를
# scatter(), '^' 마커로 출력, 색상은 blue로
# 동시에 위 scatter으로 출력한 모든 데이터 위에 5.9, 2.5 위치를 출력

for i in range(3):
    plt.scatter( iris_train_x.loc[iris_train_x['target'] == i , :]['petal_len'],
                iris_train_x.loc[iris_train_x['target'] == i , :]['petal_wid'] )

plt.scatter(5.0, 1.7, marker='^' , c='blue'  )

plt.savefig('iris.jpeg')