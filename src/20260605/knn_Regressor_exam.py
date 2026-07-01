import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test 분리
from sklearn.neighbors import KNeighborsRegressor # 특정값을 예측하는 회귀모델

#농어길이데이터 ( 캐글Fish Market 데이터참조)
perch_length=np.array([8.4,13.7,15.0,16.2,17.4,18.0,18.7,19.0,19.6,20.0,21.0,
21.0,21.0,21.3,22.0,22.0,22.0,22.0,22.0,22.5,22.5,22.7,
23.0,23.5,24.0,24.0,24.6,25.0,25.6,26.5,27.3,27.5,27.5,
27.5,28.0,28.7,30.0,32.8,34.5,35.0,36.5,36.0,37.0,37.0,
39.0,39.0,39.0,40.0,40.0,40.0,40.0,42.0,43.0,43.0,43.5,
44.0])
# 농어무게데이터 (캐글FishMarket 데이터참조)
perch_weight=np.array([5.9,32.0,40.0,51.5,70.0,100.0,78.0,80.0,85.0,85.0,110.0,
115.0,125.0,130.0,120.0,120.0,130.0,135.0,110.0,130.0,
150.0,145.0,150.0,170.0,225.0,145.0,188.0,180.0,197.0,
218.0,300.0,260.0,265.0,250.0,250.0,300.0,320.0,514.0,
556.0,840.0,685.0,700.0,700.0,690.0,900.0,650.0,820.0,
850.0,900.0,1015.0,820.0,1100.0,1000.0,1100.0,1000.0,
1000.0])


# 모델 입력 특성(x) ==> 농어의 길이
# 정답(타깃,y)  ==> 농어의 무게
print(len(perch_length)) # 56
print(len(perch_weight)) # 56

# train 데이터와 test 데이터로 분리
train_x, test_x, train_y, test_y =\
    train_test_split(perch_length, perch_weight, random_state=42)

print(len(train_x))
print(len(test_x))
print(train_x.shape)
print(test_x.shape)
# 1차원 shape ==> 2차원 shape 로 변경
train_x = train_x.reshape(-1,1) # -1 ==> 데이터의 개수만큼 알아서 shape 자동으로 지정
test_x = test_x.reshape(-1,1)
print(train_x.shape)
print(test_x.shape)

# knn 회귀 모델 준비
knn_reg = KNeighborsRegressor(n_neighbors=3)

# 모델 학습(훈련)
# train_x ==> 농어길이,  train_y ==> 농어무게
knn_reg.fit(train_x, train_y)

# 모델 성능 평가
print( knn_reg.score(test_x, test_y) ) # test 데이터에 대한 성능 0.97

print( knn_reg.score(train_x, train_y) ) # train 데이터에 대한 성능 0.98

# test 데이터를 이용해서 예측
test_pred = knn_reg.predict(test_x)
print(test_pred) # 14개 테스트 데이터의 길이에 따른 무게를 예측

# from sklearn.metrics import mean_absolute_error

# mae = mean_absolute_error(test_y, test_pred)  # 절대값 오차에대한 평균
# print('mae : ', mae)

# 농어의 길이 40인 농어의 무게를 예측?
pred = knn_reg.predict([[50]])
print(pred)  # 무게가 1033 g 예측

# 농어의 길이 80 ,  120 인 두 농어의 무게를 예측?
pred = knn_reg.predict( [ [80],[120] ] )
print(pred)

# 시각화
# train_x(길이), train_y(무게) 를 산점도 시각화
plt.scatter(train_x, train_y)

# 길이 50인 농어의 무게 예측해서 산점도로 출력 ( marker = '^', c='red' )
plt.scatter(50, 1033, marker='^', c='red')

plt.savefig('knn_reg.jpeg')