import numpy as np
np.set_printoptions(threshold=np.inf)  # 넘파이 데이터 생략 없이 모두출력

fruits = np.load('fruits_300.npy')  # 300개 과일 이미지 넘파이 배열 샘플 데이터
print(len(fruits))  # 총 300개 샘플 : 3차원배열 ( 심플개수, 너비, 높이 )
# 한개 과일 샘플에 너비 * 높이 pixel = 100 * 100
print(len(fruits[:1][0])) # 너비 길이
print(len(fruits[:1][0][0])) # 높이 길이

# k-평균 모델훈련을 위해 2차원배열로 변환
fruits_2D = fruits.reshape(-1,100*100)
print(fruits_2D.shape)

# k-평균 모델 훈련
from sklearn.cluster import KMeans
# n_clusters : 클러스터 개수 지정
# init='k-means++' 디폴트 설정
km = KMeans(n_clusters=3, random_state=42)
# 비지도학습으로 fit() 메서드 타깃 데이터 사용하지 않음
km.fit(fruits_2D)

# 군집 결과는 KMeans 클래스 labels_속성에 저장
# labels_길이는 샘플 개수 ( 300개 ) , 각 샘플이 어떤 레이블에 해당되는지 나타냄
# n_clusters=3 지정으로 labels_ 배열의 값은 0,1,2 중 하나
print(km.labels_)

# 레이블 0,1,2로 모은 샘플 개수 확인
print(np.unique(km.labels_, return_counts=True))

# 각 클러스터( 0, 1, 2 ) 이미지 그림으로 출력
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio = 1):
    n = len(arr) # 각 클러스터 샘플 개수
    # 한줄에 10개 이미지 출력위해 샘플 개수를 10으로 나눠 행 개수 계산
    rows = int( np.ceil(n/10) ) # ceil() : 입력값보다 같거나 큰 정수중 가장 가까운값 반환
    # 행이 1이면 열의 개수는 샘플 개수, 그 이상의 행이면 10 열
    cols = n if rows < 2 else 10
    # squeeze=False ==> 배열을 함축하지 말고 rows, cols 지정 형태 그대로 형성
    # 즉, (1,2) 일떄 (2,)처럼 함축하지 말고 (1,2) 형태로 형성
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)

    for row in range(rows):
        for col in range(cols):
            if row*10 + col < n: # n개 까지만 그림
                axs[row,col].imshow(arr[row*10+col], cmap='gray_r')
            axs[row,col].axis('off') # 모든 축과 눈금, 라벨 지워지고 그래프(이미지)만 출력
    plt.show()

draw_fruits(fruits[km.labels_==0])  # 불린배열 인덱싱 활용 0 클러스터 샘플 추출 (파인애플)
#draw_fruits(fruits[km.labels_==1])  # 불린배열 인덱싱 활용 1 클러스터 샘플 추출 (바나나)
#draw_fruits(fruits[km.labels_==2])  # 불린배열 인덱싱 활용 2 클러스터 샘플 추출 (사과)
