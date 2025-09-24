# 와인 데이터 분류 모델

## 📌 전체 코드 흐름 요약
| seq |   title   | description                         |
| --- | --------- | ----------------------------------- |
|  1  | 모델 정의  | 다층 퍼셉트론 (MLP) 구조, dropout 포함. |
|  2  | 데이터 준비 | 와인 데이터셋 불러오기, 입력/출력 분리. |
|  3  | 데이터 분할 | 학습/테스트 셋 분리. |
|  4  | 모델 컴파일 | 손실 함수, 옵티마이저, 평가지표 설정. |
|  5  | 모델 학습   | 300 epoch 동안 학습. |
|  6  | 성능 평가   | 테스트 데이터로 정확도 측정. |
|  7  | 시각화     | 학습 과정(loss, accuracy)을 그래프로 표시. |
<br>

## 1 cell - 모델 정의
Sequential 를 사용하여 층을 순서대로 쌓아간다.  
Input(shape=(11,)): 입력 데이터는 11개의 특성(feature)을 가짐 (와인 데이터셋의 11개 화학 성분).  
Dense(100, activation='sigmoid'): 첫 번째 은닉층, 뉴런 100개, 활성화 함수는 sigmoid.  
Dropout(0.2): 과적합 방지를 위해 20% 뉴런 무작위 비활성화.  
Dense(200, activation='relu'): 두 번째 은닉층, 뉴런 200개, ReLU 활성화.  
Dropout(0.4): 40% 드롭아웃.  
Dense(50, activation='tanh'): 세 번째 은닉층, 뉴런 50개, tanh 활성화.  
Dropout(0.1): 10% 드롭아웃.  
Dense(10, activation='softmax'): 출력층, 10개의 뉴런 (와인 품질 점수가 0~9까지 있다고 가정). 다중 클래스 확률 분포 출력.  
<br>

## 2 cell - 데이터 불러오기
`winequality-red.csv` 데이터 가져오기.  
X: 마지막 열(품질 점수)을 제외한 입력 특성 11개.  
y: 마지막 열, 와인의 품질 점수 (1~10 중 하나).  
<br>

## 3 cell - 학습 데이터 분리
train_test_split: 데이터셋을 학습용(훈련) 과 테스트용(검증) 으로 분리.  
기본값은 75% 훈련, 25% 테스트.  
<br>

## 4 cell - 모델 컴파일
loss = sparse_categorical_crossentropy: 다중 분류 손실 함수. (레이블이 원-핫 인코딩되지 않고 정수 형태일 때 사용)  
optimizer = adam: 가중치 최적화를 위한 알고리즘.  
metrics = accuracy: 학습 과정에서 정확도를 함께 계산.  
<br>

## 5 cell - 모델 학습
X_train, y_train 으로 모델 학습.  
batch_size=200: 한 번에 200개의 샘플로 경사 하강법 진행.  
epochs=300: 데이터셋을 300번 반복 학습.  
history: 손실값과 정확도가 기록됨 (그래프 그릴 때 사용).  
<br>

## 6 cell - 모델 평가
학습이 끝난 후 테스트 데이터에서 손실값과 정확도를 평가.  
결과적으로 모델의 일반화 성능을 확인.  
<br>

## 7 cell - 학습 과정 시각화
손실(loss) 과 정확도(accuracy) 변화를 한 그래프에 표시.  
ax1: 손실곡선 (파란색 실선).  
ax2: 정확도 곡선 (빨간색 점선).  
학습이 진행됨에 따라 손실이 감소하고 정확도가 증가하는지 확인할 수 있음.  
<br>

!(Wine Quality)[https://archive.ics.uci.edu/dataset/186/wine+quality]