# 콜백을 이용한 딥러닝 모델 관리 및 모니터링

## 실행 방법   
### 1. 코드 실행 : Run All  
<br>

### 2. tensorboard 실행   
작업 경로에서 아래와 같이 실행  
```
# 가상환경 .venv 실행
python3 -m venv .venv
source .venv/bin/activate

# 가상환경 확인
which python        # .../.venv/bin/python 이어야 정상

# 업그레이드 + tensorboard 설치
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade tensorboard

# tensorboard 실행
tensorboard --logdir=logs
```
<br>
<br>

### 3. tensorboard 접속  
로컬에서 `http://localhost:6006` 연결  

## 무엇을 하는 코드인가?  
iris 데이터를 3분류하는 Keras MLP 모델을 학습하고, 검증 정확도(`val_accuracy`)가 개선될 때마다 체크포인트 파일(`.h5`)을 저장하도록 설정한다.  
<br>

입력 4개 → 은닉 50 → 은닉 30 → 출력 3(softmax) 구조의 신경망 분류 모델 정의하고, `Adam` + `sparse_categorical_crossentropy`로 학습한다.  
훈련/검증 분할(validation_split=0.2)로 에폭마다 검증 성능을 측정하여, `ModelCheckpoint`로 검증 정확도 최고치 갱신 시마다 모델 파일을 저장한다.  
<br>

이 코드는 앞서 학습하고 성능 측정하여 저장한 모델 파일들을, `model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=50, callbacks=[checkpoint])` 로 아래와 같이 실행한다.  
- 학습 데이터를 80%(훈련) / 20%(검증)으로 내부 분할  
- 배치 50으로 최대 100에폭 학습  
- 각 에폭 종료 시 검증 정확도 계산하여, 이전 최고치보다 좋으면 체크포인트 저장  
<br>
<br>

## 코드 상세 설명  
### 1 cell
**모델 정의**  
units(=뉴런 수) : 층의 출력 차원이다. 마지막 층 `softmax`는 클래스별 확률을 출력  
<br>

### 2 cell  
**컴파일(학습 규칙)**  
`optimizer='adam'` : 적응형 학습률 옵티마이저  
`loss='sparse_categorical_crossentropy` : 정수 레이블(0/1/2)에 적합  
`metrics=['accuracy']` : 정확도 로그/평가  
<br>  

### 3 cell  
**데이터 준비**  
전체 150개 샘플을 `학습 120 / 테스트 30`으로 분리 (8:2)  
<br>  

### 4 cell
**콜백**  
`ModelCheckpoint`
- 파일명 예 : model-07-0.9667.h5 (7에폭, val_accuracy=0.9667)  
- save_best_only=True : 이전 최고 val_accuracy보다 좋아질 때만 저장  
- 파일명이 에폭/정확도를 포함하므로 개선될 때마다 새 파일 생성  
<br>

`EarlyStopping` (현재 미사용 / 참고사항)
```
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
```
- patience=20 : 20에폭 연속 향상 없으면 학습 중단 / 보통 `restore_best_weights=True`와 함께 사용  
<br>  

### 5 cell  
**콜백**  
`TensorBoard`
- *logs/* 폴더에 학습 로그(스칼라/히스토그램 등) 기록  
- 실제 기록하려면 fit의 callbacks에 포함해야 한다.  
<br>  

### 6 cell 
**학습 실행**
- `validation_split=0.2` : `X_train/y_train`의 **마지막 20%**를 검증 세트로 사용  
  - Keras는 fit 호출 시 분할하며, 이후 훈려용/검증용을 고정  
- `callbacks=[checkpoint]` : 에폭마다 `val_accuracy` 개선 시 모델 저장  
- EarlyStopping과 TensorBoard도 쓰려면 callbacks를 아래와 같이 정의  
  - `callbacks=[checkpoint, early_stopping, tensorboard]`
<br>
<br>
<br>

## 포인트❗
###### 왜 sparse_categorical_crossentropy?  
y가 원-핫이 아니라 정수 레이블이기 때문  
<br>

###### 검증 정확도는 어떻게 계산?  
validation_split로 떼어낸 검증 세트에서 매 에폭 후 계산 → val_accuracy  
<br>

###### 체크포인트 포맷  
지금은 `.h5`로 저장.  
나중에 로딩도 `.h5`로 (`load_model('file.h5')`)  
Keras3 권장 포맷은 `.keras`이므로, 통일을 원하면 파일명을 `.keras`로 바꾸고 로딩도 동일하게 한다.  
<br>

###### TensorBoard가 비어있다?
콜백을 `fit`에 넣지 않았거나, `logs/`에 이벤트 파일이 아직 없을 가능성이 크다.  
`callbacks`에 `tensorboard`를 추가하고 학습을 다시 돌린 뒤, `python -m tensorboard --logdir=logs`로 실행한다.