# 수업시 진행 환경  
| OS                             | python  | pip  | developer tools         |
|:------------------------------:|:-------:|:----:|:-----------------------:|
| Windows11<br>wsl (ubuntu22.04) | 3.12.10 | 25.2 | VsCode, PyCharm, Cursor |
<br>

## 1. Python 설치
```
sudo apt -y update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt -y update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version

python3.12 -m ensurepip --upgrade
python3.12 -m pip install --upgrade pip setuptools wheel
python3.12 -m pip -V
```
<br>

## 2. 가상환경 설치 & 활성화

### 설치 & 활성화 명령어
```
python3 -m venv .venv
source .venv/bin/activate      # 활성화 (프롬프트에 (.venv) 표시)
```

위 명령어를 실행하면 현재 경로에 있는 `.venv` 디렉터리가 생성된다.  
<br>

### 비활성화 & 삭제 명령어
```
deactivate                     # 가상환경 해제
rm -rf .venv                   # 필요시 폴더 삭제
```
<br>

## 3. 필요한 라이브러리 설치
몇 가지의 라이브러리만 설치하는 것이므로, 차후 필요한 라이브러리는 설치하며 진행한다.
```
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install matplotlib
```
<br>

## 4. ipynb 확장자
Jupyter Notebook 파일 형식인데, 이것은 대화형 컴퓨팅 환경을 제공하는 오픈소스 웹 응용 프로그램이다.  
코드와 설명(문서), 실행 결과(표, 그래프, 텍스트)까지 한 파일에 담는 "실험용 노트"와 같은 역할을 한다.  

### 핵심 특징  
- 내부는 JSON으로 여러 개의 셀(cell)로 구성되어 있다. 코드 셀은 파이썬 등 커널에서 실행하고, 마크다운 셀은 설명/수식(LaTex)을 작성한다. 출력은 그래프, 이미지, 표, 텍스트가 셀 옆에 저장된다.  

### 장단점
**장점**
- 코드 + 설명 + 결과를 한 눈에 볼 수 있어 데모, 튜토리얼, EDA(탐색적 분석)에 최적화 되어 있다.  
- 손쉬운 시각화/데이터 확인, 대화형 위젯(ipywidgets) 지원한다.  

**단점**
- 재현성 : 셀을 뒤죽박죽 실행하면 상태가 꼬일 수 있다. → "Restart & Run All" 습관화를 권장한다.  
- 버전 관리 : 출력(이미지/대용량 base64)이 포함되어 diff/merge가 지저분할 수 있다.  