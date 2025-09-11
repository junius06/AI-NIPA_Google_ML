# 수업시 진행 환경  
| OS                             | python  | pip  | developer tools         |
|:------------------------------:|:-------:|:----:|:-----------------------:|
| Windows11<br>wsl (ubuntu22.04) | 3.12.10 | 25.2 | VsCode, PyCharm, Cursor |

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
```