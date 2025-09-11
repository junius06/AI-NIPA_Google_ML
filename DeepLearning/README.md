# 수업 환경  
| OS                             | python  | pip  | developer tools         |
|:------------------------------:|:-------:|:----:|:-----------------------:|
| Windows11<br>wsl (ubuntu22.04) | 3.12.10 | 25.2 | VsCode, PyCharm, Cursor |

## Python 설치
```
sudo apt -y update
sudo apt -y install 
```

## 가상환경 설치 & 활성화

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