# Created: 2026-02-16 22:30

# Desktop Version — Handwritten Digit Recognizer

## Overview
Tkinter 기반 데스크톱 앱. 캔버스에 숫자를 그리면 SVM 모델이 예측합니다.

## Architecture
- `digit_recognizer.py` — Tkinter GUI + PIL 캔버스 캡처 + SVM 예측
- 모델 파일(`digit_model.pkl`)은 프로젝트 루트(`../digit_model.pkl`)에 위치

## Running
```bash
./run.sh    # Tkinter 창이 열림
```

## Dependencies
`numpy`, `Pillow`, `scikit-learn`, `joblib`, `tkinter` (Python 표준 라이브러리)
