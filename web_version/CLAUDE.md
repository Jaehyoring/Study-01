# Created: 2026-02-16 22:30

# Web Version — Handwritten Digit Recognizer

## Overview
Flask 기반 웹 앱. 브라우저에서 캔버스에 숫자를 그리면 SVM 모델이 예측합니다.

## Architecture
- `digit_recognizer.py` — Flask 서버 + 인라인 HTML/JS/CSS + SVM 예측
- 모델 파일(`digit_model.pkl`)은 프로젝트 루트(`../digit_model.pkl`)에 위치

## Running
```bash
./run.sh    # http://localhost:5000
```

## Dependencies
`numpy`, `Pillow`, `flask`, `scikit-learn`, `joblib`
