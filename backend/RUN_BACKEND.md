# Backend Run Guide

## Install Dependencies

Run from `E:\Multi-Model-Fatigue-Detection`:

```powershell
pip install -r backend/requirements.txt
```

## Run in Development

```powershell
cd backend
$env:PORT = 8000
python app.py
```

## Run in Production

```powershell
cd backend
gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$env:PORT app:app
```

## API

- `GET /` health check
- `POST /predict` with multipart form field `image`

Example request:

```powershell
curl -X POST "http://localhost:8000/predict" -F "image=@sample.jpg"
```
