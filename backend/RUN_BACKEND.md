# Backend Run Guide

## One-time setup

Run from `E:\Fullstack\backend`:

```powershell
.\setup_backend.ps1
```

## Start FastAPI

```powershell
.\run_backend.ps1
```

## Start FastAPI with auto-reload

```powershell
.\run_backend.ps1 -Reload
```

## Notes

- The scripts use an isolated virtual environment at `backend\.venv`.
- This avoids global package conflicts from system Python.
- API base URL remains `http://localhost:5000`.
