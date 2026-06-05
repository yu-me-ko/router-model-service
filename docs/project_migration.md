# Project Migration

`router-model-service` can continue running as an independent repository.

It can also be copied into the BombGPT project under:

```text
BombGPT/
  backend/
  frontend/
  services/
    router-model-service/
```

## Copy To BombGPT

Example Windows copy command:

```powershell
mkdir F:\output\BombGPT\services
xcopy F:\output\router-model-service F:\output\BombGPT\services\router-model-service /E /I /Y
```

If the service is copied into the BombGPT main repository, remove the nested Git repository metadata:

```powershell
cd F:\output\BombGPT\services\router-model-service
rmdir /S /Q .git
```

Do not copy `.venv/` or `.venv_gpu/`. Recreate the virtual environment after migration:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Start Router

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```

The SpringBoot backend should continue calling:

```text
http://127.0.0.1:9000/route
```

Changing the project directory does not change the interface address as long as the port remains the same.

## GitHub Commit Checklist

Before staging changes:

```powershell
git status
```

Do not commit:

- `.venv/`
- `.venv_gpu/`
- `outputs/`
- Large log files
- Temporary backup models

Recommended to commit:

- `app/`
- `scripts/`
- `data/router_eval_cases.csv`
- `data/router_challenge_eval_cases.csv`
- `data/router_hard_train_cases.csv`
- `data/router_label_map.json`
- `docs/`
- `README.md`
- `requirements.txt`
- `saved_models/router_distilbert` only after confirming the size is acceptable or Git LFS is enabled
