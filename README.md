# Router Model Service

BombGPT local router model service. The model predicts one of seven `workflow` labels, and `toolHints` are inferred by shared routing rules at prediction time.

## Workflow Labels

- `KNOWLEDGE_QA`
- `AGENT_TASK`
- `FILE_QA`
- `USER_KNOWLEDGE_QA`
- `DIRECT_AI`
- `SYSTEM_ACTION`
- `UNKNOWN`

## Data Design

`workflow` means execution flow, not a detailed life topic. Campus-life expressions such as "我饿了", "想吃饭了", "我想打球", and "我不舒服" are still campus public knowledge questions, so they are labeled `KNOWLEDGE_QA`.

The Excel `suggested_workflow` column is converted to `workflow`. The Excel `keywords` column is not used as `tools`; `tools` is inferred from `question + workflow` by the shared `infer_tool_hints` rules.

## Data Build

```powershell
python scripts/import_excel_dataset.py
python scripts/analyze_dataset.py
python scripts/check_router_dataset_quality.py
```

The latest Excel source path is:

```text
data/raw/BombGPT_路由训练数据_3118条.xlsx
```

`data/router_challenge_eval_cases.csv` is an independent challenge eval set and must not be merged into `data/router_train.csv`.

## Training

Do not overwrite the baseline model during training. A typical GPU auto-training command is:

```powershell
python scripts/train_auto.py --max_epochs 300 --patience 10 --batch_size 16 --seed 3407 --output_dir outputs/auto_train_gpu_full_seed3407
```

Evaluate a trained candidate:

```powershell
python scripts/evaluate_router_cases.py --model_dir outputs/auto_train_gpu_full_seed3407/best_model
python scripts/evaluate_router_cases.py --model_dir outputs/auto_train_gpu_full_seed3407/best_model --eval_path data/router_challenge_eval_cases.csv
```

## Production Release Flow

Recommended final model:

```text
outputs/auto_train_gpu_full_seed3407/best_model
```

Model file notes:

- If Git LFS is not used, avoid committing large model files directly to GitHub.
- If `saved_models/router_distilbert` is large, prefer generating it locally with the release script instead of committing it.
- If the model must be committed, use Git LFS for model weights and other large binary files.

Promote the selected model manually:

```powershell
python scripts/promote_best_model.py --source_model_dir outputs/auto_train_gpu_full_seed3407/best_model
```

`promote_best_model.py` backs up the current `saved_models/router_distilbert` to `saved_models/router_distilbert_backup_YYYYMMDD_HHMMSS`, copies the candidate model into `saved_models/router_distilbert`, and runs ordinary plus challenge evaluation. It does not delete backups.

Run release checks:

```powershell
python scripts/release_check_router.py
```

Start Router as a standalone service:

```powershell
cd router-model-service
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```

Test:

```powershell
python scripts\predict.py "给我讲个笑话"
python scripts\release_check_router.py
```

Route API smoke test:

```http
POST http://127.0.0.1:9000/route
Content-Type: application/json

{"question":"给我讲个笑话"}
```

Expected response fields:

- `question`
- `route`
- `workflow`
- `confidence`
- `toolHints`
- `correctedByRule`
- `ruleName`
- `rawWorkflow`
- `rawConfidence`

SpringBoot `/router/test` may additionally expose:

- `available`
- `message`

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

## Evaluation

Evaluate current production model:

```powershell
python scripts/evaluate_router_cases.py --model_dir saved_models/router_distilbert
python scripts/evaluate_router_cases.py --model_dir saved_models/router_distilbert --eval_path data/router_challenge_eval_cases.csv
```

Test one question:

```powershell
python scripts/predict.py "我现在饿了"
python scripts/predict.py "给我讲个笑话"
```

When `correctedByRule=true`, the final `confidence` is `max(rawConfidence, 0.99)`. The original model score remains available as `rawConfidence`.
