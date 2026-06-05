import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_MODEL_DIR = ROOT_DIR / "outputs" / "auto_train_gpu_full_seed3407" / "best_model"
DEFAULT_TARGET_MODEL_DIR = ROOT_DIR / "saved_models" / "router_distilbert"


def validate_model_dir(model_dir: Path) -> list[str]:
    missing = []
    if not model_dir.exists():
        return [f"model directory does not exist: {model_dir}"]

    required_files = ["config.json", "router_label_map.json"]
    for filename in required_files:
        if not (model_dir / filename).exists():
            missing.append(filename)

    if not ((model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()):
        missing.append("model.safetensors or pytorch_model.bin")

    if not (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "vocab.txt").exists()
        or (model_dir / "tokenizer_config.json").exists()
    ):
        missing.append("tokenizer.json or vocab.txt or tokenizer_config.json")

    return missing


def run_eval(model_dir: Path, eval_path: Path | None = None) -> bool:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "evaluate_router_cases.py"),
        "--model_dir",
        str(model_dir),
    ]
    if eval_path is not None:
        command.extend(["--eval_path", str(eval_path)])

    completed = subprocess.run(command, cwd=ROOT_DIR, text=True)
    return completed.returncode == 0


def copy_model(source_model_dir: Path, target_model_dir: Path, backup_dir: Path) -> None:
    if target_model_dir.exists():
        if backup_dir.exists():
            raise FileExistsError(f"backup_dir already exists: {backup_dir}")
        shutil.copytree(target_model_dir, backup_dir)
        shutil.rmtree(target_model_dir)

    shutil.copytree(source_model_dir, target_model_dir)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_backup_dir = ROOT_DIR / "saved_models" / f"router_distilbert_backup_{timestamp}"

    parser = argparse.ArgumentParser(description="Promote a validated router best_model to saved_models/router_distilbert.")
    parser.add_argument("--source_model_dir", default=str(DEFAULT_SOURCE_MODEL_DIR))
    parser.add_argument("--target_model_dir", default=str(DEFAULT_TARGET_MODEL_DIR))
    parser.add_argument("--backup_dir", default=str(default_backup_dir))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_model_dir = Path(args.source_model_dir)
    target_model_dir = Path(args.target_model_dir)
    backup_dir = Path(args.backup_dir)

    missing = validate_model_dir(source_model_dir)
    if missing:
        print("ERROR: source_model_dir is not a valid router model directory.")
        for item in missing:
            print(f"  missing: {item}")
        return 1

    print("Promoting router model")
    print(f"source_model_dir={source_model_dir}")
    print(f"target_model_dir={target_model_dir}")
    print(f"backup_dir={backup_dir}")

    copy_model(source_model_dir, target_model_dir, backup_dir)
    print("Model copied successfully.")
    print(f"Previous model backup kept at: {backup_dir}")

    ordinary_ok = run_eval(target_model_dir)
    challenge_ok = run_eval(target_model_dir, ROOT_DIR / "data" / "router_challenge_eval_cases.csv")

    if ordinary_ok and challenge_ok:
        print("PROMOTE BEST MODEL PASSED")
        return 0

    print("WARN: promoted model did not pass all evaluations.")
    print(f"Backup was not deleted: {backup_dir}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
