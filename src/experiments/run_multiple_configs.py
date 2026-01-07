import csv
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PYTHON = sys.executable

EXPERIMENTS_DIR = Path("experiments")
CHECKPOINTS_DIR = Path("checkpoints")


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    args: List[str]
    config: Dict[str, Any]


def clean_string(v):
    s = str(v)
    s = s.replace("/", "-").replace(" ", "").replace("_", "")
    s = s.replace(".", "p")
    
    return s


def build_run_name(cfg):
    return (
        f"_LR{clean_string(cfg['learning_rate'])}"
        f"_E{cfg['num_epochs']}"
        f"_WD{clean_string(cfg['weight_decay'])}"
        f"_S{cfg['seed']}"
    )


def run_training(spec):
    cmd = [
        PYTHON,
        "-m",
        "src.models.train_liar_roberta",
        "--run_name",
        spec.run_name,
        *spec.args,
    ]

    subprocess.run(cmd, check=True)


def load_experiment_json(run_name):
    path = EXPERIMENTS_DIR / f"{run_name}.json"
    
    if not path.exists():
        raise FileNotFoundError(f"Expected results file not found: {path}. ")
        
    return json.loads(path.read_text(encoding="utf-8"))


def get_score(exp, metric_key):
    metrics = exp.get("test_metrics", {})
    
    if metric_key not in metrics:
        raise KeyError(
            f"Metric '{metric_key}' not found in test_metrics"
        )
        
    return float(metrics[metric_key])


def pick_best_run(run_names, metric_key):
    best_name = None
    best_score = float("-inf")
    best_exp = None

    for name in run_names:
        exp = load_experiment_json(name)
        score = get_score(exp, metric_key)
        
        if score > best_score:
            best_score = score
            best_name = name
            best_exp = exp

    if best_name is None or best_exp is None:
        raise ValueError("Could not find a best run.")

    return best_name, best_score, best_exp


def save_best_summary(best_name, best_score, best_exp, metric_key):
    out = EXPERIMENTS_DIR / "best_run.json"
    payload = {
        "selected_by": metric_key,
        "best_run_name": best_name,
        "best_score": best_score,
        "best_config": best_exp.get("config", {}),
        "best_test_metrics": best_exp.get("test_metrics", {}),
    }
    
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    
    return out


def promote_best_model(best_run_name):
    src = CHECKPOINTS_DIR / best_run_name / "best_model"
    
    if not src.exists():
        print(f"Cannot promote model: {src} does not exist")
        return None

    dst = CHECKPOINTS_DIR / "BEST" / "best_model"
    
    if dst.exists():
        shutil.rmtree(dst)
        
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    
    return dst


def save_leaderboard(rows):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    (EXPERIMENTS_DIR / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        csv_path = EXPERIMENTS_DIR / "leaderboard.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def generate_specs(search_space, fp16):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    specs: List[RunSpec] = []

    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        run_name = build_run_name(cfg)

        args = [
            "--learning_rate", str(cfg["learning_rate"]),
            "--num_epochs", str(cfg["num_epochs"]),
            "--weight_decay", str(cfg["weight_decay"]),
            "--seed", str(cfg["seed"]),
        ]
        
        if fp16:
            args.append("--fp16")

        specs.append(RunSpec(run_name=run_name, args=args, config={**cfg, "fp16": fp16}))

    return specs

def main():

    metric_key = "eval_macro_f1"

    search_space = {
        "learning_rate": [2e-5, 3e-5],
        "num_epochs": [3, 4, 5],
        "weight_decay": [0.02, 0.01],
        "seed": [42],
    }

    specs = generate_specs(search_space, True)

    for spec in specs:
        run_training(spec)

    leaderboard: List[Dict[str, Any]] = []
    run_names = [s.run_name for s in specs]

    best_name, best_score, best_exp = pick_best_run(run_names, metric_key=metric_key)

    for spec in specs:
        exp = load_experiment_json(spec.run_name)
        row = {
            "run_name": spec.run_name,
            "metric": metric_key,
            "score": float(exp["test_metrics"].get(metric_key, float("nan"))),
            "eval_accuracy": float(exp["test_metrics"].get("eval_accuracy", float("nan"))),
            "eval_loss": float(exp["test_metrics"].get("eval_loss", float("nan"))),
            **spec.config,
        }
        leaderboard.append(row)

    leaderboard.sort(key=lambda r: r["score"], reverse=True)
    
    save_leaderboard(leaderboard)
    save_best_summary(best_name, best_score, best_exp, metric_key=metric_key)
    promote_best_model(best_name)
    
if __name__ == "__main__":
    main()
