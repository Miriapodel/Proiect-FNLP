import csv
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PYTHON = sys.executable

EXPERIMENTS_DIR = Path("experiments")
CHECKPOINTS_DIR = Path("checkpoints")


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    args: List[str]
    config: Dict[str, Any]


def _slug(v: Any) -> str:
    s = str(v)
    s = s.replace("/", "-").replace(" ", "").replace("_", "")
    s = s.replace(".", "p")
    return s


def build_run_name(cfg: Dict[str, Any]) -> str:
    return (
        f"_LR{_slug(cfg['learning_rate'])}"
        f"_E{cfg['num_epochs']}"
        f"_WD{_slug(cfg['weight_decay'])}"
        f"_S{cfg['seed']}"
    )


def run_training(spec: RunSpec) -> None:
    cmd = [
        PYTHON,
        "-m",
        "src.models.train_liar_roberta",
        "--run_name",
        spec.run_name,
        *spec.args,
    ]

    print("\n" + "=" * 100)
    print(f"RUN: {spec.run_name}")
    print("CONFIG:", json.dumps(spec.config, indent=2))
    print("CMD:", " ".join(cmd))
    print("=" * 100)

    subprocess.run(cmd, check=True)


def load_experiment_json(run_name: str) -> Dict[str, Any]:
    path = EXPERIMENTS_DIR / f"{run_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected results file not found: {path}. "
            f"Did the training script save experiments/{run_name}.json?"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_score(exp: Dict[str, Any], metric_key: str) -> float:
    metrics = exp.get("test_metrics", {})
    if metric_key not in metrics:
        raise KeyError(
            f"Metric '{metric_key}' not found in test_metrics. Available keys: {list(metrics.keys())}"
        )
    return float(metrics[metric_key])


def pick_best_run(run_names: List[str], metric_key: str) -> Tuple[str, float, Dict[str, Any]]:
    best_name: Optional[str] = None
    best_score: float = float("-inf")
    best_exp: Optional[Dict[str, Any]] = None

    for name in run_names:
        exp = load_experiment_json(name)
        score = get_score(exp, metric_key)
        print(f"Run {name:>35} | {metric_key} = {score:.6f}")
        if score > best_score:
            best_score = score
            best_name = name
            best_exp = exp

    assert best_name is not None and best_exp is not None
    return best_name, best_score, best_exp


def save_best_summary(best_name: str, best_score: float, best_exp: Dict[str, Any], metric_key: str) -> Path:
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


def promote_best_model(best_run_name: str) -> Optional[Path]:
    src = CHECKPOINTS_DIR / best_run_name / "best_model"
    if not src.exists():
        print(f"[WARN] Cannot promote model: {src} does not exist. Did you save best_model in training?")
        return None

    dst = CHECKPOINTS_DIR / "BEST" / "best_model"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def save_leaderboard(rows: List[Dict[str, Any]]) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    (EXPERIMENTS_DIR / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        csv_path = EXPERIMENTS_DIR / "leaderboard.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def generate_specs(search_space: Dict[str, List[Any]], *, fp16: bool = True, limit: Optional[int] = None,) -> List[RunSpec]:
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

        if limit is not None and len(specs) >= limit:
            break

    return specs

def main() -> None:

    metric_key = "eval_macro_f1"

    search_space = {
        "learning_rate": [2e-5, 3e-5],
        "num_epochs": [3, 4, 5],
        "weight_decay": [0.02, 0.01],
        "seed": [42],
    }

    limit_runs = 40  

    specs = generate_specs(search_space, fp16=True, limit=limit_runs)
    print(f"Generated {len(specs)} runs.")

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
    print(f"Saved leaderboard to: {(EXPERIMENTS_DIR / 'leaderboard.json').resolve()} and leaderboard.csv")

    print("\n" + "-" * 100)
    print(f"BEST RUN: {best_name} ({metric_key} = {best_score:.6f})")
    print("-" * 100)

    best_json_path = save_best_summary(best_name, best_score, best_exp, metric_key=metric_key)
    print(f"Saved best run summary to: {best_json_path.resolve()}")

    promoted = promote_best_model(best_name)
    if promoted:
        print(f"Promoted best model to: {promoted.resolve()}")


if __name__ == "__main__":
    main()
