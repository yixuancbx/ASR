"""
消融实验批量运行脚本

功能：
1) 读取消融计划（JSON）
2) 基于 base_config 生成每组实验配置
3) 逐组调用 main.py 训练入口
4) 汇总关键指标到 CSV
"""
import argparse
import copy
import csv
import json
import os
import random
import time
import traceback
from datetime import datetime

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from main import main as train_main


RESULT_FIELDS = [
    "experiment",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "config_path",
    "checkpoint_dir",
    "best_val_loss",
    "best_val_eer",
    "last_train_loss",
    "last_val_loss",
    "last_val_eer",
    "error",
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _resolve_path(path, base_dir):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def _write_results_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _build_experiment_config(base_config, global_overrides, overrides, checkpoint_dir):
    config = copy.deepcopy(base_config)
    _deep_update(config, global_overrides)
    _deep_update(config, overrides)

    config.setdefault("training", {})
    config["training"]["checkpoint_dir"] = checkpoint_dir
    # 消融实验默认从头训练，避免跨实验污染
    config["training"]["resume_from"] = None
    return config


def run_ablation(plan_path,
                 run_root="ablation_runs",
                 output_csv=None,
                 base_config_override=None,
                 stop_on_error=False,
                 skip_existing=False,
                 seed_override=None):
    plan_path = os.path.abspath(plan_path)
    plan_dir = os.path.dirname(plan_path)
    run_root = os.path.abspath(run_root)
    output_csv = os.path.abspath(output_csv) if output_csv else os.path.join(run_root, "ablation_results.csv")

    plan = _read_json(plan_path)
    base_config_path = base_config_override or plan.get("base_config", "config.json")
    base_config_path = _resolve_path(base_config_path, plan_dir)
    base_config = _read_json(base_config_path)

    global_overrides = plan.get("global_overrides", {})
    experiments = plan.get("experiments", [])
    if not experiments:
        raise ValueError("消融计划中 experiments 为空，请至少配置一组实验")

    plan_seed = seed_override if seed_override is not None else plan.get("seed", None)
    os.makedirs(run_root, exist_ok=True)

    all_rows = []

    print("=" * 80)
    print("开始执行消融实验")
    print("=" * 80)
    print(f"计划文件: {plan_path}")
    print(f"基础配置: {base_config_path}")
    print(f"实验数量: {len(experiments)}")
    print(f"结果输出: {output_csv}")
    print("-" * 80)

    for index, exp in enumerate(experiments, start=1):
        name = exp.get("name", f"exp_{index:02d}")
        enabled = exp.get("enabled", True)
        overrides = exp.get("overrides", {})
        exp_seed = exp.get("seed", plan_seed)

        if not enabled:
            print(f"[{index}/{len(experiments)}] 跳过禁用实验: {name}")
            continue

        exp_dir = os.path.join(run_root, name)
        result_json_path = os.path.join(exp_dir, "result.json")
        config_path = os.path.join(exp_dir, "config.json")
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")

        if skip_existing and os.path.exists(result_json_path):
            print(f"[{index}/{len(experiments)}] 已存在结果，跳过: {name}")
            existing = _read_json(result_json_path)
            all_rows.append(existing)
            _write_results_csv(output_csv, all_rows)
            continue

        os.makedirs(exp_dir, exist_ok=True)
        exp_config = _build_experiment_config(
            base_config=base_config,
            global_overrides=global_overrides,
            overrides=overrides,
            checkpoint_dir=checkpoint_dir,
        )
        _write_json(config_path, exp_config)

        print(f"[{index}/{len(experiments)}] 开始实验: {name}")
        started_at = datetime.now().isoformat(timespec="seconds")
        start_ts = time.time()

        row = {
            "experiment": name,
            "status": "success",
            "started_at": started_at,
            "finished_at": None,
            "duration_seconds": None,
            "config_path": config_path,
            "checkpoint_dir": checkpoint_dir,
            "best_val_loss": None,
            "best_val_eer": None,
            "last_train_loss": None,
            "last_val_loss": None,
            "last_val_eer": None,
            "error": "",
        }
        abort_error = None

        try:
            _set_seed(exp_seed)
            summary = train_main(config_path=config_path, disable_auto_resume=True)
            if isinstance(summary, dict):
                row["best_val_loss"] = summary.get("best_val_loss")
                row["best_val_eer"] = summary.get("best_val_eer")
                row["last_train_loss"] = summary.get("last_train_loss")
                row["last_val_loss"] = summary.get("last_val_loss")
                row["last_val_eer"] = summary.get("last_val_eer")
                row["checkpoint_dir"] = summary.get("checkpoint_dir", checkpoint_dir)
        except Exception as exc:  # pragma: no cover
            row["status"] = "failed"
            row["error"] = f"{exc}\n{traceback.format_exc(limit=5)}"
            print(f"实验失败: {name}\n{exc}")
            if stop_on_error:
                abort_error = exc
        finally:
            finished_at = datetime.now().isoformat(timespec="seconds")
            row["finished_at"] = finished_at
            row["duration_seconds"] = round(time.time() - start_ts, 2)
            _write_json(result_json_path, row)
            all_rows.append(row)
            _write_results_csv(output_csv, all_rows)

        if abort_error is not None:
            raise RuntimeError(f"实验 {name} 失败，已根据 stop_on_error 中止后续实验") from abort_error

        print(
            f"实验完成: {name} | status={row['status']} | "
            f"best_val_loss={row['best_val_loss']} | best_val_eer={row['best_val_eer']}"
        )
        print("-" * 80)

    print("全部消融实验执行结束。")
    print(f"汇总结果: {output_csv}")
    return all_rows


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="批量执行消融实验")
    parser.add_argument("--plan", type=str, default="ablation_plan.json", help="消融计划 JSON 路径")
    parser.add_argument("--run_root", type=str, default="ablation_runs", help="实验输出目录")
    parser.add_argument("--output_csv", type=str, default=None, help="结果 CSV 路径（默认 run_root/ablation_results.csv）")
    parser.add_argument("--base_config", type=str, default=None, help="覆盖 plan.base_config")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子，覆盖 plan.seed")
    parser.add_argument("--stop_on_error", action="store_true", help="任一实验失败即停止")
    parser.add_argument("--skip_existing", action="store_true", help="已存在 result.json 的实验直接跳过")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_ablation(
        plan_path=args.plan,
        run_root=args.run_root,
        output_csv=args.output_csv,
        base_config_override=args.base_config,
        stop_on_error=args.stop_on_error,
        skip_existing=args.skip_existing,
        seed_override=args.seed,
    )
