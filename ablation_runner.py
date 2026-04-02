"""
消融实验批量运行脚本

功能：
1) 读取消融计划（JSON）
2) 基于 base_config 生成每组实验配置
3) 支持按 rank/count 分片，多节点并行执行不同实验
4) 使用文件锁避免多节点重复执行同一实验
5) 汇总关键指标到 CSV
"""
import argparse
import copy
import csv
import json
import os
import random
import socket
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
    "global_index",
    "shard_rank",
    "shard_count",
    "host",
    "pid",
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

RANK_ENV_KEYS = [
    "ABLATION_SHARD_RANK",
    "SLURM_PROCID",
    "RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "MV2_COMM_WORLD_RANK",
    "PBS_VNODENUM",
]

COUNT_ENV_KEYS = [
    "ABLATION_SHARD_COUNT",
    "SLURM_NTASKS",
    "WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "PBS_NP",
]

LOCK_FILE_NAME = ".ablation.lock"


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


def _get_env_int(keys):
    for key in keys:
        raw = os.environ.get(key, None)
        if raw is None:
            continue
        raw = str(raw).strip()
        if raw == "":
            continue
        try:
            return int(raw), key
        except ValueError:
            print(f"警告: 环境变量 {key}={raw} 不是整数，忽略")
    return None, None


def _detect_shard_from_env():
    rank, rank_key = _get_env_int(RANK_ENV_KEYS)
    count, count_key = _get_env_int(COUNT_ENV_KEYS)
    if rank is None or count is None:
        return None, None, None
    return rank, count, f"{rank_key}/{count_key}"


def _resolve_shard_config(shard_rank=None, shard_count=None, auto_detect_shard=True):
    source = "default"
    if auto_detect_shard and (shard_rank is None or shard_count is None):
        env_rank, env_count, env_source = _detect_shard_from_env()
        if env_rank is not None and env_count is not None:
            if shard_rank is None:
                shard_rank = env_rank
            if shard_count is None:
                shard_count = env_count
            source = f"env:{env_source}"

    if shard_rank is None:
        shard_rank = 0
    if shard_count is None:
        shard_count = 1

    try:
        shard_rank = int(shard_rank)
        shard_count = int(shard_count)
    except (TypeError, ValueError):
        raise ValueError(f"分片参数必须为整数: shard_rank={shard_rank}, shard_count={shard_count}")

    if shard_count <= 0:
        raise ValueError(f"shard_count 必须 > 0，当前为 {shard_count}")
    if shard_rank < 0 or shard_rank >= shard_count:
        raise ValueError(
            f"shard_rank 越界: rank={shard_rank}, count={shard_count}，"
            "要求 0 <= rank < count"
        )
    if source == "default":
        source = "cli/default"
    return shard_rank, shard_count, source


def _select_shard_experiments(experiments, shard_rank, shard_count):
    indexed = list(enumerate(experiments, start=1))
    if shard_count <= 1:
        return indexed
    return [(idx, exp) for idx, exp in indexed if (idx - 1) % shard_count == shard_rank]


def _resolve_output_csv_path(run_root, output_csv, shard_rank, shard_count):
    if output_csv:
        output_csv = os.path.abspath(output_csv)
        if shard_count > 1:
            if any(token in output_csv for token in ("{rank}", "{shard_rank}", "{shard_count}")):
                return output_csv.format(
                    rank=shard_rank,
                    shard_rank=shard_rank,
                    shard_count=shard_count
                )
            base, ext = os.path.splitext(output_csv)
            ext = ext or ".csv"
            return f"{base}.rank{shard_rank:03d}{ext}"
        return output_csv

    if shard_count > 1:
        return os.path.join(run_root, f"ablation_results_rank{shard_rank:03d}.csv")
    return os.path.join(run_root, "ablation_results.csv")


def _acquire_lock(lock_path, lock_payload):
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags)
    except FileExistsError:
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(lock_payload, f, ensure_ascii=False, indent=2)
    except Exception:
        try:
            os.remove(lock_path)
        except OSError:
            pass
        raise
    return True


def _release_lock(lock_path):
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError as e:
        print(f"警告: 清理锁文件失败: {lock_path} ({e})")


def _read_lock_info(lock_path):
    if not os.path.exists(lock_path):
        return None
    try:
        return _read_json(lock_path)
    except Exception:
        return None


def _build_experiment_config(base_config,
                             global_overrides,
                             overrides,
                             checkpoint_dir,
                             exp_dir,
                             isolate_data_paths=True):
    config = copy.deepcopy(base_config)
    _deep_update(config, global_overrides)
    _deep_update(config, overrides)

    config.setdefault("training", {})
    config["training"]["checkpoint_dir"] = checkpoint_dir
    # 消融实验默认从头训练，避免跨实验污染
    config["training"]["resume_from"] = None

    if isolate_data_paths:
        config.setdefault("data", {})
        config["data"]["list_output_dir"] = os.path.join(exp_dir, "lists")
        config["data"]["subset_list_dir"] = os.path.join(exp_dir, "subset_lists")

    return config


def run_ablation(plan_path,
                 run_root="ablation_runs",
                 output_csv=None,
                 base_config_override=None,
                 stop_on_error=False,
                 skip_existing=False,
                 seed_override=None,
                 shard_rank=None,
                 shard_count=None,
                 auto_detect_shard=True,
                 isolate_data_paths=True,
                 disable_experiment_lock=False):
    plan_path = os.path.abspath(plan_path)
    plan_dir = os.path.dirname(plan_path)
    run_root = os.path.abspath(run_root)

    plan = _read_json(plan_path)
    base_config_path = base_config_override or plan.get("base_config", "config.json")
    base_config_path = _resolve_path(base_config_path, plan_dir)
    base_config = _read_json(base_config_path)

    global_overrides = plan.get("global_overrides", {})
    experiments = plan.get("experiments", [])
    if not experiments:
        raise ValueError("消融计划中 experiments 为空，请至少配置一组实验")

    shard_rank, shard_count, shard_source = _resolve_shard_config(
        shard_rank=shard_rank,
        shard_count=shard_count,
        auto_detect_shard=auto_detect_shard
    )
    sharded_experiments = _select_shard_experiments(experiments, shard_rank, shard_count)
    output_csv = _resolve_output_csv_path(run_root, output_csv, shard_rank, shard_count)

    plan_seed = seed_override if seed_override is not None else plan.get("seed", None)
    os.makedirs(run_root, exist_ok=True)

    all_rows = []
    host_name = socket.gethostname()
    process_id = os.getpid()

    print("=" * 80)
    print("开始执行消融实验")
    print("=" * 80)
    print(f"计划文件: {plan_path}")
    print(f"基础配置: {base_config_path}")
    print(f"实验总数: {len(experiments)}")
    print(f"分片信息: rank={shard_rank}, count={shard_count}, source={shard_source}")
    print(f"当前分片负责实验数: {len(sharded_experiments)}")
    if shard_count > 1 and output_csv:
        print(f"分片结果 CSV: {output_csv}")
    print(f"结果输出: {output_csv}")
    print("-" * 80)

    if not sharded_experiments:
        print("当前分片没有需要执行的实验，直接退出。")
        _write_results_csv(output_csv, all_rows)
        return all_rows

    for local_index, (global_index, exp) in enumerate(sharded_experiments, start=1):
        name = exp.get("name", f"exp_{global_index:02d}")
        enabled = exp.get("enabled", True)
        overrides = exp.get("overrides", {})
        exp_seed = exp.get("seed", plan_seed)
        progress_prefix = (
            f"[{local_index}/{len(sharded_experiments)} | "
            f"global {global_index}/{len(experiments)}]"
        )

        if not enabled:
            print(f"{progress_prefix} 跳过禁用实验: {name}")
            continue

        exp_dir = os.path.join(run_root, name)
        result_json_path = os.path.join(exp_dir, "result.json")
        config_path = os.path.join(exp_dir, "config.json")
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        lock_path = os.path.join(exp_dir, LOCK_FILE_NAME)
        lock_acquired = False

        if skip_existing and os.path.exists(result_json_path):
            print(f"{progress_prefix} 已存在结果，跳过: {name}")
            existing = _read_json(result_json_path)
            if isinstance(existing, dict):
                existing.setdefault("experiment", name)
                existing.setdefault("global_index", global_index)
                existing.setdefault("shard_rank", shard_rank)
                existing.setdefault("shard_count", shard_count)
                existing.setdefault("host", host_name)
                existing.setdefault("pid", process_id)
            all_rows.append(existing)
            _write_results_csv(output_csv, all_rows)
            continue

        os.makedirs(exp_dir, exist_ok=True)

        row = {
            "experiment": name,
            "global_index": global_index,
            "shard_rank": shard_rank,
            "shard_count": shard_count,
            "host": host_name,
            "pid": process_id,
            "status": "success",
            "started_at": None,
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

        if not disable_experiment_lock:
            lock_payload = {
                "experiment": name,
                "global_index": global_index,
                "shard_rank": shard_rank,
                "shard_count": shard_count,
                "host": host_name,
                "pid": process_id,
                "lock_time": datetime.now().isoformat(timespec="seconds"),
            }
            lock_acquired = _acquire_lock(lock_path, lock_payload)
            if not lock_acquired:
                lock_info = _read_lock_info(lock_path)
                print(f"{progress_prefix} 检测到实验锁，已跳过（可能由其他节点执行）: {name}")
                row["status"] = "skipped_locked"
                row["started_at"] = datetime.now().isoformat(timespec="seconds")
                row["finished_at"] = row["started_at"]
                row["duration_seconds"] = 0.0
                row["error"] = f"实验锁存在: {lock_info if lock_info is not None else lock_path}"
                all_rows.append(row)
                _write_results_csv(output_csv, all_rows)
                continue
        else:
            lock_acquired = False

        print(f"{progress_prefix} 开始实验: {name}")
        row["started_at"] = datetime.now().isoformat(timespec="seconds")
        start_ts = time.time()
        abort_error = None

        try:
            exp_config = _build_experiment_config(
                base_config=base_config,
                global_overrides=global_overrides,
                overrides=overrides,
                checkpoint_dir=checkpoint_dir,
                exp_dir=exp_dir,
                isolate_data_paths=isolate_data_paths,
            )
            _write_json(config_path, exp_config)

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
            try:
                row["finished_at"] = datetime.now().isoformat(timespec="seconds")
                row["duration_seconds"] = round(time.time() - start_ts, 2)
                _write_json(result_json_path, row)
                all_rows.append(row)
                _write_results_csv(output_csv, all_rows)
            finally:
                if lock_acquired:
                    _release_lock(lock_path)

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
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "结果 CSV 路径。多分片模式下会自动加 rank 后缀；"
            "也可用模板，如 results_{rank}.csv"
        )
    )
    parser.add_argument("--base_config", type=str, default=None, help="覆盖 plan.base_config")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子，覆盖 plan.seed")
    parser.add_argument("--stop_on_error", action="store_true", help="任一实验失败即停止")
    parser.add_argument("--skip_existing", action="store_true", help="已存在 result.json 的实验直接跳过")
    parser.add_argument("--shard_rank", type=int, default=None, help="当前节点分片编号（0-based）")
    parser.add_argument("--shard_count", type=int, default=None, help="总分片数/并行节点数")
    parser.add_argument(
        "--disable_auto_shard",
        action="store_true",
        help="禁用从环境变量自动识别 rank/count（SLURM/RANK/WORLD_SIZE 等）"
    )
    parser.add_argument(
        "--disable_data_path_isolation",
        action="store_true",
        help="禁用数据列表输出目录隔离（默认每个实验独立 list_output_dir/subset_list_dir）"
    )
    parser.add_argument(
        "--disable_experiment_lock",
        action="store_true",
        help="禁用实验锁（不推荐，多节点并行时可能重复执行同一实验）"
    )
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
        shard_rank=args.shard_rank,
        shard_count=args.shard_count,
        auto_detect_shard=(not args.disable_auto_shard),
        isolate_data_paths=(not args.disable_data_path_isolation),
        disable_experiment_lock=args.disable_experiment_lock,
    )
