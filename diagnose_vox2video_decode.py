"""
Quick decode health check for Vox2Video style lists.

This script samples items from a list file and measures:
- read_video success/failure ratio
- torchaudio.load success/failure ratio
- estimated fallback ratio according to dataset logic
- near-zero audio/video ratio
- top exception categories
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import torch

try:
    import torchaudio
except Exception as exc:  # pragma: no cover
    torchaudio = None
    TORCHAUDIO_IMPORT_ERROR = exc
else:
    TORCHAUDIO_IMPORT_ERROR = None

try:
    from torchvision.io import read_video
except Exception as exc:  # pragma: no cover
    read_video = None
    TORCHVISION_IMPORT_ERROR = exc
else:
    TORCHVISION_IMPORT_ERROR = None


def _short_error(error):
    text = str(error).strip().replace("\n", " ")
    if len(text) > 180:
        text = text[:177] + "..."
    return f"{type(error).__name__}: {text}"


def _is_near_zero(tensor, eps):
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return True
    return bool(tensor.detach().abs().sum().item() <= eps)


def _resolve_path(raw_path, data_root):
    path = os.path.normpath(raw_path)
    if data_root and not os.path.isabs(path):
        path = os.path.normpath(os.path.join(data_root, path))
    return path


def _load_list_entries(list_file, data_root):
    entries = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 1:
                continue
            rel_path = parts[0]
            speaker = parts[1] if len(parts) >= 2 else "unknown"
            entries.append(
                {
                    "line_idx": line_idx,
                    "path": _resolve_path(rel_path, data_root),
                    "speaker": speaker,
                    "raw": stripped,
                }
            )
    return entries


def _sample_entries(entries, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(entries) <= max_samples:
        return list(entries)
    rng = random.Random(seed)
    return rng.sample(entries, max_samples)


def _decode_with_read_video(path, zero_eps):
    if read_video is None:
        return {
            "ok": False,
            "video_non_empty": False,
            "audio_non_empty": False,
            "video_zero": True,
            "audio_zero": True,
            "error": _short_error(TORCHVISION_IMPORT_ERROR),
        }

    try:
        video_frames, audio_frames, _info = read_video(path, pts_unit="sec")
        video_non_empty = bool(isinstance(video_frames, torch.Tensor) and video_frames.numel() > 0)
        if not video_non_empty:
            raise ValueError("empty video")
        audio_non_empty = bool(isinstance(audio_frames, torch.Tensor) and audio_frames.numel() > 0)
        return {
            "ok": True,
            "video_non_empty": video_non_empty,
            "audio_non_empty": audio_non_empty,
            "video_zero": _is_near_zero(video_frames, eps=zero_eps),
            "audio_zero": _is_near_zero(audio_frames, eps=zero_eps),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "video_non_empty": False,
            "audio_non_empty": False,
            "video_zero": True,
            "audio_zero": True,
            "error": _short_error(exc),
        }


def _decode_with_torchaudio(path, zero_eps):
    if torchaudio is None:
        return {
            "ok": False,
            "audio_non_empty": False,
            "audio_zero": True,
            "error": _short_error(TORCHAUDIO_IMPORT_ERROR),
        }
    try:
        waveform, _sr = torchaudio.load(path)
        audio_non_empty = bool(isinstance(waveform, torch.Tensor) and waveform.numel() > 0)
        return {
            "ok": True,
            "audio_non_empty": audio_non_empty,
            "audio_zero": _is_near_zero(waveform, eps=zero_eps),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "audio_non_empty": False,
            "audio_zero": True,
            "error": _short_error(exc),
        }


def run_diagnosis(list_file, data_root, max_samples, seed, zero_eps, print_failures):
    entries = _load_list_entries(list_file=list_file, data_root=data_root)
    sampled = _sample_entries(entries, max_samples=max_samples, seed=seed)

    stats = {
        "list_file": os.path.normpath(list_file),
        "data_root": os.path.normpath(data_root) if data_root else None,
        "entries_total": len(entries),
        "entries_sampled": len(sampled),
        "file_missing": 0,
        "read_video_ok": 0,
        "read_video_fail": 0,
        "torchaudio_ok": 0,
        "torchaudio_fail": 0,
        "video_zero": 0,
        "audio_zero_from_read_video": 0,
        "audio_zero_from_torchaudio": 0,
        "est_video_fallback": 0,
        "est_audio_fallback": 0,
        "est_audio_zero_after_fallback": 0,
    }

    read_video_errors = Counter()
    torchaudio_errors = Counter()
    failed_cases = []

    for item in sampled:
        path = item["path"]
        if not os.path.exists(path):
            stats["file_missing"] += 1
            failed_cases.append(
                {
                    "path": path,
                    "reason": "missing_file",
                    "line_idx": item["line_idx"],
                }
            )
            continue

        rv = _decode_with_read_video(path, zero_eps=zero_eps)
        ta = _decode_with_torchaudio(path, zero_eps=zero_eps)

        if rv["ok"]:
            stats["read_video_ok"] += 1
        else:
            stats["read_video_fail"] += 1
            read_video_errors[rv["error"]] += 1

        if ta["ok"]:
            stats["torchaudio_ok"] += 1
        else:
            stats["torchaudio_fail"] += 1
            torchaudio_errors[ta["error"]] += 1

        if rv["video_zero"]:
            stats["video_zero"] += 1
        if rv["audio_zero"]:
            stats["audio_zero_from_read_video"] += 1
        if ta["audio_zero"]:
            stats["audio_zero_from_torchaudio"] += 1

        # Match dataset fallback policy:
        # video fallback: read_video failed / empty
        # audio fallback: read_video failed OR read_video has no audio track
        est_video_fallback = not rv["ok"]
        est_audio_fallback = (not rv["ok"]) or (rv["ok"] and not rv["audio_non_empty"])
        est_audio_zero_after_fallback = est_audio_fallback and (
            (not ta["ok"]) or ta["audio_zero"]
        )

        if est_video_fallback:
            stats["est_video_fallback"] += 1
        if est_audio_fallback:
            stats["est_audio_fallback"] += 1
        if est_audio_zero_after_fallback:
            stats["est_audio_zero_after_fallback"] += 1

        if (not rv["ok"]) or (not ta["ok"]):
            failed_cases.append(
                {
                    "path": path,
                    "line_idx": item["line_idx"],
                    "read_video_ok": rv["ok"],
                    "read_video_error": rv["error"],
                    "torchaudio_ok": ta["ok"],
                    "torchaudio_error": ta["error"],
                }
            )

    valid = max(1, stats["entries_sampled"] - stats["file_missing"])

    summary = {
        **stats,
        "ratios": {
            "file_missing": stats["file_missing"] / max(1, stats["entries_sampled"]),
            "read_video_fail": stats["read_video_fail"] / valid,
            "torchaudio_fail": stats["torchaudio_fail"] / valid,
            "video_zero": stats["video_zero"] / valid,
            "audio_zero_from_read_video": stats["audio_zero_from_read_video"] / valid,
            "audio_zero_from_torchaudio": stats["audio_zero_from_torchaudio"] / valid,
            "est_video_fallback": stats["est_video_fallback"] / valid,
            "est_audio_fallback": stats["est_audio_fallback"] / valid,
            "est_audio_zero_after_fallback": stats["est_audio_zero_after_fallback"] / valid,
        },
        "top_read_video_errors": read_video_errors.most_common(10),
        "top_torchaudio_errors": torchaudio_errors.most_common(10),
        "sample_failures": failed_cases[: max(0, int(print_failures))],
        "zero_eps_used": float(zero_eps),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose Vox2Video decode health.")
    parser.add_argument("--list", type=str, required=True, help="Path to list file.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional data root to resolve relative paths in list.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Random sample count from list (<=0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--zero_eps",
        type=float,
        default=1e-8,
        help="Near-zero energy threshold for tensors.",
    )
    parser.add_argument(
        "--print_failures",
        type=int,
        default=10,
        help="How many failed sample cases to print in JSON output.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output JSON file path.",
    )
    args = parser.parse_args()

    start = time.time()
    summary = run_diagnosis(
        list_file=args.list,
        data_root=args.data_root,
        max_samples=args.max_samples,
        seed=args.seed,
        zero_eps=args.zero_eps,
        print_failures=args.print_failures,
    )
    summary["elapsed_sec"] = round(time.time() - start, 3)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        out_path = os.path.normpath(args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()

