"""
生成 Vox2Video 数据集列表文件的工具脚本。
列表格式（每行）:
relative/path/to/video.mp4 speaker_id
"""
import argparse
import glob
import os
import random
from collections import defaultdict


DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _normalize_extensions(extensions):
    """将扩展名列表标准化为形如 ['.mp4', '.mov']。"""
    if not extensions:
        return list(DEFAULT_VIDEO_EXTENSIONS)

    normalized = []
    for ext in extensions:
        ext = str(ext).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)

    if not normalized:
        return list(DEFAULT_VIDEO_EXTENSIONS)
    return normalized


def _to_posix_path(path):
    """统一输出为正斜杠路径，便于跨平台使用。"""
    return path.replace("\\", "/")


def _infer_speaker_id_from_rel_path(rel_path, speaker_prefix="id"):
    """
    从相对路径中推断 speaker_id：
    1) 优先取以 speaker_prefix 开头的目录名（如 id00012）
    2) 否则取第一级目录
    3) 兜底取父目录名
    """
    normalized = os.path.normpath(rel_path)
    parts = normalized.split(os.sep)

    prefix = (speaker_prefix or "").lower()
    if prefix:
        for part in parts:
            if part.lower().startswith(prefix):
                return part

    if len(parts) >= 2 and parts[0]:
        return parts[0]

    parent = os.path.basename(os.path.dirname(normalized))
    if parent:
        return parent

    return "unknown_speaker"


def collect_vox2video_entries(data_root, extensions=None, speaker_prefix="id"):
    """扫描目录并返回 [(相对路径, speaker_id), ...]。"""
    data_root = os.path.abspath(data_root)
    extensions = _normalize_extensions(extensions)

    all_video_files = set()
    for ext in extensions:
        pattern = os.path.join(data_root, "**", f"*{ext}")
        for filepath in glob.glob(pattern, recursive=True):
            if os.path.isfile(filepath):
                all_video_files.add(os.path.normpath(filepath))

    entries = []
    for filepath in sorted(all_video_files):
        rel_path = _to_posix_path(os.path.relpath(filepath, data_root))
        speaker_id = _infer_speaker_id_from_rel_path(rel_path, speaker_prefix=speaker_prefix)
        entries.append((rel_path, speaker_id))

    return entries


def _format_count_stats(values):
    if not values:
        return "min=0, avg=0.00, max=0"
    return f"min={min(values)}, avg={sum(values) / len(values):.2f}, max={max(values)}"


def filter_entries_by_speaker(entries, min_samples_per_speaker=1, max_speakers=None, seed=42):
    """按说话人样本数过滤，并可随机采样固定数量说话人。"""
    grouped = defaultdict(list)
    for rel_path, speaker_id in entries:
        grouped[speaker_id].append((rel_path, speaker_id))

    min_samples = max(1, int(min_samples_per_speaker))
    candidate_speakers = [spk for spk, items in grouped.items() if len(items) >= min_samples]

    if not candidate_speakers:
        raise ValueError(
            f"没有说话人满足最小样本阈值: min_samples_per_speaker={min_samples}"
        )

    selected_speakers = sorted(candidate_speakers)
    if max_speakers is not None:
        max_speakers = int(max_speakers)
        if max_speakers > 0 and len(selected_speakers) > max_speakers:
            rng = random.Random(seed)
            selected_speakers = sorted(rng.sample(selected_speakers, max_speakers))

    selected = []
    for speaker_id in selected_speakers:
        selected.extend(grouped[speaker_id])

    return selected, selected_speakers


def split_entries_by_speaker(entries, train_ratio=0.9, seed=42):
    """
    按说话人内部分割 train/val，确保标签空间一致。
    对仅 1 条样本的说话人，保留在训练集。
    """
    grouped = defaultdict(list)
    for rel_path, speaker_id in entries:
        grouped[speaker_id].append((rel_path, speaker_id))

    rng = random.Random(seed)
    train_entries = []
    val_entries = []

    for speaker_id in sorted(grouped.keys()):
        items = grouped[speaker_id]
        rng.shuffle(items)

        if len(items) == 1:
            train_entries.extend(items)
            continue

        train_size = int(len(items) * float(train_ratio))
        train_size = max(1, min(train_size, len(items) - 1))
        train_entries.extend(items[:train_size])
        val_entries.extend(items[train_size:])

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def _write_entries(entries, output_file):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rel_path, speaker_id in entries:
            f.write(f"{rel_path} {speaker_id}\n")


def generate_list_from_directory(data_root,
                                 output_file,
                                 min_samples_per_speaker=1,
                                 max_speakers=None,
                                 seed=42,
                                 speaker_prefix="id",
                                 extensions=None):
    """从目录扫描并生成完整列表（可按说话人阈值过滤）。"""
    entries = collect_vox2video_entries(
        data_root=data_root,
        extensions=extensions,
        speaker_prefix=speaker_prefix,
    )
    entries, selected_speakers = filter_entries_by_speaker(
        entries,
        min_samples_per_speaker=min_samples_per_speaker,
        max_speakers=max_speakers,
        seed=seed,
    )
    _write_entries(entries, output_file)

    speaker_counts = defaultdict(int)
    for _, speaker_id in entries:
        speaker_counts[speaker_id] += 1

    print(f"输出完整列表: {output_file}")
    print(f"样本数: {len(entries)}")
    print(f"说话人数: {len(selected_speakers)}")
    print(f"每说话人样本统计: {_format_count_stats(list(speaker_counts.values()))}")


def generate_and_split_from_directory(data_root,
                                      train_list_file,
                                      val_list_file,
                                      train_ratio=0.9,
                                      seed=42,
                                      min_samples_per_speaker=2,
                                      max_speakers=None,
                                      speaker_prefix="id",
                                      extensions=None):
    """从目录扫描并按说话人分割生成 train/val 列表。"""
    entries = collect_vox2video_entries(
        data_root=data_root,
        extensions=extensions,
        speaker_prefix=speaker_prefix,
    )
    entries, selected_speakers = filter_entries_by_speaker(
        entries,
        min_samples_per_speaker=min_samples_per_speaker,
        max_speakers=max_speakers,
        seed=seed,
    )

    train_entries, val_entries = split_entries_by_speaker(
        entries,
        train_ratio=train_ratio,
        seed=seed,
    )

    _write_entries(train_entries, train_list_file)
    _write_entries(val_entries, val_list_file)

    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for _, speaker_id in train_entries:
        train_counts[speaker_id] += 1
    for _, speaker_id in val_entries:
        val_counts[speaker_id] += 1

    total_counts = []
    for speaker_id in selected_speakers:
        total_counts.append(train_counts.get(speaker_id, 0) + val_counts.get(speaker_id, 0))

    print("=" * 80)
    print("Vox2Video 列表生成完成")
    print("=" * 80)
    print(f"数据根目录: {os.path.abspath(data_root)}")
    print(f"训练列表: {train_list_file} ({len(train_entries)} 条)")
    print(f"验证列表: {val_list_file} ({len(val_entries)} 条)")
    print(f"说话人数: {len(selected_speakers)}")
    print(f"每说话人样本统计(train): {_format_count_stats(list(train_counts.values()))}")
    print(f"每说话人样本统计(val): {_format_count_stats(list(val_counts.values()))}")
    print(f"每说话人样本统计(total): {_format_count_stats(total_counts)}")

    return train_list_file, val_list_file


def _parse_extensions_arg(value):
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    parts = [item for item in parts if item]
    return parts if parts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Vox2Video 训练/验证列表")
    parser.add_argument("--data_root", type=str, required=True, help="Vox2Video 数据根目录")
    parser.add_argument("--train_list", type=str, default="vox2video_train_list.txt", help="训练列表输出路径")
    parser.add_argument("--val_list", type=str, default="vox2video_val_list.txt", help="验证列表输出路径")
    parser.add_argument("--all_list", type=str, default=None, help="可选：输出完整列表路径")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_samples_per_speaker", type=int, default=2, help="每个说话人的最小样本数")
    parser.add_argument("--max_speakers", type=int, default=None, help="可选：最多保留的说话人数")
    parser.add_argument("--speaker_prefix", type=str, default="id", help="说话人目录前缀（如 id）")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".mp4,.avi,.mov,.mkv,.webm",
        help="逗号分隔的视频扩展名，例如 .mp4,.mov",
    )

    args = parser.parse_args()
    extensions = _parse_extensions_arg(args.extensions)

    if args.all_list:
        generate_list_from_directory(
            data_root=args.data_root,
            output_file=args.all_list,
            min_samples_per_speaker=args.min_samples_per_speaker,
            max_speakers=args.max_speakers,
            seed=args.seed,
            speaker_prefix=args.speaker_prefix,
            extensions=extensions,
        )

    generate_and_split_from_directory(
        data_root=args.data_root,
        train_list_file=args.train_list,
        val_list_file=args.val_list,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_samples_per_speaker=args.min_samples_per_speaker,
        max_speakers=args.max_speakers,
        speaker_prefix=args.speaker_prefix,
        extensions=extensions,
    )
