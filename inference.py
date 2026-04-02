"""
真实推理脚本：
1) eval  : 基于 train_list 建库、在 val_list 上评估 Top-1/Top-K
2) single: 对单个文件做闭集识别（需 enroll_list）
3) pair  : 计算两个文件的余弦相似度
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import SpeakerRecognitionModel


def _read_json(path):
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path, data_root=None):
    path = os.path.normpath(path)
    if data_root and not os.path.isabs(path):
        path = os.path.join(data_root, path)
    return os.path.normpath(path)


def _resolve_from_base(path, base_dir):
    if path and not os.path.isabs(path):
        return os.path.normpath(os.path.join(base_dir, path))
    return os.path.normpath(path)


def _load_list_entries(list_file, data_root=None):
    entries = []
    skipped_invalid = 0
    skipped_missing = 0

    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                skipped_invalid += 1
                continue
            file_path = _resolve_path(parts[0], data_root=data_root)
            speaker = parts[1]
            if not os.path.exists(file_path):
                skipped_missing += 1
                continue
            entries.append((file_path, speaker))

    return entries, skipped_invalid, skipped_missing


def _limit_entries(entries, max_total=None, max_per_speaker=None, seed=42):
    if max_per_speaker is not None and max_per_speaker > 0:
        grouped = defaultdict(list)
        for item in entries:
            grouped[item[1]].append(item)
        rng = random.Random(seed)
        limited = []
        for speaker, items in grouped.items():
            if len(items) <= max_per_speaker:
                limited.extend(items)
            else:
                limited.extend(rng.sample(items, max_per_speaker))
        entries = limited

    if max_total is not None and max_total > 0 and len(entries) > max_total:
        rng = random.Random(seed)
        entries = rng.sample(entries, max_total)

    return entries


def _infer_dataset_type_from_path(path):
    ext = os.path.splitext(path.lower())[1]
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        return "video_like"
    return "audio_like"


def _guess_subset_lists(data_cfg, config_dir):
    subset_num = data_cfg.get("subset_num_speakers")
    if subset_num is None:
        return None, None

    try:
        subset_num = int(subset_num)
    except (TypeError, ValueError):
        return None, None
    if subset_num <= 0:
        return None, None

    try:
        subset_min = int(data_cfg.get("subset_min_utterances", 80))
    except (TypeError, ValueError):
        subset_min = 80

    try:
        subset_seed = int(data_cfg.get("subset_seed", 42))
    except (TypeError, ValueError):
        subset_seed = 42

    subset_dir = _resolve_from_base(data_cfg.get("subset_list_dir", "subset_lists"), config_dir)
    subset_min = max(subset_min, 0)
    train_subset = os.path.join(
        subset_dir,
        f"train_subset_spk{subset_num}_min{subset_min}_seed{subset_seed}.txt"
    )
    val_subset = os.path.join(
        subset_dir,
        f"val_subset_spk{subset_num}_min{subset_min}_seed{subset_seed}.txt"
    )
    return train_subset, val_subset


class InferenceDataset(Dataset):
    def __init__(self,
                 entries,
                 sample_rate=16000,
                 segment_length=16000,
                 use_video=False,
                 num_frames=8,
                 frame_size=112):
        self.entries = entries
        self.sample_rate = int(sample_rate)
        self.segment_length = int(segment_length)
        self.use_video = bool(use_video)
        self.num_frames = int(num_frames)
        self.frame_size = int(frame_size)

        self.resampler = None
        self.last_sr = None

    def _load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                if self.resampler is None or self.last_sr != sr:
                    self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    self.last_sr = sr
                waveform = self.resampler(waveform)
            return waveform.squeeze(0)
        except Exception:
            return torch.zeros(self.segment_length)

    def _pad_or_truncate_audio(self, audio):
        if len(audio) > self.segment_length:
            start = (len(audio) - self.segment_length) // 2
            audio = audio[start:start + self.segment_length]
        elif len(audio) < self.segment_length:
            audio = F.pad(audio, (0, self.segment_length - len(audio)))
        return audio

    def _sample_eval_frame_indices(self, total_frames):
        if total_frames <= 0:
            return torch.zeros(self.num_frames, dtype=torch.long)
        if total_frames >= self.num_frames:
            return torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        base = torch.arange(total_frames, dtype=torch.long)
        pad = base[-1].repeat(self.num_frames - total_frames)
        return torch.cat([base, pad], dim=0)

    def _load_video_frames(self, path):
        try:
            from torchvision.io import read_video
            video_frames, _, _ = read_video(path, pts_unit="sec")
            if video_frames.numel() == 0:
                raise ValueError("empty video")

            frame_indices = self._sample_eval_frame_indices(video_frames.shape[0])
            frames = video_frames[frame_indices].permute(0, 3, 1, 2).float() / 255.0
            frames = F.interpolate(
                frames,
                size=(self.frame_size, self.frame_size),
                mode="bilinear",
                align_corners=False
            )
            frames = (frames - 0.5) / 0.5
            return frames
        except Exception:
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, speaker = self.entries[idx]
        audio = self._load_audio(path)
        audio = self._pad_or_truncate_audio(audio).unsqueeze(0)  # [1, T]
        if self.use_video:
            video = self._load_video_frames(path)  # [F, C, H, W]
            return {
                "audio": audio,
                "video": video,
                "speaker": speaker,
                "path": path,
            }
        return {
            "audio": audio,
            "speaker": speaker,
            "path": path,
        }


class SpeakerIdentifier:
    """加载 checkpoint 并提供真实推理能力。"""
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 格式错误，期望是包含 model_state_dict/config 的字典")

        self.config = checkpoint.get("config", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if isinstance(state_dict, dict) and state_dict:
            first_key = next(iter(state_dict.keys()))
            if first_key.startswith("module."):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            state_dict = {}

        has_frequency_branch = any(
            key.startswith("multi_scale_frontend.frequency_transform.")
            for key in state_dict.keys()
        )
        use_frequency_transform = self.config.get("use_frequency_transform", has_frequency_branch)

        self.model = SpeakerRecognitionModel(
            in_channels=self.config.get("in_channels", 1),
            frontend_channels=self.config.get("frontend_channels", 64),
            attention_channels=self.config.get("attention_channels", self.config.get("frontend_channels", 64) * 3),
            embedding_dim=self.config.get("embedding_dim", 256),
            num_classes=self.config.get("num_classes", 100),
            num_heads=self.config.get("num_heads", 8),
            dropout=self.config.get("dropout", 0.1),
            temporal_pool_stride=self.config.get("temporal_pool_stride", 1),
            max_attention_frames=self.config.get("max_attention_frames", 0),
            temporal_pool_type=self.config.get("temporal_pool_type", "avg"),
            use_attention_checkpoint=self.config.get("use_attention_checkpoint", False),
            use_frequency_transform=use_frequency_transform,
            freq_n_fft=self.config.get("freq_n_fft", 512),
            freq_hop_length=self.config.get("freq_hop_length", 160),
            freq_win_length=self.config.get("freq_win_length", 400),
            freq_projection_channels=self.config.get("freq_projection_channels", 64),
            freq_fusion_scale=self.config.get("freq_fusion_scale", 0.4),
            use_video=self.config.get("use_video", False),
            video_in_channels=self.config.get("video_in_channels", 3),
            video_channels=self.config.get("video_channels", 32),
            visual_encoder_dropout=self.config.get("visual_encoder_dropout", self.config.get("dropout", 0.1)),
            fusion_dropout=self.config.get("fusion_dropout", self.config.get("dropout", 0.1)),
            modality_dropout=self.config.get("modality_dropout", 0.0),
            fusion_num_heads=self.config.get("fusion_num_heads", 4),
        ).to(self.device)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"警告: 模型加载缺失参数 {missing}")
        if unexpected:
            print(f"警告: 模型加载存在多余参数 {unexpected}")

        self.model.eval()
        print(f"模型已加载: {checkpoint_path}")
        print(f"设备: {self.device}")
        print(f"use_video: {self.config.get('use_video', False)}")

    def extract_embedding(self, audio, video=None):
        """支持输入 [T] / [1,T] / [B,1,T] 并返回归一化 embedding。"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            if audio.shape[0] == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.unsqueeze(1)

        audio = audio.to(self.device)
        video = video.to(self.device) if video is not None else None

        with torch.inference_mode():
            embedding = self.model(audio, video=video)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def compute_similarity(self, embedding1, embedding2):
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        if similarity.shape[0] == 1:
            return similarity.item()
        return similarity

    def _create_loader(self, entries, sample_rate, segment_length, use_video, num_frames, frame_size, batch_size, num_workers):
        dataset = InferenceDataset(
            entries=entries,
            sample_rate=sample_rate,
            segment_length=segment_length,
            use_video=use_video,
            num_frames=num_frames,
            frame_size=frame_size,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def build_centroids(self,
                        enroll_entries,
                        sample_rate=16000,
                        segment_length=16000,
                        use_video=False,
                        num_frames=8,
                        frame_size=112,
                        batch_size=8,
                        num_workers=0):
        """从注册集构建每个说话人的均值中心。"""
        loader = self._create_loader(
            enroll_entries,
            sample_rate,
            segment_length,
            use_video,
            num_frames,
            frame_size,
            batch_size,
            num_workers,
        )

        sums = {}
        counts = defaultdict(int)

        with torch.inference_mode():
            for batch in tqdm(loader, desc="提取注册集特征"):
                audio = batch["audio"].to(self.device, non_blocking=True)
                video = batch["video"].to(self.device, non_blocking=True) if use_video else None
                embeddings = self.model(audio, video=video)
                embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
                speakers = batch["speaker"]

                for emb, speaker in zip(embeddings, speakers):
                    if speaker not in sums:
                        sums[speaker] = emb.clone()
                    else:
                        sums[speaker] += emb
                    counts[speaker] += 1

        if not sums:
            raise RuntimeError("注册集为空，无法建立说话人中心")

        speaker_list = sorted(sums.keys())
        centroids = []
        for speaker in speaker_list:
            center = sums[speaker] / max(counts[speaker], 1)
            center = F.normalize(center.unsqueeze(0), p=2, dim=1).squeeze(0)
            centroids.append(center)
        centroids = torch.stack(centroids, dim=0)

        return speaker_list, centroids, dict(counts)

    def evaluate_closed_set(self,
                            query_entries,
                            speaker_list,
                            centroids,
                            sample_rate=16000,
                            segment_length=16000,
                            use_video=False,
                            num_frames=8,
                            frame_size=112,
                            batch_size=8,
                            num_workers=0,
                            top_k=5,
                            show_examples=5):
        """闭集识别评估：query 与注册中心做余弦相似度分类。"""
        speaker_to_index = {speaker: i for i, speaker in enumerate(speaker_list)}
        centroids_device = centroids.to(self.device)
        top_k = max(1, min(int(top_k), len(speaker_list)))

        loader = self._create_loader(
            query_entries,
            sample_rate,
            segment_length,
            use_video,
            num_frames,
            frame_size,
            batch_size,
            num_workers,
        )

        total = 0
        correct_top1 = 0
        correct_topk = 0
        skipped_unknown = 0
        true_scores = []
        imp_scores = []
        examples = []

        with torch.inference_mode():
            for batch in tqdm(loader, desc="评估查询集"):
                audio = batch["audio"].to(self.device, non_blocking=True)
                video = batch["video"].to(self.device, non_blocking=True) if use_video else None
                embeddings = self.model(audio, video=video)
                embeddings = F.normalize(embeddings, p=2, dim=1)

                scores = torch.matmul(embeddings, centroids_device.t())
                top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

                speakers = batch["speaker"]
                paths = batch["path"]
                for idx, true_speaker in enumerate(speakers):
                    if true_speaker not in speaker_to_index:
                        skipped_unknown += 1
                        continue

                    total += 1
                    true_idx = speaker_to_index[true_speaker]
                    pred_idx = top_indices[idx, 0].item()
                    if pred_idx == true_idx:
                        correct_top1 += 1
                    if (top_indices[idx] == true_idx).any().item():
                        correct_topk += 1

                    true_score = scores[idx, true_idx].item()
                    impostor = scores[idx].clone()
                    impostor[true_idx] = -1e9
                    impostor_score = impostor.max().item()
                    true_scores.append(true_score)
                    imp_scores.append(impostor_score)

                    if len(examples) < show_examples:
                        pred_items = []
                        for k in range(top_k):
                            pred_speaker = speaker_list[top_indices[idx, k].item()]
                            pred_score = top_scores[idx, k].item()
                            pred_items.append((pred_speaker, pred_score))
                        examples.append({
                            "path": paths[idx],
                            "true_speaker": true_speaker,
                            "predictions": pred_items,
                        })

        top1 = (correct_top1 / total * 100.0) if total > 0 else 0.0
        topk = (correct_topk / total * 100.0) if total > 0 else 0.0
        avg_true = float(torch.tensor(true_scores).mean().item()) if true_scores else 0.0
        avg_imp = float(torch.tensor(imp_scores).mean().item()) if imp_scores else 0.0
        avg_margin = avg_true - avg_imp

        return {
            "total_eval_samples": total,
            "skipped_unknown_speaker": skipped_unknown,
            "top1_acc": top1,
            "topk_acc": topk,
            "topk_value": top_k,
            "avg_true_score": avg_true,
            "avg_impostor_score": avg_imp,
            "avg_margin": avg_margin,
            "examples": examples,
        }

    def identify_single_file(self, file_path, speaker_list, centroids, sample_rate, segment_length, use_video, num_frames, frame_size, top_k=5):
        top_k = max(1, min(int(top_k), len(speaker_list)))
        dataset = InferenceDataset(
            entries=[(file_path, "__query__")],
            sample_rate=sample_rate,
            segment_length=segment_length,
            use_video=use_video,
            num_frames=num_frames,
            frame_size=frame_size,
        )
        sample = dataset[0]
        audio = sample["audio"].unsqueeze(0).to(self.device)
        video = sample["video"].unsqueeze(0).to(self.device) if use_video else None
        with torch.inference_mode():
            embedding = self.model(audio, video=video)
            embedding = F.normalize(embedding, p=2, dim=1)
            scores = torch.matmul(embedding, centroids.to(self.device).t())[0]
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=0)

        predictions = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            predictions.append((speaker_list[idx], score))
        return predictions

    def compare_two_files(self,
                          file_a,
                          file_b,
                          sample_rate,
                          segment_length,
                          use_video=False,
                          num_frames=8,
                          frame_size=112):
        dataset = InferenceDataset(
            entries=[(file_a, "__a__"), (file_b, "__b__")],
            sample_rate=sample_rate,
            segment_length=segment_length,
            use_video=use_video,
            num_frames=num_frames,
            frame_size=frame_size,
        )
        sample_a = dataset[0]
        sample_b = dataset[1]
        video_a = sample_a["video"].unsqueeze(0) if use_video else None
        video_b = sample_b["video"].unsqueeze(0) if use_video else None
        emb_a = self.extract_embedding(sample_a["audio"], video=video_a).squeeze(0).cpu()
        emb_b = self.extract_embedding(sample_b["audio"], video=video_b).squeeze(0).cpu()
        similarity = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=1).item()
        return similarity


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="真实说话人推理脚本")
    parser.add_argument("--mode", choices=["eval", "single", "pair"], default="eval")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_type", type=str, default=None, choices=["voxceleb", "lrs", "vox2video"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--segment_length", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--frame_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--enroll_list", type=str, default=None, help="注册列表（默认取 config.data.train_list）")
    parser.add_argument("--query_list", type=str, default=None, help="查询列表（默认取 config.data.val_list）")
    parser.add_argument("--max_enroll_per_speaker", type=int, default=5)
    parser.add_argument("--max_enroll_total", type=int, default=None)
    parser.add_argument("--max_query_total", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--show_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input", type=str, default=None, help="single/pair 模式输入文件A")
    parser.add_argument("--input_b", type=str, default=None, help="pair 模式输入文件B")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    config_json = _read_json(args.config)
    data_cfg = config_json.get("data", {}) if isinstance(config_json, dict) else {}
    config_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()

    identifier = SpeakerIdentifier(args.checkpoint, device=args.device)

    # 优先级：命令行 > config.json > checkpoint-config > 默认值
    sample_rate = args.sample_rate or data_cfg.get("sample_rate") or identifier.config.get("sample_rate", 16000)
    segment_length = args.segment_length or data_cfg.get("seq_length") or identifier.config.get("segment_length", 16000)
    num_frames = args.num_frames or data_cfg.get("video_num_frames", 8)
    frame_size = args.frame_size or data_cfg.get("video_frame_size", 112)

    dataset_type = args.dataset_type or data_cfg.get("dataset_type")
    if dataset_type is None and args.input:
        inferred = _infer_dataset_type_from_path(args.input)
        dataset_type = "vox2video" if inferred == "video_like" else "voxceleb"
    dataset_type = dataset_type or "voxceleb"
    use_video = bool(identifier.config.get("use_video", False) or dataset_type == "vox2video")

    data_root = args.data_root if args.data_root is not None else data_cfg.get("data_root")
    if data_root:
        data_root = _resolve_from_base(data_root, config_dir)

    print("=" * 80)
    print("推理配置")
    print("=" * 80)
    print(f"mode: {args.mode}")
    print(f"dataset_type: {dataset_type}")
    print(f"use_video: {use_video}")
    print(f"sample_rate: {sample_rate}")
    print(f"segment_length: {segment_length}")
    print(f"data_root: {data_root}")
    print("-" * 80)

    if args.mode == "pair":
        if not args.input or not args.input_b:
            raise ValueError("pair 模式需要 --input 和 --input_b")
        file_a = _resolve_path(args.input, data_root=data_root)
        file_b = _resolve_path(args.input_b, data_root=data_root)
        sim = identifier.compare_two_files(
            file_a=file_a,
            file_b=file_b,
            sample_rate=sample_rate,
            segment_length=segment_length,
            use_video=use_video,
            num_frames=num_frames,
            frame_size=frame_size,
        )
        print(f"文件A: {file_a}")
        print(f"文件B: {file_b}")
        print(f"余弦相似度: {sim:.4f}")
        return

    subset_train_list, subset_val_list = _guess_subset_lists(data_cfg, config_dir)

    if args.enroll_list:
        enroll_list = args.enroll_list
    elif subset_train_list and os.path.exists(subset_train_list):
        enroll_list = subset_train_list
        print(f"检测到子集训练列表，默认使用: {enroll_list}")
    else:
        enroll_list = data_cfg.get("train_list")

    if not enroll_list:
        raise ValueError("未提供 enroll_list，且 config.json 中没有 data.train_list")
    enroll_list = _resolve_from_base(enroll_list, config_dir)

    enroll_entries, enroll_invalid, enroll_missing = _load_list_entries(enroll_list, data_root=data_root)
    enroll_entries = _limit_entries(
        enroll_entries,
        max_total=args.max_enroll_total,
        max_per_speaker=args.max_enroll_per_speaker,
        seed=args.seed,
    )
    print(f"注册列表: {enroll_list}")
    print(f"注册样本数: {len(enroll_entries)} (invalid={enroll_invalid}, missing={enroll_missing})")

    speaker_list, centroids, counts = identifier.build_centroids(
        enroll_entries=enroll_entries,
        sample_rate=sample_rate,
        segment_length=segment_length,
        use_video=use_video,
        num_frames=num_frames,
        frame_size=frame_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"已建立说话人中心: {len(speaker_list)} 人")
    print(f"每人注册条数统计: min={min(counts.values())}, avg={sum(counts.values()) / len(counts):.2f}, max={max(counts.values())}")

    if args.mode == "single":
        if not args.input:
            raise ValueError("single 模式需要 --input")
        query_file = _resolve_path(args.input, data_root=data_root)
        predictions = identifier.identify_single_file(
            file_path=query_file,
            speaker_list=speaker_list,
            centroids=centroids,
            sample_rate=sample_rate,
            segment_length=segment_length,
            use_video=use_video,
            num_frames=num_frames,
            frame_size=frame_size,
            top_k=args.top_k,
        )
        print(f"\n查询文件: {query_file}")
        print(f"Top-{len(predictions)} 识别结果:")
        for i, (speaker, score) in enumerate(predictions, start=1):
            print(f"  {i}. {speaker:>12s}  score={score:.4f}")
        return

    if args.query_list:
        query_list = args.query_list
    elif subset_val_list and os.path.exists(subset_val_list):
        query_list = subset_val_list
        print(f"检测到子集验证列表，默认使用: {query_list}")
    else:
        query_list = data_cfg.get("val_list")

    if not query_list:
        raise ValueError("eval 模式需要 query_list，且 config.json 中没有 data.val_list")
    query_list = _resolve_from_base(query_list, config_dir)

    query_entries, query_invalid, query_missing = _load_list_entries(query_list, data_root=data_root)
    query_entries = _limit_entries(
        query_entries,
        max_total=args.max_query_total,
        max_per_speaker=None,
        seed=args.seed,
    )

    print(f"\n查询列表: {query_list}")
    print(f"查询样本数: {len(query_entries)} (invalid={query_invalid}, missing={query_missing})")

    results = identifier.evaluate_closed_set(
        query_entries=query_entries,
        speaker_list=speaker_list,
        centroids=centroids,
        sample_rate=sample_rate,
        segment_length=segment_length,
        use_video=use_video,
        num_frames=num_frames,
        frame_size=frame_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
        show_examples=args.show_examples,
    )

    print("\n" + "=" * 80)
    print("评估结果（真实推理）")
    print("=" * 80)
    print(f"有效评估样本: {results['total_eval_samples']}")
    print(f"跳过（query说话人不在注册库）: {results['skipped_unknown_speaker']}")
    print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"Top-{results['topk_value']} Accuracy: {results['topk_acc']:.2f}%")
    print(f"平均真类分数: {results['avg_true_score']:.4f}")
    print(f"平均伪类最高分: {results['avg_impostor_score']:.4f}")
    print(f"平均分数间隔(真-伪): {results['avg_margin']:.4f}")

    if results["examples"]:
        print("\n示例预测:")
        for i, item in enumerate(results["examples"], start=1):
            top_line = " | ".join([f"{spk}:{score:.3f}" for spk, score in item["predictions"]])
            print(f"[{i}] true={item['true_speaker']} | file={item['path']}")
            print(f"    pred -> {top_line}")


if __name__ == "__main__":
    main()





