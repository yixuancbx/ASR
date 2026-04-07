"""
Vox2Video 音视频数据集加载器
支持直接从 mp4 中联合读取并对齐音频与视频片段。
"""
import os
import random
import ctypes

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    from torchvision.io import read_video, read_video_timestamps
except ImportError:
    from torchvision.io import read_video
    read_video_timestamps = None
try:
    from retinaface import RetinaFace
except ImportError:
    RetinaFace = None


class Vox2VideoDatasetFromList(Dataset):
    def __init__(self,
                 list_file,
                 data_root=None,
                 sample_rate=16000,
                 segment_length=16000,
                 num_frames=8,
                 frame_size=112,
                 frame_stride=2,
                 train=True,
                 augmentation=False,
                 speaker_to_id=None,
                 allow_new_speakers=True,
                 align_av_segment=True,
                 decode_window_seconds=None,
                 worker_trim_interval=256,
                 video_return_float16=True,
                 use_musan_noise=False,
                 musan_root=None,
                 musan_apply_prob=0.7,
                 musan_snr_min_db=5.0,
                 musan_snr_max_db=20.0,
                 musan_noise_weight=0.5,
                 musan_speech_weight=0.25,
                 musan_music_weight=0.25,
                 disable_audio_time_aug_for_avsync=True,
                 use_face_roi=True,
                 face_roi_expand_ratio=1.35,
                 face_roi_smooth_alpha=0.75,
                 face_roi_score_thr=0.8,
                 face_roi_min_size=24):
        self.data_root = data_root
        self.sample_rate = int(sample_rate)
        self.segment_length = int(segment_length)
        self.num_frames = int(num_frames)
        self.frame_size = int(frame_size)
        self.frame_stride = max(1, int(frame_stride))
        self.train = train
        self.augmentation = augmentation
        self.allow_new_speakers = allow_new_speakers
        self.align_av_segment = bool(align_av_segment)
        if decode_window_seconds is None:
            self.decode_window_seconds = None
        else:
            self.decode_window_seconds = max(0.0, float(decode_window_seconds))
        try:
            self.worker_trim_interval = max(0, int(worker_trim_interval))
        except (TypeError, ValueError):
            self.worker_trim_interval = 0
        self.video_return_float16 = bool(video_return_float16)
        self.disable_audio_time_aug_for_avsync = bool(disable_audio_time_aug_for_avsync)
        self.use_face_roi = bool(use_face_roi)
        self.face_roi_expand_ratio = max(1.0, float(face_roi_expand_ratio))
        self.face_roi_smooth_alpha = min(0.99, max(0.0, float(face_roi_smooth_alpha)))
        self.face_roi_score_thr = float(face_roi_score_thr)
        self.face_roi_min_size = max(4, int(face_roi_min_size))
        self._face_roi_warning_emitted = False
        self._samples_since_trim = 0
        # 注意：不要在主进程里持有 ctypes 函数指针，spawn 模式下会触发 pickle 失败。
        self._malloc_trim = None
        self._malloc_trim_initialized = False

        self.video_files = []
        self.speaker_ids = []
        self.speaker_to_id = dict(speaker_to_id) if speaker_to_id is not None else {}

        self.resampler = None
        self.last_sr = None
        self.use_musan_noise = bool(use_musan_noise)
        self.musan_root = musan_root
        self.musan_apply_prob = min(1.0, max(0.0, float(musan_apply_prob)))
        snr_lo = float(musan_snr_min_db)
        snr_hi = float(musan_snr_max_db)
        self.musan_snr_min_db = min(snr_lo, snr_hi)
        self.musan_snr_max_db = max(snr_lo, snr_hi)
        self.musan_file_index = {'noise': [], 'speech': [], 'music': []}
        self.musan_weights = {
            'noise': max(0.0, float(musan_noise_weight)),
            'speech': max(0.0, float(musan_speech_weight)),
            'music': max(0.0, float(musan_music_weight)),
        }
        self._init_musan_noise_index()

        self._load_list_file(list_file)

        print(f"从 Vox2Video 列表文件加载了 {len(self.video_files)} 个文件")
        print(f"共 {len(self.speaker_to_id)} 个说话人")

    def __getstate__(self):
        """
        兼容 multiprocessing=spawn：
        避免将 ctypes 函数指针序列化到 worker。
        """
        state = self.__dict__.copy()
        state['_malloc_trim'] = None
        state['_malloc_trim_initialized'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 在 worker 侧按需初始化，保持对象可 pickle。
        self._malloc_trim = None
        self._malloc_trim_initialized = False

    def _load_list_file(self, list_file):
        skipped_missing = 0
        skipped_unknown_speaker = 0

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                video_file = os.path.normpath(parts[0])
                speaker_name = parts[1]

                if self.data_root and not os.path.isabs(video_file):
                    video_file = os.path.join(self.data_root, video_file)

                if speaker_name not in self.speaker_to_id:
                    if not self.allow_new_speakers:
                        skipped_unknown_speaker += 1
                        continue
                    self.speaker_to_id[speaker_name] = len(self.speaker_to_id)

                if os.path.exists(video_file):
                    self.video_files.append(video_file)
                    self.speaker_ids.append(self.speaker_to_id[speaker_name])
                else:
                    skipped_missing += 1

        if skipped_unknown_speaker > 0:
            print(f"警告: 跳过 {skipped_unknown_speaker} 条未在训练集中出现的说话人样本")
        if skipped_missing > 0:
            print(f"警告: 跳过 {skipped_missing} 条不存在的视频文件")

    def _load_audio_fallback(self, filepath):
        """当视频解码失败或无音轨时，回退到 torchaudio 读取。"""
        try:
            waveform, sr = torchaudio.load(filepath)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio = waveform.squeeze(0)
            return self._resample_audio(audio, sr)
        except Exception:
            return torch.zeros(self.segment_length)

    def _resample_audio(self, audio, source_sr):
        source_sr = int(source_sr)
        if audio.numel() == 0:
            return audio
        if source_sr != self.sample_rate:
            if self.resampler is None or self.last_sr != source_sr:
                self.resampler = torchaudio.transforms.Resample(source_sr, self.sample_rate)
                self.last_sr = source_sr
            audio = self.resampler(audio.unsqueeze(0)).squeeze(0)
        return audio

    def _init_musan_noise_index(self):
        if not self.use_musan_noise:
            return
        if not self.musan_root:
            print("警告: use_musan_noise=True 但未配置 musan_root，已禁用 MUSAN 增强")
            self.use_musan_noise = False
            return
        if not os.path.isdir(self.musan_root):
            print(f"警告: MUSAN 目录不存在: {self.musan_root}，已禁用 MUSAN 增强")
            self.use_musan_noise = False
            return

        audio_exts = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
        for root, _, files in os.walk(self.musan_root):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in audio_exts:
                    continue
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, self.musan_root).lower()
                if rel_path.startswith('noise'):
                    self.musan_file_index['noise'].append(full_path)
                elif rel_path.startswith('speech'):
                    self.musan_file_index['speech'].append(full_path)
                elif rel_path.startswith('music'):
                    self.musan_file_index['music'].append(full_path)

        available_types = [k for k, v in self.musan_file_index.items() if len(v) > 0]
        if not available_types:
            print(f"警告: MUSAN 目录中未发现可用音频: {self.musan_root}，已禁用 MUSAN 增强")
            self.use_musan_noise = False
            return

        total_files = sum(len(v) for v in self.musan_file_index.values())
        print(
            "已启用 MUSAN 噪声增强: "
            f"noise={len(self.musan_file_index['noise'])}, "
            f"speech={len(self.musan_file_index['speech'])}, "
            f"music={len(self.musan_file_index['music'])}, "
            f"total={total_files}"
        )

    def _sample_musan_noise_file(self):
        available_types = [k for k, v in self.musan_file_index.items() if len(v) > 0]
        if not available_types:
            return None

        weighted_types = []
        weighted_values = []
        for noise_type in available_types:
            weight = float(self.musan_weights.get(noise_type, 0.0))
            if weight > 0:
                weighted_types.append(noise_type)
                weighted_values.append(weight)
        if not weighted_types:
            weighted_types = available_types
            weighted_values = [1.0] * len(available_types)

        selected_type = random.choices(weighted_types, weights=weighted_values, k=1)[0]
        return random.choice(self.musan_file_index[selected_type])

    def _load_musan_noise_segment(self, target_length):
        noise_path = self._sample_musan_noise_file()
        if not noise_path:
            return None
        try:
            waveform, sr = torchaudio.load(noise_path)
            if waveform.numel() == 0:
                return None
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            noise = waveform.squeeze(0)
            noise = self._resample_audio(noise, sr)
            if noise.numel() == 0:
                return None
            if noise.numel() < target_length:
                repeat_count = (
                    int(target_length) + max(1, int(noise.numel())) - 1
                ) // max(1, int(noise.numel()))
                noise = noise.repeat(repeat_count)
            if noise.numel() > target_length:
                max_start = int(noise.numel()) - int(target_length)
                start = random.randint(0, max_start) if max_start > 0 else 0
                noise = noise[start:start + int(target_length)]
            return noise
        except Exception:
            return None

    def _mix_with_snr(self, clean_audio, noise_audio, snr_db):
        clean_power = clean_audio.pow(2).mean().clamp_min(1e-8)
        noise_power = noise_audio.pow(2).mean().clamp_min(1e-8)
        target_noise_power = clean_power / (10 ** (float(snr_db) / 10.0))
        scale = torch.sqrt(target_noise_power / noise_power)
        mixed = clean_audio + noise_audio * scale
        peak = mixed.abs().max().clamp_min(1.0)
        mixed = mixed / peak
        return mixed

    def _apply_musan_noise(self, audio):
        if (
            not self.use_musan_noise
            or audio.numel() == 0
            or random.random() > self.musan_apply_prob
        ):
            return audio

        noise = self._load_musan_noise_segment(audio.numel())
        if noise is None or noise.numel() != audio.numel():
            return audio
        snr_db = random.uniform(self.musan_snr_min_db, self.musan_snr_max_db)
        return self._mix_with_snr(audio, noise, snr_db)

    def _extract_mono_audio_from_video_tensor(self, audio_frames):
        """
        read_video 返回的音频张量格式在不同版本可能不同，这里做统一兼容。
        目标输出: [T] 的单声道波形。
        """
        if not isinstance(audio_frames, torch.Tensor) or audio_frames.numel() == 0:
            return torch.zeros(0)

        audio = audio_frames.float()
        if audio.dim() == 1:
            return audio

        if audio.dim() == 2:
            # 常见情况:
            # [num_samples, num_channels] 或 [num_channels, num_samples]
            if audio.shape[0] <= 4 and audio.shape[1] > audio.shape[0]:
                audio = audio.transpose(0, 1)
            if audio.shape[-1] <= 4:
                return audio.mean(dim=-1)
            if audio.shape[0] <= 4:
                return audio.mean(dim=0)
            return audio.mean(dim=-1)

        # 兜底处理更高维情况
        audio = audio.reshape(audio.shape[0], -1)
        return audio.mean(dim=-1)

    def _sample_frame_indices(self, total_frames):
        if total_frames <= 0:
            return torch.zeros(self.num_frames, dtype=torch.long)

        required_span = (self.num_frames - 1) * self.frame_stride + 1
        if total_frames >= required_span:
            start_max = total_frames - required_span
            if self.train:
                start = random.randint(0, start_max) if start_max > 0 else 0
            else:
                start = start_max // 2
            return start + torch.arange(self.num_frames, dtype=torch.long) * self.frame_stride

        if total_frames >= self.num_frames:
            if self.train:
                start_max = total_frames - self.num_frames
                start = random.randint(0, start_max) if start_max > 0 else 0
                return torch.arange(start, start + self.num_frames, dtype=torch.long)
            return torch.linspace(0, total_frames - 1, steps=self.num_frames).long()

        base = torch.arange(total_frames, dtype=torch.long)
        if base.numel() == 0:
            return torch.zeros(self.num_frames, dtype=torch.long)
        pad = base[-1].repeat(self.num_frames - total_frames)
        return torch.cat([base, pad], dim=0)

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _select_face_bbox(self, detections, prev_bbox, frame_w, frame_h):
        candidates = []
        if isinstance(detections, dict):
            det_values = detections.values()
        elif isinstance(detections, list):
            det_values = detections
        else:
            det_values = []

        for det in det_values:
            if not isinstance(det, dict):
                continue
            bbox = det.get('facial_area') or det.get('bbox')
            if bbox is None or len(bbox) < 4:
                continue
            score = float(det.get('score', 0.0))
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            x1 = max(0.0, min(x1, float(frame_w - 1)))
            y1 = max(0.0, min(y1, float(frame_h - 1)))
            x2 = max(x1 + 1.0, min(x2, float(frame_w)))
            y2 = max(y1 + 1.0, min(y2, float(frame_h)))
            if (x2 - x1) < self.face_roi_min_size or (y2 - y1) < self.face_roi_min_size:
                continue
            candidates.append((score, [x1, y1, x2, y2]))

        if not candidates:
            return None

        if prev_bbox is None:
            candidates.sort(key=lambda item: (item[0], (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])), reverse=True)
            best_score, best_bbox = candidates[0]
            if best_score < self.face_roi_score_thr:
                return None
            return best_bbox

        best_bbox = None
        best_value = -1e9
        for score, bbox in candidates:
            iou = self._bbox_iou(prev_bbox, bbox)
            value = 0.65 * iou + 0.35 * max(0.0, score)
            if value > best_value:
                best_value = value
                best_bbox = bbox
        return best_bbox

    def _retinaface_detect_bbox(self, frame_rgb, prev_bbox=None):
        if not self.use_face_roi or RetinaFace is None:
            return None
        try:
            frame_np = frame_rgb
            if isinstance(frame_np, torch.Tensor):
                frame_np = frame_np.detach().cpu().numpy()
            if str(frame_np.dtype) != 'uint8':
                frame_np = frame_np.clip(0, 255).astype('uint8')
            # RetinaFace 常见实现按 BGR 输入。
            frame_bgr = frame_np[:, :, ::-1]
            detections = RetinaFace.detect_faces(frame_bgr)
            return self._select_face_bbox(
                detections=detections,
                prev_bbox=prev_bbox,
                frame_w=frame_np.shape[1],
                frame_h=frame_np.shape[0]
            )
        except Exception:
            return None

    @staticmethod
    def _center_square_bbox(frame_h, frame_w, ratio=0.7):
        side = max(2, int(min(frame_h, frame_w) * ratio))
        cx = frame_w // 2
        cy = frame_h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(frame_w, x1 + side)
        y2 = min(frame_h, y1 + side)
        return [float(x1), float(y1), float(x2), float(y2)]

    def _smooth_and_expand_bbox(self, current_bbox, prev_smooth_bbox, frame_h, frame_w):
        if prev_smooth_bbox is None:
            smooth_bbox = [float(v) for v in current_bbox]
        else:
            alpha = self.face_roi_smooth_alpha
            smooth_bbox = [
                alpha * prev_smooth_bbox[i] + (1.0 - alpha) * float(current_bbox[i])
                for i in range(4)
            ]

        x1, y1, x2, y2 = smooth_bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        side = max(x2 - x1, y2 - y1) * self.face_roi_expand_ratio
        side = max(float(self.face_roi_min_size), side)

        nx1 = max(0.0, cx - side * 0.5)
        ny1 = max(0.0, cy - side * 0.5)
        nx2 = min(float(frame_w), cx + side * 0.5)
        ny2 = min(float(frame_h), cy + side * 0.5)
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            return smooth_bbox, self._center_square_bbox(frame_h, frame_w, ratio=0.75)
        return smooth_bbox, [nx1, ny1, nx2, ny2]

    def _crop_single_frame_with_bbox(self, frame_hwc, bbox):
        frame_h, frame_w = frame_hwc.shape[0], frame_hwc.shape[1]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), frame_w - 1))
        y1 = max(0, min(int(round(y1)), frame_h - 1))
        x2 = max(x1 + 1, min(int(round(x2)), frame_w))
        y2 = max(y1 + 1, min(int(round(y2)), frame_h))
        cropped = frame_hwc[y1:y2, x1:x2, :]
        if cropped.numel() == 0:
            return frame_hwc
        return cropped

    def _apply_face_roi_with_tracking(self, frames_hwc):
        if not self.use_face_roi:
            return frames_hwc
        if RetinaFace is None:
            if not self._face_roi_warning_emitted:
                print("警告: 未安装 RetinaFace，已回退到整帧输入（建议安装 retinaface）")
                self._face_roi_warning_emitted = True
            return frames_hwc
        if frames_hwc.numel() == 0:
            return frames_hwc

        cropped_frames = []
        prev_detected_bbox = None
        prev_smooth_bbox = None
        frame_h = int(frames_hwc.shape[1])
        frame_w = int(frames_hwc.shape[2])
        center_bbox = self._center_square_bbox(frame_h, frame_w, ratio=0.75)

        for i in range(frames_hwc.shape[0]):
            frame = frames_hwc[i]
            detected_bbox = self._retinaface_detect_bbox(frame, prev_bbox=prev_detected_bbox)
            if detected_bbox is None:
                detected_bbox = prev_detected_bbox if prev_detected_bbox is not None else center_bbox
            prev_detected_bbox = detected_bbox
            prev_smooth_bbox, expanded_bbox = self._smooth_and_expand_bbox(
                current_bbox=detected_bbox,
                prev_smooth_bbox=prev_smooth_bbox,
                frame_h=frame_h,
                frame_w=frame_w
            )
            cropped = self._crop_single_frame_with_bbox(frame, expanded_bbox)
            cropped = cropped.permute(2, 0, 1).float().unsqueeze(0) / 255.0
            cropped = F.interpolate(
                cropped,
                size=(self.frame_size, self.frame_size),
                mode='bilinear',
                align_corners=False
            )
            cropped = (cropped.squeeze(0) * 255.0).clamp(0.0, 255.0).to(dtype=frame.dtype).permute(1, 2, 0)
            cropped_frames.append(cropped)
        return torch.stack(cropped_frames, dim=0)

    def _crop_audio_with_video_window(self, audio, audio_sr, frame_indices, video_fps):
        """
        根据采样到的视频窗口截取对应时间段的音频，实现弱对齐。
        与此同时避免裁剪窗口过短：优先围绕视频窗口中心截取至少 segment_length 的音频。
        """
        if (
            not self.align_av_segment
            or audio.numel() == 0
            or video_fps is None
            or float(video_fps) <= 0
            or frame_indices.numel() == 0
        ):
            return audio

        video_fps = float(video_fps)
        start_frame = int(frame_indices[0].item())
        end_frame = int(frame_indices[-1].item()) + 1
        if end_frame <= start_frame:
            return audio

        start_sample = int((start_frame / video_fps) * int(audio_sr))
        end_sample = int((end_frame / video_fps) * int(audio_sr))

        start_sample = max(0, min(start_sample, audio.numel() - 1))
        end_sample = max(start_sample + 1, min(end_sample, audio.numel()))
        window_samples = end_sample - start_sample
        target_samples = max(window_samples, self.segment_length)
        target_samples = min(target_samples, int(audio.numel()))

        center_sample = (start_sample + end_sample) // 2
        target_start = max(0, center_sample - target_samples // 2)
        target_end = target_start + target_samples
        if target_end > audio.numel():
            target_end = int(audio.numel())
            target_start = max(0, target_end - target_samples)
        return audio[target_start:target_end]

    def _decode_aligned_av_clip(self, filepath):
        """
        单次解码同时拿到视频帧和音频轨，减少 I/O 并提升模态对齐质量。
        """
        try:
            video_frames, audio_frames, info = self._read_video_with_optional_window(filepath)
            if video_frames.numel() == 0:
                raise ValueError("empty video")

            frame_indices = self._sample_frame_indices(video_frames.shape[0])
            frames = video_frames[frame_indices]  # [F, H, W, C]
            frames = self._apply_face_roi_with_tracking(frames)
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
            if frames.shape[-1] != self.frame_size or frames.shape[-2] != self.frame_size:
                frames = F.interpolate(
                    frames,
                    size=(self.frame_size, self.frame_size),
                    mode='bilinear',
                    align_corners=False
                )

            if self.train and self.augmentation:
                if random.random() > 0.5:
                    frames = torch.flip(frames, dims=[3])
                if random.random() > 0.7:
                    # 轻量亮度扰动，增强视觉鲁棒性
                    brightness = random.uniform(0.85, 1.15)
                    frames = torch.clamp(frames * brightness, 0.0, 1.0)

            frames = (frames - 0.5) / 0.5

            audio = self._extract_mono_audio_from_video_tensor(audio_frames)
            audio_sr = self.sample_rate
            video_fps = 0.0
            if isinstance(info, dict):
                audio_sr = int(info.get('audio_fps', self.sample_rate))
                video_fps = float(info.get('video_fps', 0.0))

            audio_fallback_used = 0
            if audio.numel() == 0:
                audio = self._load_audio_fallback(filepath)
                audio_fallback_used = 1
            else:
                audio = self._crop_audio_with_video_window(
                    audio=audio,
                    audio_sr=audio_sr,
                    frame_indices=frame_indices,
                    video_fps=video_fps
                )
                audio = self._resample_audio(audio, audio_sr)

            return audio, frames, {
                'audio_fallback_used': audio_fallback_used,
                'video_fallback_used': 0,
            }
        except Exception:
            fallback_audio = self._load_audio_fallback(filepath)
            fallback_video = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
            return fallback_audio, fallback_video, {
                'audio_fallback_used': 1,
                'video_fallback_used': 1,
            }

    @staticmethod
    def _resolve_malloc_trim():
        if os.name == 'nt':
            return None
        try:
            libc = ctypes.CDLL("libc.so.6")
            trim_fn = getattr(libc, "malloc_trim", None)
            if trim_fn is None:
                return None
            trim_fn.argtypes = [ctypes.c_size_t]
            trim_fn.restype = ctypes.c_int
            return trim_fn
        except Exception:
            return None

    def _ensure_malloc_trim(self):
        if self._malloc_trim_initialized:
            return
        self._malloc_trim = self._resolve_malloc_trim()
        self._malloc_trim_initialized = True

    def _maybe_trim_worker_memory(self):
        if self.worker_trim_interval <= 0:
            return
        self._ensure_malloc_trim()
        if self._malloc_trim is None:
            return
        self._samples_since_trim += 1
        if self._samples_since_trim < self.worker_trim_interval:
            return
        self._samples_since_trim = 0
        try:
            self._malloc_trim(0)
        except Exception:
            self._malloc_trim = None

    def _resolve_decode_window_seconds(self, video_fps):
        safe_fps = max(float(video_fps), 1.0)
        frame_window_seconds = (
            ((self.num_frames - 1) * self.frame_stride + 1) / safe_fps
        )
        audio_window_seconds = self.segment_length / max(float(self.sample_rate), 1.0)
        auto_window_seconds = max(frame_window_seconds, audio_window_seconds)
        if self.decode_window_seconds is None:
            return auto_window_seconds
        return max(auto_window_seconds, float(self.decode_window_seconds))

    def _read_video_with_optional_window(self, filepath):
        """
        优先使用时间窗解码，避免对长视频执行整段解码导致主机内存峰值过高。
        """
        if read_video_timestamps is None:
            decode_window = self._resolve_decode_window_seconds(25.0)
            if decode_window > 0:
                try:
                    return read_video(
                        filepath,
                        start_pts=0.0,
                        end_pts=float(decode_window),
                        pts_unit='sec'
                    )
                except Exception:
                    pass
            return read_video(filepath, pts_unit='sec')

        decode_start = None
        decode_end = None
        estimated_fps = 0.0

        try:
            pts_list, pts_fps = read_video_timestamps(filepath, pts_unit='sec')
            if pts_fps is not None:
                estimated_fps = float(pts_fps)
            if pts_list:
                duration_sec = float(pts_list[-1])
                if estimated_fps > 0:
                    duration_sec += 1.0 / estimated_fps
                decode_window = self._resolve_decode_window_seconds(
                    estimated_fps if estimated_fps > 0 else 25.0
                )
                if duration_sec > decode_window and decode_window > 0:
                    max_start = max(0.0, duration_sec - decode_window)
                    if self.train:
                        decode_start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
                    else:
                        decode_start = max_start * 0.5
                    decode_end = decode_start + decode_window
        except Exception:
            decode_start = None
            decode_end = None

        if (
            decode_start is not None
            and decode_end is not None
            and decode_end > decode_start
        ):
            video_frames, audio_frames, info = read_video(
                filepath,
                start_pts=float(decode_start),
                end_pts=float(decode_end),
                pts_unit='sec',
            )
            if isinstance(video_frames, torch.Tensor) and video_frames.numel() > 0:
                if isinstance(info, dict) and estimated_fps > 0 and float(info.get('video_fps', 0.0)) <= 0:
                    info = dict(info)
                    info['video_fps'] = estimated_fps
                return video_frames, audio_frames, info

        return read_video(filepath, pts_unit='sec')

    def _augment_audio(self, audio):
        if not self.augmentation or not self.train:
            return audio

        if audio.numel() == 0:
            return audio

        # 在音视频联合训练中优先保持 A/V 时间对齐，禁用随机时间偏移。
        allow_time_aug = not (self.disable_audio_time_aug_for_avsync and self.align_av_segment)
        if allow_time_aug and len(audio) > self.segment_length:
            max_shift = len(audio) - self.segment_length
            shift = random.randint(0, max_shift)
            audio = audio[shift:shift + self.segment_length]

        if random.random() > 0.5:
            audio = audio * random.uniform(0.8, 1.2)

        audio = self._apply_musan_noise(audio)

        return audio

    def _pad_or_truncate_audio(self, audio):
        if audio.numel() == 0:
            return torch.zeros(self.segment_length)

        if len(audio) > self.segment_length:
            allow_random_crop = (
                self.train and
                not (self.disable_audio_time_aug_for_avsync and self.align_av_segment)
            )
            if allow_random_crop:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                start = (len(audio) - self.segment_length) // 2
                audio = audio[start:start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - len(audio)))
        return audio

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        filepath = self.video_files[idx]
        audio, video, decode_meta = self._decode_aligned_av_clip(filepath)
        audio = self._augment_audio(audio)
        audio_effective_samples = int(min(int(audio.numel()), self.segment_length))
        audio = self._pad_or_truncate_audio(audio).unsqueeze(0)  # [1, T]
        if self.video_return_float16:
            video = video.half()
        label = int(self.speaker_ids[idx])
        self._maybe_trim_worker_memory()

        return {
            'audio': audio,
            'video': video,
            'label': label,
            'audio_effective_samples': audio_effective_samples,
            'audio_fallback_used': int(decode_meta.get('audio_fallback_used', 0)),
            'video_fallback_used': int(decode_meta.get('video_fallback_used', 0)),
        }

    def get_num_speakers(self):
        return len(self.speaker_to_id)
