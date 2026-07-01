# -*- coding: utf-8 -*-
"""
train_intensity_model.py
========================

JPGU 2026 — 震度預估模型訓練程式
（基於 hugging_face_seismic_model.py 重構，銜接到 1D 時間序列輸入）

設計目標
--------
* 輸入：單一觀測站的「逐秒最大震度」時間序列（整數 0–9，已從三軸取最大），
  shape = (T,)，T = 120 秒（依 data/hist 的 CWB JSON）。
  可用 --input-window N 截短到前 N 秒（早期震度預估）。
* 輸出（多任務）：
  - 震度階級分類（10 類：0,1,2,3,4,5-,5+,6-,6+,7）
  - 連續迴歸值（label 用整數轉 float，便於 ordinal-aware 學習）
* 模型骨架：Hugging Face `Wav2Vec2Model`（針對 1Hz 低取樣率客製 conv stem），
  上接 masked-mean pooling → 分類頭 + 迴歸頭。
* 訓練流程：Hugging Face `Trainer`，自訂 `compute_loss`、`compute_metrics`。

資料 Schema (CWB 格式)
----------------------
{
  "eq_info": {origin_time, longitude, latitude, depth, magnitude, isnumber, number},
  "times":  [120 個 ISO8601 字串],
  "variables": ["intensity", "epicenter_distance"],
  "stids":  {"A001": {city, town, name, elev, lat, lon}, ...},
  "intensity":          {"A001": [int×120], ...},   # 逐秒最大震度
  "epicenter_distance": {"A001": float, ...}
}

每個 (event, station) 組合 → 一筆樣本：
  - signal           = intensity[stid]                     # 長度 = T_window
  - intensity_class  = max(intensity[stid])                # 該站峰值震度
  - intensity_value  = float(max(intensity[stid]))         # 同上、轉 float

子命令
------
inspect    : 探勘一個 JSON 檔的結構（兼用於非 CWB 的 schema）
train      : 訓練模型
predict    : 對單筆 JSON 推論
smoke-test : 合成資料跑通整條 pipeline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# ============================================================================
# 1. JMA 震度 ↔ 連續計測震度 ↔ 類別索引 工具
# ============================================================================

JMA_CLASSES: List[str] = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
JMA_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(JMA_CLASSES)}

JMA_ALIASES: Dict[str, str] = {
    "5弱": "5-", "5jaku": "5-", "5_lower": "5-", "5L": "5-", "5Lower": "5-",
    "5強": "5+", "5kyo": "5+", "5_upper": "5+", "5U": "5+", "5Upper": "5+",
    "6弱": "6-", "6jaku": "6-", "6_lower": "6-", "6L": "6-", "6Lower": "6-",
    "6強": "6+", "6kyo": "6+", "6_upper": "6+", "6U": "6+", "6Upper": "6+",
}


def normalize_intensity_class(value: Any) -> Optional[int]:
    """
    把任意表示的震度轉為 0–9 類別索引（10 級制：0,1,2,3,4,5-,5+,6-,6+,7）。

    分派規則（重要）：
      * 整數 (int) 0–9      → 直接當 class index（CWB JSON 慣例）
      * 整數其他範圍        → 視為連續 Imeas，用閾值轉
      * 浮點 (float)        → 視為連續 Imeas，用閾值轉
      * 字串 "5-" / "5弱"   → 別名表查
      * 字串純整數 "5"      → 當 class index
      * 字串浮點 "5.0"      → 視為 Imeas
      * 其他無法解析        → None
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if 0 <= value <= 9:
            return value
        return imeas_to_class(float(value))
    if isinstance(value, float):
        return imeas_to_class(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s in JMA_CLASS_TO_IDX:
            return JMA_CLASS_TO_IDX[s]
        if s in JMA_ALIASES:
            return JMA_CLASS_TO_IDX[JMA_ALIASES[s]]
        try:
            num = float(s)
        except ValueError:
            return None
        if "." not in s and "e" not in s.lower() and 0 <= int(num) <= 9:
            return int(num)
        return imeas_to_class(num)
    return None


def imeas_to_class(imeas: float) -> int:
    """計測震度（0.0~7.0）轉 10 類索引。閾值依 JMA 公告。"""
    if imeas < 0.5:   return 0
    if imeas < 1.5:   return 1
    if imeas < 2.5:   return 2
    if imeas < 3.5:   return 3
    if imeas < 4.5:   return 4
    if imeas < 5.0:   return 5  # 5-
    if imeas < 5.5:   return 6  # 5+
    if imeas < 6.0:   return 7  # 6-
    if imeas < 6.5:   return 8  # 6+
    return 9                    # 7


def class_to_imeas_centroid(idx: int) -> float:
    centroids = [0.25, 1.0, 2.0, 3.0, 4.0, 4.75, 5.25, 5.75, 6.25, 6.75]
    return centroids[idx]


# ============================================================================
# 2. JSON schema 自動探勘
# ============================================================================

WAVEFORM_HINT_KEYS = (
    "intensity", "imeas", "wave", "data", "signal", "amp", "trace",
    "max_intensity_per_sec", "intensity_per_sec", "per_sec", "sec",
    "震度", "計測震度", "波形",
)
LABEL_HINT_KEYS = (
    "intensity", "intensity_class", "intensity_label", "shindo", "scale",
    "max_intensity", "計測震度", "震度",
)
EVENT_HINT_KEYS = (
    "magnitude", "mag", "M", "lat", "latitude", "lon", "longitude",
    "depth", "origin_time", "time", "event",
)


def walk_schema(obj: Any, path: str = "$", depth: int = 0, max_depth: int = 6,
                seen_lists: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> Dict[str, Dict[str, Any]]:
    if seen_lists is None:
        seen_lists = {}
    if depth > max_depth:
        return seen_lists

    t = type(obj).__name__
    info = seen_lists.setdefault(path, {"type": t, "samples": [], "len": None,
                                        "is_numeric_list": False})

    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_schema(v, f"{path}.{k}", depth + 1, max_depth, seen_lists)
    elif isinstance(obj, list):
        info["len"] = len(obj)
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj[:200]):
            info["is_numeric_list"] = True
            info["samples"] = [round(float(x), 4) for x in obj[:5]]
        elif obj:
            walk_schema(obj[0], f"{path}[]", depth + 1, max_depth, seen_lists)
    else:
        if len(info["samples"]) < 3:
            info["samples"].append(obj)

    return seen_lists


def guess_paths(schema: Dict[str, Dict[str, Any]]
                ) -> Tuple[Optional[str], Optional[str]]:
    waveform_candidates = [
        (p, info) for p, info in schema.items()
        if info.get("is_numeric_list") and (info.get("len") or 0) >= 5
    ]

    def waveform_score(p: str, info: dict) -> float:
        score = float(info.get("len") or 0)
        plower = p.lower()
        for hint in WAVEFORM_HINT_KEYS:
            if hint in plower:
                score *= 1.5
        return score

    waveform_candidates.sort(key=lambda pi: waveform_score(*pi), reverse=True)
    waveform_path = waveform_candidates[0][0] if waveform_candidates else None

    label_candidates: List[Tuple[str, float]] = []
    for p, info in schema.items():
        plower = p.lower().split(".")[-1]
        if any(h in plower for h in LABEL_HINT_KEYS):
            samples = info.get("samples") or []
            if not samples:
                continue
            ok = False
            for s in samples:
                if normalize_intensity_class(s) is not None:
                    ok = True
                    break
            if ok:
                label_candidates.append((p, len(samples)))

    label_path = label_candidates[0][0] if label_candidates else None
    return waveform_path, label_path


# ============================================================================
# 3. JSON 路徑語法（"a.b[].c" → 多筆 leaf 值）
# ============================================================================

def jsonpath_iter(obj: Any, path: str) -> Iterable[Any]:
    if path in ("", "$"):
        yield obj
        return
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]
    parts = _split_path(path)
    yield from _walk(obj, parts)


def _split_path(path: str) -> List[str]:
    out: List[str] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                out.append(buf)
                buf = ""
        elif ch == "[":
            if buf:
                out.append(buf)
                buf = ""
            j = path.find("]", i)
            if j < 0:
                raise ValueError(f"unclosed [ in path: {path}")
            inner = path[i + 1:j]
            out.append(f"[{inner}]")
            i = j
        else:
            buf += ch
        i += 1
    if buf:
        out.append(buf)
    return out


def _walk(obj: Any, parts: List[str]) -> Iterable[Any]:
    if not parts:
        yield obj
        return
    head, rest = parts[0], parts[1:]
    if head == "[]":
        if isinstance(obj, list):
            for item in obj:
                yield from _walk(item, rest)
    elif head.startswith("[") and head.endswith("]"):
        idx = int(head[1:-1])
        if isinstance(obj, list) and -len(obj) <= idx < len(obj):
            yield from _walk(obj[idx], rest)
    else:
        if isinstance(obj, dict) and head in obj:
            yield from _walk(obj[head], rest)


# ============================================================================
# 4. 樣本擷取
# ============================================================================

@dataclass
class Sample:
    signal: List[float]
    intensity_class: int
    intensity_value: float
    source_file: str
    meta: Dict[str, Any] = field(default_factory=dict)


def extract_samples_cwb(
    event: Dict[str, Any],
    source_file: str = "",
    input_window: int = 0,
    label_mode: str = "max",
    min_signal_len: int = 5,
) -> List[Sample]:
    intensity = event.get("intensity")
    if not isinstance(intensity, dict) or not intensity:
        return []

    eq_info = event.get("eq_info") or {}
    stids = event.get("stids") or {}
    epi = event.get("epicenter_distance") or {}

    samples: List[Sample] = []
    for stid, series in intensity.items():
        if not isinstance(series, list) or len(series) < min_signal_len:
            continue
        try:
            full = [float(x) for x in series]
        except (TypeError, ValueError):
            continue

        if label_mode == "last":
            label_int = int(round(full[-1]))
        elif label_mode == "argmax_value":
            label_int = int(round(max(full)))
        else:
            label_int = int(round(max(full)))
        label_int = max(0, min(9, label_int))

        if input_window and input_window > 0:
            sig = full[:input_window]
            if len(sig) < min_signal_len:
                continue
        else:
            sig = full

        samples.append(Sample(
            signal=sig,
            intensity_class=label_int,
            intensity_value=float(label_int),
            source_file=source_file,
            meta={
                "station_id": stid,
                "station_info": stids.get(stid, {}),
                "epicenter_distance": epi.get(stid),
                "eq_info": eq_info,
                "full_length": len(full),
                "input_window": input_window or len(full),
            },
        ))
    return samples


def extract_samples_generic(
    event: Any,
    waveform_path: str,
    label_path: str,
    source_file: str = "",
) -> List[Sample]:
    waves = list(jsonpath_iter(event, waveform_path))
    labels = list(jsonpath_iter(event, label_path))
    if not waves or not labels:
        return []
    n = min(len(waves), len(labels))
    samples: List[Sample] = []
    for i in range(n):
        sig, lbl = waves[i], labels[i]
        if not isinstance(sig, list) or len(sig) == 0:
            continue
        try:
            sig_arr = [float(x) for x in sig]
        except (TypeError, ValueError):
            continue
        cls = normalize_intensity_class(lbl)
        if cls is None:
            continue
        if isinstance(lbl, (int, float)) and not isinstance(lbl, bool):
            ival = float(lbl)
        else:
            try:
                ival = float(str(lbl))
            except ValueError:
                ival = class_to_imeas_centroid(cls)
        samples.append(Sample(signal=sig_arr, intensity_class=cls,
                              intensity_value=ival, source_file=source_file,
                              meta={"index": i}))
    return samples


# ============================================================================
# 5. 載入整個資料夾
# ============================================================================

def iter_event_files(data_dir: Union[str, Path]) -> Iterable[Path]:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"data_dir not found: {p}")
    yield from sorted(p.rglob("*.json"))


def load_dataset_dict(
    data_dir: Union[str, Path],
    schema: str = "cwb",
    waveform_path: Optional[str] = None,
    label_path: Optional[str] = None,
    input_window: int = 0,
    label_mode: str = "max",
    max_files: Optional[int] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_by_event: bool = True,
) -> Tuple["DatasetDict", Dict[str, Any]]:  # noqa: F821
    from datasets import Dataset, DatasetDict

    files = list(iter_event_files(data_dir))
    if max_files is not None:
        files = files[:max_files]

    print(f"[load] 找到 {len(files)} 個 JSON 檔，開始解析（schema={schema}）…",
          flush=True)

    samples_by_event: Dict[str, List[Sample]] = {}
    skipped = 0
    for i, fp in enumerate(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                ev = json.load(f)
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")
            skipped += 1
            continue

        if schema == "cwb":
            ss = extract_samples_cwb(
                ev, source_file=fp.name,
                input_window=input_window, label_mode=label_mode,
            )
        else:
            if not waveform_path or not label_path:
                print(f"  [skip] {fp.name}: schema=generic 但未提供 waveform/label path")
                skipped += 1
                continue
            ss = extract_samples_generic(
                ev, waveform_path, label_path, source_file=fp.name,
            )
        if ss:
            samples_by_event[fp.name] = ss
        if (i + 1) % 50 == 0:
            n_so_far = sum(len(v) for v in samples_by_event.values())
            print(f"  [load] {i+1}/{len(files)} files → {n_so_far} samples",
                  flush=True)

    all_samples: List[Sample] = [s for v in samples_by_event.values() for s in v]
    if not all_samples:
        raise RuntimeError(
            f"在 {data_dir} 內找不到任何可用樣本。請用 `inspect` 看一下 schema。"
        )

    print(f"[load] 完成，共 {len(all_samples)} 筆樣本（{skipped} 個檔案跳過）。")

    cls_counts: Dict[int, int] = {}
    for s in all_samples:
        cls_counts[s.intensity_class] = cls_counts.get(s.intensity_class, 0) + 1
    lengths = [len(s.signal) for s in all_samples]
    stats = {
        "n_samples": len(all_samples),
        "class_counts": {JMA_CLASSES[k]: v for k, v in sorted(cls_counts.items())},
        "len_min": min(lengths), "len_max": max(lengths),
        "len_mean": sum(lengths) / len(lengths),
        "imeas_min": min(s.intensity_value for s in all_samples),
        "imeas_max": max(s.intensity_value for s in all_samples),
    }

    rng = random.Random(seed)
    if split_by_event and len(samples_by_event) >= 3:
        event_names = list(samples_by_event.keys())
        rng.shuffle(event_names)
        n_train_ev = max(1, int(len(event_names) * train_ratio))
        remaining = len(event_names) - n_train_ev
        if remaining < 2:
            n_train_ev = max(1, len(event_names) - 2)
            remaining = len(event_names) - n_train_ev
        n_val_ev = max(1, remaining // 2)
        train_ev = event_names[:n_train_ev]
        val_ev = event_names[n_train_ev:n_train_ev + n_val_ev]
        test_ev = event_names[n_train_ev + n_val_ev:]
        splits = {
            "train":      [s for ev in train_ev for s in samples_by_event[ev]],
            "validation": [s for ev in val_ev   for s in samples_by_event[ev]],
            "test":       [s for ev in test_ev  for s in samples_by_event[ev]],
        }
        stats["split_strategy"] = "by_event"
        stats["n_events"] = {"train": len(train_ev), "val": len(val_ev),
                             "test": len(test_ev)}
    elif split_by_event and len(samples_by_event) < 3:
        print(f"[load] WARN: 只有 {len(samples_by_event)} 個事件，無法按事件切；"
              f"改用隨機切。建議用更多檔案或 --split-random。")
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_ratio))
        remaining = len(indices) - n_train
        if remaining < 2:
            n_train = max(1, len(indices) - 2)
            remaining = len(indices) - n_train
        n_val = max(1, remaining // 2)
        splits = {
            "train":      [all_samples[i] for i in indices[:n_train]],
            "validation": [all_samples[i] for i in indices[n_train:n_train + n_val]],
            "test":       [all_samples[i] for i in indices[n_train + n_val:]],
        }
        stats["split_strategy"] = "random_fallback"
    else:
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_ratio))
        remaining = len(indices) - n_train
        if remaining < 2:
            n_train = max(1, len(indices) - 2)
            remaining = len(indices) - n_train
        n_val = max(1, remaining // 2)
        splits = {
            "train":      [all_samples[i] for i in indices[:n_train]],
            "validation": [all_samples[i] for i in indices[n_train:n_train + n_val]],
            "test":       [all_samples[i] for i in indices[n_train + n_val:]],
        }
        stats["split_strategy"] = "random"

    def to_hf(samples: List[Sample]) -> "Dataset":
        return Dataset.from_dict({
            "signal":          [s.signal for s in samples],
            "intensity_class": [s.intensity_class for s in samples],
            "intensity_value": [s.intensity_value for s in samples],
            "source_file":     [s.source_file for s in samples],
            "station_id":      [s.meta.get("station_id", "") for s in samples],
            "epicenter_distance": [
                float(s.meta.get("epicenter_distance") or 0.0) for s in samples
            ],
        })

    ds = DatasetDict({k: to_hf(v) for k, v in splits.items()})
    return ds, stats


# ============================================================================
# 6. 模型：Wav2Vec2 backbone + 多任務 head
# ============================================================================

def build_model(
    seq_len_hint: int = 256,
    num_classes: int = 10,
    hidden_size: int = 192,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    use_pretrained: bool = False,
):
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    cfg = Wav2Vec2Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 2,
        conv_dim=(64, 128, hidden_size),
        conv_stride=(1, 1, 1),
        conv_kernel=(3, 3, 3),
        feat_extract_norm="layer",
        feat_extract_activation="gelu",
        do_stable_layer_norm=True,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        layerdrop=0.0,
        vocab_size=32,
    )

    class Wav2Vec2IntensityModel(nn.Module):
        def __init__(self, config: Wav2Vec2Config, num_classes: int):
            super().__init__()
            self.config = config
            self.backbone = Wav2Vec2Model(config)
            self.dropout = nn.Dropout(0.1)
            self.cls_head = nn.Linear(config.hidden_size, num_classes)
            self.reg_head = nn.Linear(config.hidden_size, 1)
            self.num_classes = num_classes

        def forward(
            self,
            input_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            intensity_class: Optional[torch.Tensor] = None,
            intensity_value: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            outputs = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state

            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1).long()
                output_lengths = self.backbone._get_feat_extract_output_lengths(
                    input_lengths
                ).to(hidden.device)
                T_out = hidden.shape[1]
                rng = torch.arange(T_out, device=hidden.device).unsqueeze(0)
                frame_mask = (rng < output_lengths.unsqueeze(1)).type_as(hidden)
                frame_mask = frame_mask.unsqueeze(-1)
                pooled = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = hidden.mean(dim=1)

            pooled = self.dropout(pooled)
            cls_logits = self.cls_head(pooled)
            reg_pred = self.reg_head(pooled).squeeze(-1)

            loss = None
            if intensity_class is not None and intensity_value is not None:
                cls_loss = nn.functional.cross_entropy(cls_logits, intensity_class)
                reg_loss = nn.functional.smooth_l1_loss(reg_pred, intensity_value.float())
                loss = 0.7 * reg_loss + 0.3 * cls_loss

            return {
                "loss": loss,
                "logits": cls_logits,
                "regression": reg_pred,
            }

    model = Wav2Vec2IntensityModel(cfg, num_classes=num_classes)

    if use_pretrained:
        try:
            from transformers import Wav2Vec2Model as HFW2V2
            base = HFW2V2.from_pretrained("facebook/wav2vec2-base")
            sd = base.encoder.state_dict()
            missing, unexpected = model.backbone.encoder.load_state_dict(sd, strict=False)
            print(f"[model] 套入 wav2vec2-base.encoder 權重；"
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[model] 載入預訓練權重失敗（將從零開始）：{e}")

    return model


# ============================================================================
# 7. Data collator
# ============================================================================

class SeismicSignalCollator:
    def __init__(self, pad_value: float = 0.0, normalize: bool = True):
        self.pad_value = pad_value
        self.normalize = normalize

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        signals = [f["signal"] for f in features]
        max_len = max(len(s) for s in signals)
        bsz = len(features)
        x = np.full((bsz, max_len), self.pad_value, dtype=np.float32)
        mask = np.zeros((bsz, max_len), dtype=np.int64)
        for i, s in enumerate(signals):
            arr = np.asarray(s, dtype=np.float32)
            if self.normalize:
                mu = float(arr.mean())
                sd = float(arr.std()) + 1e-6
                arr = (arr - mu) / sd
            x[i, :len(arr)] = arr
            mask[i, :len(arr)] = 1

        out = {
            "input_values": torch.from_numpy(x),
            "attention_mask": torch.from_numpy(mask),
            "intensity_class": torch.tensor(
                [f["intensity_class"] for f in features], dtype=torch.long),
            "intensity_value": torch.tensor(
                [f["intensity_value"] for f in features], dtype=torch.float32),
        }
        return out


# ============================================================================
# 8. 評估
# ============================================================================

def make_compute_metrics():
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, (list, tuple)):
            logits, reg_pred = preds[0], preds[1]
        else:
            logits, reg_pred = preds, None

        labels = eval_pred.label_ids
        if isinstance(labels, (list, tuple)):
            cls_label, reg_label = labels[0], labels[1]
        else:
            cls_label = labels
            reg_label = None

        cls_pred = np.argmax(logits, axis=-1)
        acc = float((cls_pred == cls_label).mean())
        off1 = float((np.abs(cls_pred - cls_label) <= 1).mean())

        out = {"accuracy": acc, "off1_accuracy": off1}
        if reg_pred is not None and reg_label is not None:
            mae = float(np.abs(reg_pred - reg_label).mean())
            rmse = float(np.sqrt(((reg_pred - reg_label) ** 2).mean()))
            out["mae"] = mae
            out["rmse"] = rmse
        return out

    return compute_metrics


# ============================================================================
# 9. CLI: inspect / train / predict
# ============================================================================

def cmd_inspect(args: argparse.Namespace) -> None:
    files = list(iter_event_files(args.data_dir))[: args.max_files]
    if not files:
        print(f"[inspect] 在 {args.data_dir} 找不到 *.json")
        return

    merged: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        print(f"\n=== {fp.name} ===")
        with open(fp, "r", encoding="utf-8") as f:
            ev = json.load(f)
        schema = walk_schema(ev, max_depth=args.max_depth)
        for k, v in schema.items():
            if k not in merged:
                merged[k] = v

    for path, info in sorted(merged.items()):
        line = f"  {path}  <{info['type']}>"
        if info.get("len") is not None:
            line += f"  len={info['len']}"
        if info.get("is_numeric_list"):
            line += "  [numeric_list]"
        if info.get("samples"):
            line += f"  samples={info['samples']}"
        print(line)

    wp, lp = guess_paths(merged)
    print("\n=== 自動猜測 (generic schema) ===")
    print(f"  --waveform-path  {wp}")
    print(f"  --label-path     {lp}")

    if files:
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                ev = json.load(f)
            cwb_samples = extract_samples_cwb(ev, source_file=files[0].name)
            if cwb_samples:
                print(f"\n=== CWB schema 預覽（{files[0].name}） ===")
                print(f"  抽出 {len(cwb_samples)} 筆樣本（每站一筆）")
                first = cwb_samples[0]
                print(f"  樣本範例：")
                print(f"    station_id  = {first.meta.get('station_id')}")
                print(f"    signal[:10] = {first.signal[:10]}")
                print(f"    signal len  = {len(first.signal)}")
                print(f"    label class = {JMA_CLASSES[first.intensity_class]} "
                      f"(idx={first.intensity_class})")
                print(f"    epi_dist    = {first.meta.get('epicenter_distance')}")
                from collections import Counter
                cnt = Counter(s.intensity_class for s in cwb_samples)
                dist = {JMA_CLASSES[k]: cnt[k] for k in sorted(cnt)}
                print(f"    本檔 label 分佈：{dist}")
                print("\n  → CWB schema 看起來可用，直接 train 即可：")
                print(f'    python {Path(__file__).name} train --data-dir "{args.data_dir}"')
        except Exception as e:
            print(f"\n  [warn] CWB 預覽失敗：{e}")


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from transformers import TrainingArguments, Trainer

    ds, stats = load_dataset_dict(
        data_dir=args.data_dir,
        schema=args.schema,
        waveform_path=args.waveform_path,
        label_path=args.label_path,
        input_window=args.input_window,
        label_mode=args.label_mode,
        max_files=args.max_files,
        seed=args.seed,
        split_by_event=not args.split_random,
    )
    print(f"[train] 資料集：{ds}")
    print(f"[train] 統計：{json.dumps(stats, ensure_ascii=False, indent=2)}")

    model = build_model(
        num_classes=len(JMA_CLASSES),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        use_pretrained=args.use_pretrained,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] 可訓練參數：{n_params:,}")

    collator = SeismicSignalCollator(normalize=not args.no_normalize)

    use_cpu_flag = args.cpu or (not torch.cuda.is_available())
    targs = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-2,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
        use_cpu=use_cpu_flag,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["intensity_class", "intensity_value"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )

    resume = args.resume_from_checkpoint
    if resume:
        if resume.lower() in ("auto", "true", "1"):
            print("[train] resume_from_checkpoint=auto → "
                  f"從 {args.output_dir} 內最新 checkpoint 接著跑")
            trainer.train(resume_from_checkpoint=True)
        else:
            print(f"[train] 從 checkpoint 接著跑：{resume}")
            trainer.train(resume_from_checkpoint=resume)
    else:
        print("[train] 開始訓練…")
        trainer.train()

    print("\n[train] 在 test split 評估…")
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    save_dir = Path(args.output_dir) / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    with open(save_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump({
            "schema": args.schema,
            "waveform_path": args.waveform_path,
            "label_path": args.label_path,
            "input_window": args.input_window,
            "label_mode": args.label_mode,
            "num_classes": len(JMA_CLASSES),
            "classes": JMA_CLASSES,
            "config": model.config.to_dict(),
            "stats": stats,
            "test_metrics": test_metrics,
        }, f, ensure_ascii=False, indent=2)
    print(f"[train] 模型已存到 {save_dir}")


def cmd_predict(args: argparse.Namespace) -> None:
    import torch
    save_dir = Path(args.model_dir)
    with open(save_dir / "schema.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = build_model(
        num_classes=meta["num_classes"],
        hidden_size=meta["config"]["hidden_size"],
        num_hidden_layers=meta["config"]["num_hidden_layers"],
        num_attention_heads=meta["config"]["num_attention_heads"],
        use_pretrained=False,
    )
    model.load_state_dict(torch.load(save_dir / "pytorch_model.bin", map_location="cpu", weights_only=True))
    model.eval()

    with open(args.input, "r", encoding="utf-8") as f:
        ev = json.load(f)

    schema_kind = meta.get("schema", "cwb")
    if schema_kind == "cwb":
        samples = extract_samples_cwb(
            ev,
            source_file=Path(args.input).name,
            input_window=meta.get("input_window", 0),
            label_mode=meta.get("label_mode", "max"),
        )
    else:
        samples = extract_samples_generic(
            ev,
            waveform_path=meta["waveform_path"],
            label_path=meta["label_path"],
            source_file=Path(args.input).name,
        )
    if not samples:
        print("[predict] 沒有可用的樣本（檢查 schema）。")
        return

    collator = SeismicSignalCollator()
    batch = collator([{
        "signal": s.signal,
        "intensity_class": s.intensity_class,
        "intensity_value": s.intensity_value,
    } for s in samples])

    with torch.no_grad():
        out = model(input_values=batch["input_values"],
                    attention_mask=batch["attention_mask"])
    cls_pred = out["logits"].argmax(dim=-1).tolist()
    reg_pred = out["regression"].tolist()

    print(f"\n=== {args.input} ({len(samples)} 筆樣本) ===")
    print(f"{'stid':>6} {'pred_cls':>8} {'pred_val':>9} {'true_cls':>8} {'true_val':>9}")
    for i, s in enumerate(samples):
        stid = s.meta.get("station_id", str(i))
        print(f"{stid:>6} {JMA_CLASSES[cls_pred[i]]:>8} {reg_pred[i]:>9.3f} "
              f"{JMA_CLASSES[s.intensity_class]:>8} {s.intensity_value:>9.3f}")


# ============================================================================
# 10. Smoke test
# ============================================================================

def _make_synthetic_event(seed: int, n_stations: int = 6, T: int = 120) -> Dict[str, Any]:
    rng = random.Random(seed)
    times = [f"2024-01-01T00:{m:02d}:{s:02d}"
             for m in range(2) for s in range(60)][:T]
    intensity: Dict[str, List[int]] = {}
    epi: Dict[str, float] = {}
    stids: Dict[str, Dict[str, Any]] = {}
    for i in range(n_stations):
        stid = f"S{i:03d}"
        peak = rng.randint(2, 7)
        peak_at = rng.randint(20, T - 10)
        series = []
        for t in range(T):
            base = rng.choice([0, 0, 0, 1])
            if abs(t - peak_at) < 10:
                v = max(base, peak - abs(t - peak_at) // 2)
            else:
                v = base
            series.append(int(max(0, min(9, v))))
        intensity[stid] = series
        epi[stid] = rng.uniform(10, 200)
        stids[stid] = {"city": "TEST", "town": "x", "name": stid,
                       "elev": rng.uniform(0, 1000),
                       "lat": 23.0 + rng.uniform(-1, 1),
                       "lon": 121.0 + rng.uniform(-1, 1)}
    return {
        "eq_info": {
            "origin_time": "2024-01-01T00:00:05",
            "longitude": 121.5, "latitude": 23.5,
            "depth": 10.0, "magnitude": 6.2,
            "isnumber": 1, "number": "TEST_EVENT",
        },
        "times": times,
        "variables": ["intensity", "epicenter_distance"],
        "stids": stids,
        "intensity": intensity,
        "epicenter_distance": epi,
    }


class _Check:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.fails: List[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.fails.append(f"{name} {detail}".strip())

    def summary(self) -> int:
        print(f"\n  Summary: {self.passed} passed / {self.failed} failed")
        return 0 if self.failed == 0 else 1


def cmd_smoke_test(args: argparse.Namespace) -> None:
    import tempfile

    print("=" * 70)
    print("Smoke test (合成資料、不依賴 data/hist)")
    print("=" * 70)
    chk = _Check()
    rc = 0

    print("\n[A] 純資料路徑 (numpy)")
    try:
        chk("imeas_to_class boundaries",
            imeas_to_class(0.49) == 0 and imeas_to_class(0.5) == 1
            and imeas_to_class(4.99) == 5 and imeas_to_class(5.0) == 6
            and imeas_to_class(7.0) == 9)
        ok = (normalize_intensity_class("5弱") ==
              normalize_intensity_class("5-") ==
              normalize_intensity_class(5))
        chk("normalize_intensity_class 對 5弱 / 5- / 5 一致", ok,
            f"got {normalize_intensity_class('5弱')}, "
            f"{normalize_intensity_class('5-')}, "
            f"{normalize_intensity_class(5)}")
        chk("_split_path stations[].x",
            _split_path("stations[].max_intensity_per_sec") ==
            ["stations", "[]", "max_intensity_per_sec"])
        sample = {"a": [{"b": 1}, {"b": 2}, {"b": 3}]}
        chk("jsonpath_iter a[].b",
            list(jsonpath_iter(sample, "a[].b")) == [1, 2, 3])
        ev = _make_synthetic_event(seed=0, n_stations=4, T=60)
        ss = extract_samples_cwb(ev, source_file="syn.json", input_window=30)
        chk("extract_samples_cwb 樣本數 == 站數",
            len(ss) == 4, f"got {len(ss)}")
        chk("signal 長度 == input_window",
            all(len(s.signal) == 30 for s in ss),
            f"lens={[len(s.signal) for s in ss]}")
        chk("intensity_class == max(整段)",
            all(s.intensity_class == max(ev["intensity"][s.meta["station_id"]])
                for s in ss))
        schema = walk_schema(ev)
        any_numeric = any(info.get("is_numeric_list")
                          for p, info in schema.items() if "intensity." in p)
        chk("walk_schema 對 intensity.* 偵測到 numeric_list", any_numeric)
    except Exception as e:
        chk(f"layer-A 例外：{type(e).__name__}: {e}", False)
        rc = 1

    print("\n[B] 模型與訓練 (torch + transformers)")
    try:
        import torch
    except ImportError as e:
        print(f"  [SKIP] 未安裝 torch：{e}")
        chk("torch 可載入", False, "未安裝")
        sys.exit(chk.summary() or rc or 1)

    try:
        import torch.nn as nn  # noqa: F401
        from transformers import TrainingArguments, Trainer

        torch.manual_seed(0)
        model = build_model(num_classes=10, hidden_size=64,
                            num_hidden_layers=2, num_attention_heads=2,
                            use_pretrained=False)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        chk("build_model 可建立", n_params > 0, f"params={n_params:,}")

        T_test = 30
        B_test = 4
        x = torch.randn(B_test, T_test)
        m = torch.ones(B_test, T_test, dtype=torch.long)
        out = model(input_values=x, attention_mask=m)
        chk("forward logits shape",
            tuple(out["logits"].shape) == (B_test, 10),
            str(tuple(out["logits"].shape)))
        chk("forward regression shape",
            tuple(out["regression"].shape) == (B_test,),
            str(tuple(out["regression"].shape)))

        cls_lbl = torch.randint(0, 10, (B_test,))
        reg_lbl = cls_lbl.float()
        out = model(input_values=x, attention_mask=m,
                    intensity_class=cls_lbl, intensity_value=reg_lbl)
        loss = out["loss"]
        chk("loss 是純量 tensor",
            torch.is_tensor(loss) and loss.ndim == 0,
            str(loss))
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        chk("backward 後至少一個參數有梯度", has_grad)

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "m.bin"
            torch.save(model.state_dict(), save_path)
            model2 = build_model(num_classes=10, hidden_size=64,
                                 num_hidden_layers=2, num_attention_heads=2,
                                 use_pretrained=False)
            model2.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
            model.eval(); model2.eval()
            with torch.no_grad():
                o1 = model(input_values=x, attention_mask=m)
                o2 = model2(input_values=x, attention_mask=m)
            same_logits = torch.allclose(o1["logits"], o2["logits"], atol=1e-5)
            same_reg = torch.allclose(o1["regression"], o2["regression"], atol=1e-5)
        chk("save / load roundtrip 一致", same_logits and same_reg)

        from datasets import Dataset, DatasetDict
        synth_events = [_make_synthetic_event(seed=i, n_stations=4, T=60)
                        for i in range(6)]
        all_samples = []
        for i, ev in enumerate(synth_events):
            all_samples.extend(extract_samples_cwb(
                ev, source_file=f"syn{i}.json", input_window=30))

        def _to_hf(samples):
            return Dataset.from_dict({
                "signal":          [s.signal for s in samples],
                "intensity_class": [s.intensity_class for s in samples],
                "intensity_value": [s.intensity_value for s in samples],
                "source_file":     [s.source_file for s in samples],
                "station_id":      [s.meta.get("station_id", "") for s in samples],
                "epicenter_distance": [
                    float(s.meta.get("epicenter_distance") or 0.0) for s in samples
                ],
            })
        n = len(all_samples)
        ds = DatasetDict({
            "train": _to_hf(all_samples[: int(n * 0.7)]),
            "validation": _to_hf(all_samples[int(n * 0.7):]),
        })

        with tempfile.TemporaryDirectory() as tmp:
            targs = TrainingArguments(
                output_dir=tmp,
                eval_strategy="epoch",
                save_strategy="no",
                logging_strategy="no",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                learning_rate=1e-3,
                weight_decay=0.0,
                report_to=[],
                use_cpu=True,
                dataloader_num_workers=0,
                remove_unused_columns=False,
                label_names=["intensity_class", "intensity_value"],
                seed=0,
            )
            trainer = Trainer(
                model=build_model(num_classes=10, hidden_size=32,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  use_pretrained=False),
                args=targs,
                train_dataset=ds["train"],
                eval_dataset=ds["validation"],
                data_collator=SeismicSignalCollator(),
                compute_metrics=make_compute_metrics(),
            )
            trainer.train()
            metrics = trainer.evaluate()
            chk("Trainer 1 epoch + evaluate 無例外",
                "eval_mae" in metrics, str(list(metrics.keys())[:5]))
    except Exception as e:
        import traceback
        print("  [TRACE]", traceback.format_exc())
        chk(f"layer-B 例外：{type(e).__name__}: {e}", False)
        rc = 1

    sys.exit(chk.summary() or rc)


# ============================================================================
# 11. main
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="JPGU 2026 震度預估訓練程式（基於 Wav2Vec2 多任務微調）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="探勘 JSON schema、自動猜 waveform/label 路徑")
    pi.add_argument("--data-dir", required=True)
    pi.add_argument("--max-files", type=int, default=3)
    pi.add_argument("--max-depth", type=int, default=6)
    pi.set_defaults(func=cmd_inspect)

    pt = sub.add_parser("train", help="訓練模型")
    pt.add_argument("--data-dir", required=True)
    pt.add_argument("--output-dir", default="./output_dir/intensity_w2v2")
    pt.add_argument("--schema", choices=["cwb", "generic"], default="cwb",
                    help="cwb=直接吃 data/hist 的格式（預設）；generic=用 JSONPath")
    pt.add_argument("--input-window", type=int, default=0,
                    help="輸入序列只取前 N 秒（早期震度預估）；0 = 全部使用")
    pt.add_argument("--label-mode", choices=["max", "last", "argmax_value"],
                    default="max",
                    help="如何從整段時序產生標籤；預設 max=峰值震度")
    pt.add_argument("--waveform-path", default=None,
                    help="(僅 schema=generic) JSON path to waveform list")
    pt.add_argument("--label-path", default=None,
                    help="(僅 schema=generic) JSON path to label")
    pt.add_argument("--split-random", action="store_true",
                    help="按樣本切（預設按事件切，避免洩題）")
    pt.add_argument("--max-files", type=int, default=None)
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--hidden-size", type=int, default=192)
    pt.add_argument("--num-layers", type=int, default=4)
    pt.add_argument("--num-heads", type=int, default=4)
    pt.add_argument("--use-pretrained", action="store_true",
                    help="嘗試套用 facebook/wav2vec2-base 的 encoder 權重")
    pt.add_argument("--no-normalize", action="store_true",
                    help="關閉 per-sample z-score normalization")
    pt.add_argument("--cpu", action="store_true",
                    help="強制使用 CPU（預設自動判斷）")
    pt.add_argument("--resume-from-checkpoint", default=None,
                    help="從指定 checkpoint 目錄續訓；填 'auto' 讓 Trainer 自動挑 output_dir 內最新的")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="對單一 JSON 推論")
    pp.add_argument("--model-dir", required=True,
                    help="train 階段的 output_dir/final")
    pp.add_argument("--input", required=True, help="JSON 檔路徑")
    pp.set_defaults(func=cmd_predict)

    ps = sub.add_parser(
        "smoke-test",
        help="合成資料跑通整條 pipeline（不需要 data/hist 也不需要外網）")
    ps.set_defaults(func=cmd_smoke_test)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
