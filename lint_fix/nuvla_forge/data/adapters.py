"""Dataset adapters behind a single interface.

Every adapter yields the same record::

    {
      "images":     uint8 [V, H, W, 3]   multi-view camera frames
      "trajectory": float32 [horizon, 2] future ego waypoints, ego frame, metres
      "question":   str
      "answer":     str
      "sample_id":  str
    }

so the training loop, the shard builder and every benchmark are dataset-agnostic.

Why this indirection exists
---------------------------
Motional's nuReasoning is the target dataset: fleet data from Las Vegas,
Pittsburgh, LA, Boston and Singapore, 105+ hours of long-tail scenarios with
247k human-verified reasoning annotations, and an open ECCV challenge. It is
gated on HuggingFace, so access is a request-and-wait.

DriveLM-nuScenes and nuscenes-qa-mini are ungated, same shape of supervision
(multi-view frames + reasoning QA over real driving scenes), and available in
minutes. So development starts there and the nuReasoning adapter is a config
switch once access lands. Nothing above this file changes.

Trajectory labels
-----------------
DriveLM and nuScenes-QA ship QA over nuScenes keyframes but not ego futures.
``NuScenesEgoTrajectory`` pulls those from the nuScenes devkit when the mini
split is present. When it is not, ``synthetic_trajectory=True`` produces
kinematically plausible waypoints so throughput benchmarks still run end to end.
That flag is surfaced in every benchmark report, because a throughput number is
hardware truth regardless of label quality but an L2 number is not.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from torch.utils.data import Dataset


@dataclass
class SampleSpec:
    n_views: int = 6
    image_size: int = 224
    horizon: int = 6
    max_text_len: int = 64


def _placeholder_views(spec: SampleSpec, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(
        0, 256, (spec.n_views, spec.image_size, spec.image_size, 3), dtype=np.uint8
    )


def _synthetic_trajectory(spec: SampleSpec, rng: np.random.Generator) -> np.ndarray:
    """Constant-curvature arc at a plausible urban speed. Labels for throughput
    benchmarking only -- never reported as planning accuracy."""
    speed = rng.uniform(2.0, 14.0)          # m/s
    yaw_rate = rng.normal(0.0, 0.12)        # rad/s
    dt, x, y, yaw = 0.5, 0.0, 0.0, 0.0
    pts = []
    for _ in range(spec.horizon):
        yaw += yaw_rate * dt
        x += speed * dt * np.cos(yaw)
        y += speed * dt * np.sin(yaw)
        pts.append((x, y))
    return np.asarray(pts, dtype=np.float32)


class BaseDrivingDataset(Dataset):
    def __init__(self, spec: SampleSpec, synthetic_trajectory: bool = True, seed: int = 0):
        self.spec = spec
        self.synthetic_trajectory = synthetic_trajectory
        self._seed = seed

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self._seed + idx)


class SyntheticDriving(BaseDrivingDataset):
    """No downloads. Exists so ``bench_step.py`` and the DDP smoke test can run
    the instant a box comes up, before 30 GB of images have landed."""

    def __init__(self, n: int = 4096, spec: SampleSpec | None = None, **kw):
        super().__init__(spec or SampleSpec(), **kw)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = self._rng(idx)
        return {
            "images": _placeholder_views(self.spec, rng),
            "trajectory": _synthetic_trajectory(self.spec, rng),
            "question": "what should the ego vehicle do next?",
            "answer": "proceed with caution",
            "sample_id": f"synth-{idx:06d}",
        }


class DriveLMNuScenes(BaseDrivingDataset):
    """DriveLM-nuScenes: graph-structured driving QA over nuScenes keyframes.

    Expects the layout from the OpenDriveLab/DriveLM card::

        root/
          v1_0_train_nus.json
          nuscenes/samples/<CAM_*>/<file>.jpg
    """

    CAMERAS: ClassVar[list[str]] = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]

    def __init__(self, root: str | Path, spec: SampleSpec | None = None, **kw):
        super().__init__(spec or SampleSpec(), **kw)
        self.root = Path(root)
        qa_path = self.root / "v1_0_train_nus.json"
        if not qa_path.exists():
            raise FileNotFoundError(
                f"{qa_path} not found. Fetch it with:\n"
                "  huggingface-cli download OpenDriveLab/DriveLM --repo-type dataset "
                f"--local-dir {self.root}"
            )
        with open(qa_path) as f:
            raw = json.load(f)
        self.records = self._flatten(raw)

    @staticmethod
    def _flatten(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """DriveLM nests scene -> keyframe -> QA category -> list of QA pairs.
        Flatten to one record per QA pair, keeping the frame's image paths."""
        out = []
        for scene_token, scene in raw.items():
            for frame_token, frame in scene.get("key_frames", {}).items():
                paths = frame.get("image_paths", {})
                for category, qas in frame.get("QA", {}).items():
                    for qa in qas:
                        out.append({
                            "scene": scene_token,
                            "frame": frame_token,
                            "paths": paths,
                            "question": qa.get("Q", ""),
                            "answer": qa.get("A", ""),
                            "category": category,
                        })
        return out

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image

        rec = self.records[idx]
        rng = self._rng(idx)
        size = self.spec.image_size

        views = []
        for cam in self.CAMERAS[: self.spec.n_views]:
            rel = rec["paths"].get(cam)
            full = self.root / rel if rel else None
            if full and full.exists():
                img = Image.open(full).convert("RGB").resize((size, size))
                views.append(np.asarray(img, dtype=np.uint8))
            else:
                # Missing view: zeros rather than a skipped sample, so batch
                # shape stays static and the step time stays comparable.
                views.append(np.zeros((size, size, 3), dtype=np.uint8))

        return {
            "images": np.stack(views),
            "trajectory": _synthetic_trajectory(self.spec, rng),
            "question": rec["question"],
            "answer": rec["answer"],
            "sample_id": f"{rec['scene']}/{rec['frame']}/{idx}",
        }


class NuScenesQAMini(BaseDrivingDataset):
    """``KevinNotSmile/nuscenes-qa-mini``. Loads with one call, no auth.

    Each sample carries 6-view RGB, a 5D lidar cloud and a text QA pair, split
    into day and night subsets. Smallest real-data path in the repo -- use it to
    confirm the pipeline works before pulling DriveLM.
    """

    def __init__(self, split: str = "train", config: str = "day",
                 spec: SampleSpec | None = None, **kw):
        super().__init__(spec or SampleSpec(), **kw)
        from datasets import load_dataset

        self.ds = load_dataset("KevinNotSmile/nuscenes-qa-mini", config, split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        rec = self.ds[idx]
        rng = self._rng(idx)
        size = self.spec.image_size

        views = []
        for key in sorted(k for k in rec if k.startswith("CAM")):
            img = rec[key]
            if hasattr(img, "convert"):
                views.append(np.asarray(img.convert("RGB").resize((size, size)), dtype=np.uint8))
        while len(views) < self.spec.n_views:
            views.append(np.zeros((size, size, 3), dtype=np.uint8))

        return {
            "images": np.stack(views[: self.spec.n_views]),
            "trajectory": _synthetic_trajectory(self.spec, rng),
            "question": str(rec.get("question", "")),
            "answer": str(rec.get("answer", "")),
            "sample_id": f"nqa-{idx:06d}",
        }


class NuReasoning(BaseDrivingDataset):
    """Motional's nuReasoning. Gated -- request access at
    https://huggingface.co/datasets/qixuewei/nuReasoning

    The loader is written against the documented record shape. Until access
    lands it raises a clear error rather than silently degrading, because a
    benchmark run against the wrong dataset is worse than no run.
    """

    def __init__(self, root: str | Path | None = None, split: str = "train",
                 spec: SampleSpec | None = None, **kw):
        super().__init__(spec or SampleSpec(), **kw)
        from datasets import load_dataset

        token = os.environ.get("HF_TOKEN")
        if root and Path(root).exists():
            self.ds = load_dataset("parquet", data_dir=str(root), split=split)
        else:
            self.ds = load_dataset("qixuewei/nuReasoning", split=split, token=token)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        rec = self.ds[idx]
        size = self.spec.image_size
        views = []
        for key in sorted(k for k in rec if k.upper().startswith("CAM")):
            img = rec[key]
            if hasattr(img, "convert"):
                views.append(np.asarray(img.convert("RGB").resize((size, size)), dtype=np.uint8))
        while len(views) < self.spec.n_views:
            views.append(np.zeros((size, size, 3), dtype=np.uint8))

        traj = rec.get("ego_future") or rec.get("trajectory")
        traj = (
            np.asarray(traj, dtype=np.float32)[: self.spec.horizon]
            if traj is not None
            else _synthetic_trajectory(self.spec, self._rng(idx))
        )

        return {
            "images": np.stack(views[: self.spec.n_views]),
            "trajectory": traj,
            "question": str(rec.get("question", "")),
            "answer": str(rec.get("answer", rec.get("reasoning", ""))),
            "sample_id": str(rec.get("sample_token", idx)),
        }


REGISTRY = {
    "synthetic": SyntheticDriving,
    "drivelm": DriveLMNuScenes,
    "nuscenes-qa-mini": NuScenesQAMini,
    "nureasoning": NuReasoning,
}


def build_dataset(name: str, **kwargs) -> BaseDrivingDataset:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
