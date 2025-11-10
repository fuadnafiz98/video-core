import click
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from typing import List, Tuple


class ObjectDominanceAnalyzer:
    def __init__(
        self,
        sample_rate: int = 5,
        conf_threshold: float = 0.5,
        batch_size: int = 8,
        model_name: str = "yolo12n.pt",
        device: str = "cpu",
    ):
        self.sample_rate = max(1, int(sample_rate))
        self.conf_threshold = float(conf_threshold)
        self.batch_size = max(1, int(batch_size))
        self.model_name = model_name
        self.device = device
        try:
            self.model = YOLO(self.model_name)
        except Exception:
            # Try a safe fallback model; ultralytics will download weights if needed
            click.echo(
                click.style(
                    f"  WARNING: failed to load {self.model_name}, falling back to yolov8n.pt",
                    dim=True,
                )
            )
            self.model = YOLO("yolov8n.pt")

        # Resolve person class index
        names = getattr(self.model, "names", None)
        self.person_class = 0
        if names is not None:
            if isinstance(names, dict):
                for k, v in names.items():
                    if str(v).lower() == "person":
                        self.person_class = int(k)
                        break
            else:
                for i, v in enumerate(names):
                    if str(v).lower() == "person":
                        self.person_class = int(i)
                        break

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames: List[np.ndarray] = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total_frames, desc="Reading frames", unit="f", leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_rate == 0:
                h, w = frame.shape[:2]
                max_side = max(h, w)
                if max_side > 640:
                    scale = 640.0 / max_side
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized = cv2.resize(frame, (new_w, new_h))
                else:
                    resized = frame
                frames.append(resized)
            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        return frames

    def _process_batch(self, batch: List[np.ndarray]) -> Tuple[int, int]:
        if not batch:
            return 0, 0

        results = self.model(batch, verbose=False, device=self.device)

        persons = 0
        objects = 0
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            try:
                cls = r.boxes.cls.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
            except Exception:
                continue
            if len(conf) == 0:
                continue
            mask = conf > self.conf_threshold
            valid_cls = cls[mask]
            persons += int((valid_cls == self.person_class).sum())
            objects += int(len(valid_cls) - int((valid_cls == self.person_class).sum()))

        return persons, objects

    def extract(self, video_path: str) -> dict:
        click.echo("")
        click.echo(click.style("PERSON vs OBJECT DOMINANCE", bold=True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        click.echo(click.style(f"  SAMPLE RATE: 1/{self.sample_rate}", dim=True))
        click.echo(click.style(f"  BATCH SIZE: {self.batch_size}", dim=True))
        click.echo(click.style(f"  DEVICE: {self.device}", dim=True))
        click.echo(click.style(f"  MODEL: {self.model_name}", dim=True))
        click.echo(click.style(f"  DURATION: {duration:.2f}s", dim=True))

        frames = self._extract_frames(video_path)
        total_sampled = len(frames)
        click.echo(click.style(f"  SAMPLED FRAMES: {total_sampled}", dim=True))

        if total_sampled == 0:
            return {
                "person_object_ratio": 0.0,
                "total_persons": 0,
                "total_objects": 0,
                "sampled_frames": 0,
            }

        total_persons = 0
        total_objects = 0

        with tqdm(total=total_sampled, desc="YOLO", unit="f") as pbar:
            for i in range(0, total_sampled, self.batch_size):
                batch = frames[i : i + self.batch_size]
                p, o = self._process_batch(batch)
                total_persons += p
                total_objects += o
                pbar.set_postfix({"P": total_persons, "O": total_objects})
                pbar.update(len(batch))

        total = total_persons + total_objects
        ratio = total_persons / total if total > 0 else 0.0

        result = {
            "person_object_ratio": round(ratio, 2),
            "total_persons": int(total_persons),
            "total_objects": int(total_objects),
            "sampled_frames": int(total_sampled),
        }

        click.echo(click.style("\nCOMPLETE", bold=True))
        click.echo(
            click.style(
                f"  Ratio: {result['person_object_ratio']:.2f} ({result['total_persons']} persons / {result['total_objects']} objects)",
                dim=True,
            )
        )
        click.echo(click.style(f"  Samples: {total_sampled}\n", dim=True))

        return result
