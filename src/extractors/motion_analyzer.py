import cv2
import numpy as np
import click


def classify_motion(avg_motion):
    if avg_motion < 1.5:
        return "static"
    elif avg_motion < 4:
        return "low"
    elif avg_motion < 8:
        return "moderate"
    elif avg_motion < 15:
        return "high"
    else:
        return "very_high"


class MotionAnalyzer:
    def __init__(self, sample_rate=5, downscale=2):
        self.sample_rate = sample_rate
        self.downscale = downscale

        self.flow_params = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 13,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.1,
            "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        }

    def _process(self, cap, total_frames):
        motion_magnitudes = []
        prev_gray = None
        frame_idx = 0
        sampled_count = 0

        click.echo("")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                small = cv2.resize(
                    frame,
                    (
                        frame.shape[1] // self.downscale,
                        frame.shape[0] // self.downscale,
                    ),
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, **self.flow_params
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    percentile_90 = np.percentile(mag, 90)
                    avg_motion = np.mean(mag)
                    max_motion = np.max(mag)
                    motion_magnitudes.append(
                        {"avg": avg_motion, "p90": percentile_90, "max": max_motion}
                    )
                    click.echo(
                        click.style(
                            f"\r  MOTION [{frame_idx}/{total_frames if total_frames > 0 else '?'}] Avg: {avg_motion:.2f} P90: {percentile_90:.2f}",
                            dim=True,
                        ),
                        nl=False,
                    )

                prev_gray = gray
                sampled_count += 1

            frame_idx += 1

        click.echo("")
        return motion_magnitudes, sampled_count

    def extract(self, video_path):
        click.echo("")
        click.echo(click.style("MOTION ANALYSIS", bold=True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0.0

        click.echo(click.style(f"  SAMPLE RATE: 1/{self.sample_rate}", dim=True))
        click.echo(click.style(f"  DOWNSCALE: {self.downscale}x", dim=True))
        click.echo(click.style(f"  DURATION: {duration:.2f}s", dim=True))

        motion_magnitudes, sampled_count = self._process(cap, total_frames)

        cap.release()

        if motion_magnitudes:
            avg_motion = np.mean([m["avg"] for m in motion_magnitudes])
            p90_motion = np.mean([m["p90"] for m in motion_magnitudes])
            max_motion = np.max([m["max"] for m in motion_magnitudes])
        else:
            avg_motion = 0.0
            p90_motion = 0.0
            max_motion = 0.0

        intensity = classify_motion(p90_motion)

        click.echo("")
        click.echo(click.style("COMPLETE", bold=True))
        click.echo(
            click.style(
                f"  AVG: {avg_motion:.2f} | P90: {p90_motion:.2f} ({intensity}) | MAX: {max_motion:.2f}",
                dim=True,
            )
        )
        click.echo(click.style(f"  SAMPLES: {sampled_count}", dim=True))
        click.echo("")

        return {
            "average_motion": float(round(avg_motion, 2)),
            "p90_motion": float(round(p90_motion, 2)),
            "max_motion": float(round(max_motion, 2)),
            "motion_intensity": intensity,
            "sampled_frames": int(sampled_count),
        }
