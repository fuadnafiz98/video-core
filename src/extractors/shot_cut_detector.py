from scenedetect import detect, ContentDetector
import cv2
import click


class ShotCutDetector:
    def __init__(self, threshold=27.0, min_scene_len=15, use_gpu=False):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def extract(self, video_path):
        click.echo("")
        click.echo(click.style("INIT", bold=True))

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()

        click.echo(click.style(f"  DURATION: {duration:.2f}s", dim=True))
        click.echo(click.style(f"  FRAMES: {total_frames}", dim=True))
        click.echo(click.style(f"  FPS: {fps:.2f}", dim=True))
        click.echo("")
        click.echo(click.style("SCANNING", bold=True))

        scene_list = detect(
            video_path,
            ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len),
            show_progress=True,
        )

        click.echo("")
        click.echo(click.style("PROCESSING", bold=True))

        cuts = []
        for i in range(len(scene_list) - 1):
            cut_time = scene_list[i][1].get_seconds()
            cuts.append(round(cut_time, 2))

        if len(scene_list) > 1:
            scene_lengths = [
                (scene_list[i + 1][0] - scene_list[i][0]).get_seconds()
                for i in range(len(scene_list) - 1)
            ]
            avg_scene_length = sum(scene_lengths) / len(scene_lengths)
        else:
            avg_scene_length = 0

        return {
            "total_cuts": len(cuts),
            "cut_timestamps": cuts,
            "avg_scene_length": round(avg_scene_length, 2),
            "scene_count": len(scene_list),
            "duration": round(duration, 2),
        }
