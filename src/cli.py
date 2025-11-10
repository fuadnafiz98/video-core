import click
import json
import sys
import questionary
from pathlib import Path

from src.extractors.shot_cut_detector import ShotCutDetector
from src.extractors.motion_analyzer import MotionAnalyzer
from src.extractors.text_analyzer import TextAnalyzer
from src.extractors.object_dominance import ObjectDominanceAnalyzer

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


def bold(text):
    return click.style(text, bold=True)


def dim(text):
    return click.style(text, dim=True)


def signal_handler(sig, frame):
    click.echo("\n\nINTERRUPT")
    click.echo("SYSTEM HALT")
    sys.exit(0)


def get_videos_from_folder(folder_path):
    videos = []
    folder = Path(folder_path)

    if not folder.exists():
        return videos

    for video_file in folder.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(video_file)

    return sorted(videos)


def select_video_interactive():
    videos_folder = Path("videos")
    videos = get_videos_from_folder(videos_folder)

    if not videos:
        click.echo(bold("NO VIDEOS DETECTED"))
        click.echo(dim(f"FORMATS: {', '.join(sorted(VIDEO_EXTENSIONS))}"))
        return None

    click.echo("")
    click.echo(bold("VIDEO ARCHIVE"))
    click.echo("")

    choices = [v.name for v in videos]

    custom_style = questionary.Style(
        [
            ("selected", "bold"),
            ("pointer", "bold"),
            ("highlighted", "bold"),
        ]
    )

    selected = questionary.select(
        "SELECT",
        choices=choices,
        style=custom_style,
        use_indicator=False,
        use_shortcuts=False,
    ).ask()

    if not selected:
        return None

    for video in videos:
        if video.name == selected:
            return video

    return None


@click.command()
@click.argument("video_path", type=click.Path(exists=True), required=False)
@click.option("--threshold", default=27.0, help="Scene detection sensitivity threshold")
@click.option("--min-scene-len", default=15, help="Minimum scene length in frames")
@click.option("--gpu/--no-gpu", default=True, help="Enable/disable GPU acceleration")
@click.option(
    "--motion-sample-rate", default=5, help="Motion analysis frame sample rate"
)
@click.option("--motion-downscale", default=2, help="Motion analysis downscale factor")
@click.option("--text-sample-rate", default=2, help="Text analysis frame sample rate")
@click.option(
    "--text-downscale-width", default=640, help="Text analysis downscale width"
)
@click.option("--obj-sample-rate", default=5, help="Object analysis frame sample rate")
@click.option("--obj-conf", default=0.5, help="Object analysis confidence threshold")
def main(
    video_path,
    threshold,
    min_scene_len,
    gpu,
    motion_sample_rate,
    motion_downscale,
    text_sample_rate,
    text_downscale_width,
    obj_sample_rate,
    obj_conf,
):
    try:
        click.echo("")
        click.echo(bold("VIDEO CORE ANALYSIS SYSTEM"))
        click.echo(dim("FEATURE EXTRACTION MODULE"))
        click.echo("")

        if video_path:
            selected_video = Path(video_path)
        else:
            selected_video = select_video_interactive()
            if not selected_video:
                return

        click.echo("")
        click.echo(bold("TARGET"))
        click.echo(f"  {selected_video.name}")
        click.echo("")

        detector = ShotCutDetector(
            threshold=threshold, min_scene_len=min_scene_len, use_gpu=gpu
        )
        analyzer = MotionAnalyzer(
            sample_rate=motion_sample_rate,
            downscale=motion_downscale,
        )
        text_analyzer = TextAnalyzer(
            sample_rate=text_sample_rate,
            downscale_width=text_downscale_width,
        )
        obj_analyzer = ObjectDominanceAnalyzer(
            sample_rate=obj_sample_rate, conf_threshold=obj_conf
        )

        click.echo(bold("PROCESSING"))
        click.echo("")

        shot_result = detector.extract(str(selected_video))
        motion_result = analyzer.extract(str(selected_video))
        text_result = text_analyzer.extract(str(selected_video))
        object_result = obj_analyzer.extract(str(selected_video))

        output_data = {
            "video_file": str(selected_video),
            "features": {
                "shot_cuts": shot_result,
                "motion": motion_result,
                "text": text_result,
                "object_dominance": object_result,
            },
        }

        video_name = selected_video.stem
        results_dir = Path("results") / video_name
        results_dir.mkdir(parents=True, exist_ok=True)

        output_file = results_dir / "output.json"

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        click.echo("")
        click.echo(bold("COMPLETE"))
        click.echo(f"  CUTS DETECTED: {shot_result['total_cuts']}")
        click.echo(f"  AVG SCENE: {shot_result['avg_scene_length']}s")
        click.echo(f"  DURATION: {shot_result['duration']}s")
        click.echo(
            f"  MOTION: P90={motion_result['p90_motion']:.2f} ({motion_result['motion_intensity']}) | AVG={motion_result['average_motion']:.2f}"
        )
        click.echo(
            f"  TEXT RATIO: {text_result['text_present_ratio']:.2f} | KEYWORDS: {len(text_result['top_keywords'])}"
        )
        click.echo(
            f"  PERSON/OBJECT RATIO: {object_result['person_object_ratio']:.2f} (P={object_result['total_persons']} O={object_result['total_objects']})"
        )
        click.echo("")
        click.echo(dim(f"OUTPUT: {output_file}"))
        click.echo("")

    except KeyboardInterrupt:
        click.echo("")
        click.echo("")
        click.echo(bold("INTERRUPT"))
        click.echo("SYSTEM HALT")
        sys.exit(0)
    except Exception as e:
        click.echo("")
        click.echo(bold("ERROR"))
        click.echo(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
