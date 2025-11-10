import cv2
import numpy as np
import pytesseract
import click
import re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class TextAnalyzer:
    def __init__(
        self,
        sample_rate=2,
        lang="eng",
        downscale_width=640,
        workers=None,
        min_confidence=60,
    ):
        self.sample_rate = sample_rate
        self.lang = lang
        self.downscale_width = downscale_width
        self.tesseract_config = "--psm 6 --oem 3 --dpi 150"
        self.workers = workers or max(1, cpu_count() - 1)
        self.min_confidence = min_confidence

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return processed

    def _extract_keywords(self, data):
        high_conf_words = []
        for i, word in enumerate(data["text"]):
            word_clean = word.strip()
            if word_clean and len(word_clean) > 2:
                conf = data["conf"][i]
                if conf >= self.min_confidence:
                    word_lower = re.sub(r"[^\w]", "", word_clean.lower())
                    if word_lower:
                        high_conf_words.append(word_lower)
        return high_conf_words

    def _process_frame(self, frame_data):
        frame_idx, frame = frame_data
        height, width = frame.shape[:2]
        scale = self.downscale_width / width
        new_height = int(height * scale)
        resized = cv2.resize(frame, (self.downscale_width, new_height))

        processed_frame = self._preprocess(resized)
        ocr_data = pytesseract.image_to_data(
            processed_frame,
            lang=self.lang,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT,
        )

        keywords = []
        has_text = False
        if ocr_data and ocr_data["text"]:
            keywords = self._extract_keywords(ocr_data)
            if keywords:
                has_text = True

        return frame_idx, has_text, keywords

    def _process(self, cap, total_frames):
        frames_to_process = []
        frame_idx = 0

        click.echo("")
        click.echo(click.style("  COLLECTING FRAMES", dim=True))

        with tqdm(
            total=total_frames,
            desc="  Sampling",
            bar_format="  {desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}",
            leave=False,
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_rate == 0:
                    frames_to_process.append((frame_idx, frame.copy()))

                pbar.update(1)
                frame_idx += 1

        cap.release()

        text_frames = 0
        all_keywords = []
        sampled_count = len(frames_to_process)

        click.echo("")
        click.echo(click.style(f"  WORKERS: {self.workers}", dim=True))
        click.echo(
            click.style(f"  PROCESSING {sampled_count} FRAMES WITH OCR", dim=True)
        )
        click.echo("")

        with tqdm(
            total=sampled_count,
            desc="OCR Progress",
            bar_format="  {desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            with Pool(processes=self.workers) as pool:
                for result in pool.imap_unordered(
                    self._process_frame, frames_to_process
                ):
                    frame_idx, has_text, keywords = result
                    if has_text:
                        text_frames += 1
                        all_keywords.extend(keywords)
                    pbar.update(1)

        click.echo("")
        return text_frames, sampled_count, all_keywords

    def extract(self, video_path):
        click.echo("")
        click.echo(click.style("TEXT ANALYSIS", bold=True))

        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0.0

        click.echo(click.style(f"  SAMPLE RATE: 1/{self.sample_rate}", dim=True))
        click.echo(
            click.style(f"  DOWNSCALE WIDTH: {self.downscale_width}px", dim=True)
        )
        click.echo(click.style(f"  DURATION: {duration:.2f}s", dim=True))

        text_frames, sampled_count, all_keywords = self._process(cap, total_frames)

        text_present_ratio = text_frames / sampled_count if sampled_count > 0 else 0.0

        top_keywords = []
        if all_keywords:
            keyword_counts = Counter(all_keywords).most_common(10)
            top_keywords = [{"word": kw[0], "count": kw[1]} for kw in keyword_counts]

        elapsed_time = time.time() - start_time

        click.echo("")
        click.echo(click.style("COMPLETE", bold=True))
        click.echo(
            click.style(
                f"  TEXT RATIO: {text_present_ratio:.2f} ({text_frames}/{sampled_count})",
                dim=True,
            )
        )
        click.echo(
            click.style(
                f"  KEYWORDS: {len(top_keywords)} unique",
                dim=True,
            )
        )
        click.echo(
            click.style(
                f"  TIME TAKEN: {elapsed_time:.2f}s",
                dim=True,
            )
        )
        click.echo("")

        return {
            "text_present_ratio": float(round(text_present_ratio, 2)),
            "text_frames": int(text_frames),
            "sampled_frames": int(sampled_count),
            "top_keywords": top_keywords,
            "processing_time_seconds": float(round(elapsed_time, 2)),
        }
