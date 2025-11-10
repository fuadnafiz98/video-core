# Video Core

> V.I.D.A.R (Video Intelligent Detection and Analytics Resource)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuadnafiz98/video-core/blob/master/video_core.ipynb)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Sample Input and Output](#sample-input-and-output)
- [Installation](#installation)
- [Usage](#usage)
- [Google Colab](#google-colab)
- [Performance](#performance)
- [Resources](#resources)

## Introduction

This is a experimental video analysis tool built using SceneDetect for detecting scene changes and extracting key frames from videos. It is designed to facilitate video content analysis and summarization.

## Features

- Shot cut detection

  This module detects scene changes in the video using PySceneDetect's content-aware detection algorithm. It identifies significant transitions between scenes and extracts key frames representing each scene.

- Video Motion Analysis

  This module analyzes motion within the video frames to identify segments with significant movement. It helps in understanding the dynamics of the video content.

- OCR text extraction from video frames

  This module extracts text from video frames using Optical Character Recognition (OCR) techniques. It captures textual information present in the video, which can be useful for various applications such as subtitles, captions, or on-screen text analysis.

- Person VS Object dominance detection

  This module utilizes the YOLOv8 object detection model to analyze video frames and determine the dominance of persons versus other objects within the scenes. It provides insights into the content composition of the video.

## Sample Input and Output

A sample input video and its corresponding output JSON file are provided in the repository for reference.

- Input Video: [videos/sample_video.mp4 🡕](https://github.com/fuadnafiz98/video-core/blob/master/videos/sample-video.mp4)
- Output JSON: [results/sample-video/output.json 🡕](https://github.com/fuadnafiz98/video-core/blob/master/results/sample-video/output.json)

## Installation

- Python 3.13 used
- uv used for package management
- Install dependencies:
  ```bash
  uv install
  ```
- Install `tesseract-ocr` for OCR functionality:
  - **Ubuntu**:
    ```bash
    sudo apt-get install tesseract-ocr
    ```
  - **macOS**:
    ```bash
    brew install tesseract
    ```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/fuadnafiz98/video-core.git
   ```
1. Navigate to the project directory:
   ```bash
   cd video-core
   ```
1. Install dependencies:
   ```bash
   uv install
   ```
1. Put sample videos in the `videos/` directory.
1. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

1. Run the analysis script:

   ```bash
    make run
   ```

   or

   ```bash
    uv run python -m src.cli
   ```

   It will prompt to select a video from the `videos/` directory and start the analysis.

   The output json files and extracted frames will be saved in the `results/` directory.

## Google Colab

Run video analysis in the cloud without local setup.

### Quick Start

```bash
  uv run python -m src.cli
```

## Google Colab

Run video analysis in the cloud without local setup.

### Quick Start

1. [Open the notebook in Colab](https://colab.research.google.com/github/fuadnafiz98/video-core/blob/master/video_core.ipynb)

2. Run all cells or:
   - Clone repo
   - Install dependencies
   - Upload video
   - Run analysis
   - Download results

## Performance

I have used multiprocessing to speed up the analysis. Performance may vary based on hardware and video length.
I am continuously working on optimizing the performance further. My future plan is to use GPU acceleration for object detection and OCR tasks to significantly reduce processing time.

## Resources

- [https://www.scenedetect.com/docs/latest/](https://www.scenedetect.com/docs/latest/)
- [https://docs.ultralytics.com/models/yolo12](https://docs.ultralytics.com/models/yolo12)
