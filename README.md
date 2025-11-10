# Video Core

> V.I.D.A.R (Video Intelligent Detection and Analytics Resource)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuadnafiz98/video-core/blob/master/video_core.ipynb)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Google Colab](#google-colab)
- [Performance](#performance)
- [Resources](#resources)

## Introduction

This is a experimental video analysis tool built using SceneDetect for detecting scene changes and extracting key frames from videos. It is designed to facilitate video content analysis and summarization.

## Features

- Shot cut detection
- Video Motion Analysis
- OCR text extraction from video frames
- Person VS Object dominance detection

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

## Resources

- [https://www.scenedetect.com/docs/latest/](https://www.scenedetect.com/docs/latest/)
-
