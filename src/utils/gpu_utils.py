"""GPU utility functions for hardware-accelerated video processing."""

import cv2
import numpy as np


class GPUContext:
    """Manages GPU/CPU context for video processing."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.cuda_available = False
            self.device_count = 0
            self._check_cuda()
            GPUContext._initialized = True

    def _check_cuda(self):
        try:
            self.device_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_available = self.device_count > 0

            if self.cuda_available:
                test = cv2.cuda.GpuMat(1, 1, cv2.CV_8UC1)
                test.upload(np.zeros((1, 1), dtype=np.uint8))
        except Exception:
            self.cuda_available = False
            self.device_count = 0

    def is_available(self):
        return self.cuda_available

    def get_device_count(self):
        return self.device_count


def upload_to_gpu(frame):
    gpu_frame = cv2.cuda.GpuMat()
    gpu_frame.upload(frame)
    return gpu_frame


def download_from_gpu(gpu_frame):
    return gpu_frame.download()


def create_gpu_optical_flow():
    ctx = GPUContext()
    if not ctx.is_available():
        return None

    try:
        return cv2.cuda.FarnebackOpticalFlow_create(
            numLevels=3,
            pyrScale=0.5,
            fastPyramids=False,
            winSize=15,
            numIters=2,
            polyN=5,
            polySigma=1.1,
            flags=0,
        )
    except Exception:
        return None


def resize_gpu(gpu_frame, width, height):
    return cv2.cuda.resize(gpu_frame, (width, height))


def cvt_color_gpu(gpu_frame, code):
    return cv2.cuda.cvtColor(gpu_frame, code)


def create_gpu_video_reader(video_path):
    ctx = GPUContext()
    if not ctx.is_available():
        return None

    try:
        if not hasattr(cv2, "cudacodec"):
            return None
        return cv2.cudacodec.createVideoReader(video_path)
    except Exception:
        return None


def get_processing_mode():
    ctx = GPUContext()
    return "GPU" if ctx.is_available() else "CPU"
