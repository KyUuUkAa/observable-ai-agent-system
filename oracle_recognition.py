from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "oracle" / "best_portable.pt"
MAX_IMAGE_PIXELS = 25_000_000

_prediction_lock = Lock()


class OracleRecognitionError(RuntimeError):
    """Base error for Oracle Bone Script recognition."""


class OracleModelUnavailableError(OracleRecognitionError):
    """Raised when the local classifier cannot be loaded."""


class OracleImageError(OracleRecognitionError):
    """Raised when uploaded bytes are not a supported image."""


def get_model_path() -> Path:
    """Return the configured local model path without exposing it through the API."""

    configured = os.getenv("ORACLE_MODEL_PATH")
    path = Path(configured).expanduser() if configured else DEFAULT_MODEL_PATH
    path = path if path.is_absolute() else PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise OracleModelUnavailableError(
            "甲骨文分类模型未配置。请将 best_portable.pt 放到 "
            "models/oracle/，或设置 ORACLE_MODEL_PATH。"
        )
    return path


def get_device() -> str:
    """Return the inference device configured for the local classifier."""

    return os.getenv("ORACLE_DEVICE", "cpu")


def get_image_size() -> int:
    """Return and validate the square inference size."""

    try:
        image_size = int(os.getenv("ORACLE_IMGSZ", "224"))
    except ValueError as exc:
        raise OracleModelUnavailableError("ORACLE_IMGSZ 必须是整数。") from exc
    if image_size <= 0:
        raise OracleModelUnavailableError("ORACLE_IMGSZ 必须大于 0。")
    return image_size


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad a glyph to square with its median border color, preserving the full shape."""

    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image

    pixels = np.asarray(image)
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]),
        axis=0,
    )
    fill = tuple(int(value) for value in np.median(border, axis=0))
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def decode_image(content: bytes) -> Image.Image:
    """Decode validated upload bytes into an RGB image."""

    if not content:
        raise OracleImageError("上传文件为空。")

    try:
        with Image.open(io.BytesIO(content)) as uploaded:
            width, height = uploaded.size
            if width <= 0 or height <= 0:
                raise OracleImageError("图片尺寸无效。")
            if width * height > MAX_IMAGE_PIXELS:
                raise OracleImageError("图片像素过大，请上传不超过 2500 万像素的单字图片。")
            return uploaded.convert("RGB")
    except OracleImageError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise OracleImageError("上传内容不是可读取的图片。") from exc


@lru_cache(maxsize=1)
def get_model():
    """Load the portable YOLO classifier once, on the first recognition request."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise OracleModelUnavailableError(
            "未安装 ultralytics。请运行 python -m pip install -r requirements.txt。"
        ) from exc

    try:
        return YOLO(str(get_model_path()), task="classify")
    except Exception as exc:
        raise OracleModelUnavailableError("甲骨文分类模型加载失败。") from exc


def get_model_status() -> dict:
    """Return safe model configuration status without loading the checkpoint."""

    try:
        get_model_path()
        return {
            "status": "ready",
            "task": "single_glyph_classification",
            "device": get_device(),
            "image_size": get_image_size(),
        }
    except OracleModelUnavailableError as exc:
        return {
            "status": "unavailable",
            "detail": str(exc),
        }


def recognize_oracle_image(content: bytes) -> dict:
    """Classify one cropped Oracle Bone Script glyph and return Top-5 predictions."""

    image = decode_image(content)
    source = np.asarray(pad_to_square(image)).copy()

    try:
        with _prediction_lock:
            result = get_model().predict(
                source=source,
                imgsz=get_image_size(),
                device=get_device(),
                verbose=False,
            )[0]
    except OracleRecognitionError:
        raise
    except Exception as exc:
        raise OracleRecognitionError("甲骨文分类推理失败。") from exc

    if result.probs is None:
        raise OracleRecognitionError("模型没有返回分类概率。")

    indices = [int(index) for index in result.probs.top5]
    confidences = [float(value) for value in result.probs.top5conf]
    top5 = [
        {
            "class_id": index,
            "class_code": str(result.names[index]),
            "confidence": confidence,
        }
        for index, confidence in zip(indices, confidences)
    ]

    return {
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "prediction": top5[0],
        "top5": top5,
        "model": {
            "task": "single_glyph_classification",
            "class_count": len(result.names),
            "image_size": get_image_size(),
            "device": get_device(),
        },
    }
