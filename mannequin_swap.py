#!/usr/bin/env python3
"""Locked-scene mannequin garment replacement app and headless runner."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import os
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime.
    np = None

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageOps
except ImportError as exc:  # pragma: no cover - hard dependency.
    raise RuntimeError("Pillow is required. Install it with `pip install pillow`.") from exc


MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "high")
INPUT_FIDELITY = os.environ.get("OPENAI_IMAGE_INPUT_FIDELITY", "high")
OUTPUT_FORMAT = "png"
MAX_ATTEMPTS = 3

STATUS_READY = "READY"
STATUS_GENERATING = "GENERATING"
STATUS_NEEDS_RETRY = "NEEDS RETRY"
STATUS_DONE = "DONE"
STATUS_ERROR = "ERROR"

RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS


FRONT_VIEW_PROMPT = """Use the base mannequin image as a locked scene reference.

Task:
Completely remove the original top worn by the mannequin and replace it with the garment from the front garment reference image.

Critical constraints:
- modify only the garment region
- preserve all non-garment parts exactly:
  mannequin, background, flowers, table, bag, butterfly, jewelry, shorts, lighting, shadows, framing, and environment
- do not blend the new garment with the old garment
- do not leave visible remnants of the original top
- do not reinterpret or regenerate the full scene
- treat the garment reference image only as a clothing-design reference, never as a scene reference
- the replacement garment must fit naturally on the mannequin with realistic folds, seams, tension, drape, perspective, and lighting
- output must look like a real ecommerce mannequin photo

Success condition:
The final image should look like the original mannequin photo, with only the top changed."""

FULL_TOP_REPLACEMENT_APPENDIX = """Replacement mode: full_top_replacement.

Critical replacement rules:
- remove all original garment pixels inside the garment region
- do not preserve original neckline shape
- do not preserve original strap structure
- reconstruct the garment fit from the garment reference only
- preserve the mannequin body, but not the original garment structure"""

CLOSE_UP_VIEW_PROMPT = """Use the accepted front-result image as the source image.

Create a close-up torso crop from the accepted front-result image.
Do not generate a new scene.
Do not change the garment design.
Do not change accessories, lighting, or background style.
Focus on upper torso detail only.
Keep the result fully consistent with the accepted front-result image.
This should feel like a tighter crop from the same photoshoot."""

SIDE_VIEW_PROMPT = """Use the accepted front-result image as the garment anchor and the locked base mannequin image as the scene-consistency anchor.

Create a realistic side or 3/4 side product-photo view of the same mannequin wearing the same accepted garment.

Critical constraints:
- preserve the same background, lighting, bag, butterfly, flowers, jewelry, shorts, and product-photo feel
- do not redesign the garment
- do not invent a new mannequin
- do not reinterpret the scene
- garment must wrap naturally around the mannequin
- output must feel like the same photoshoot from a side angle"""

BACK_VIEW_PROMPT = """Use the accepted front-result image as the mannequin and scene anchor.
Use the back garment reference image only as the back-garment design reference.

Create a realistic back product-photo view of the same mannequin wearing the same garment.
Preserve the same environment, lighting, realism, accessories where logically visible, and overall photoshoot consistency.
Do not invent a new scene.
Do not stylize.
Do not redesign the garment."""

VIEW_SPECS = {
    "front": ("Original view", "front_original_pov.png"),
    "closeup": ("Close-up torso", "close_up_pov.png"),
    "side": ("Side view", "side_pov.png"),
    "back": ("Back view", "back_pov.png"),
}


class MannequinSwapError(Exception):
    """Raised when the locked-scene pipeline cannot continue."""


@dataclass
class LoadedImage:
    label: str
    filename: str
    image: Image.Image
    png_bytes: bytes


@dataclass
class LoadedInputs:
    base: LoadedImage
    front_garment: LoadedImage
    back_garment: Optional[LoadedImage] = None


@dataclass
class MaskConfig:
    top: float = 0.18
    bottom: float = 0.66
    left: float = 0.29
    right: float = 0.71
    shoulder_flare: float = 0.08
    feather_radius: int = 6


@dataclass
class ValidationThresholds:
    outside_mean_max: float = 9.0
    outside_p95_max: float = 34.0
    border_mean_max: float = 12.0
    edge_mean_max: float = 20.0
    inside_mean_min: float = 14.0
    inside_change_ratio_min: float = 0.22


@dataclass
class ValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ViewOutput:
    key: str
    label: str
    filename: str
    status: str = STATUS_READY
    message: str = ""
    image_bytes: Optional[bytes] = None
    output_path: Optional[Path] = None
    prompt: str = ""
    validation: Optional[ValidationReport] = None


@dataclass
class PipelineConfig:
    output_dir: Path
    replacement_mode: str = "full_top_replacement"
    model: str = MODEL
    mask_config: MaskConfig = field(default_factory=MaskConfig)
    validation: ValidationThresholds = field(default_factory=ValidationThresholds)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGBA")
    return normalized


def normalize_canvas(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    if image.size == target_size:
        return image.convert("RGBA")
    return image.convert("RGBA").resize(target_size, RESAMPLING)


def make_view_output(key: str, status: str = STATUS_READY, message: str = "") -> ViewOutput:
    label, filename = VIEW_SPECS[key]
    return ViewOutput(key=key, label=label, filename=filename, status=status, message=message)


def default_results(include_back: bool) -> Dict[str, ViewOutput]:
    results = {
        "front": make_view_output("front"),
        "closeup": make_view_output("closeup"),
        "side": make_view_output("side"),
    }
    if include_back:
        results["back"] = make_view_output("back")
    return results


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise MannequinSwapError("OPENAI_API_KEY is not set.")
    return api_key


def build_client(api_key: Optional[str] = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MannequinSwapError(
            "The openai package is not installed. Install it with `pip install openai`."
        ) from exc
    return OpenAI(api_key=api_key or require_api_key())


def read_source_bytes(source: Any, label: str) -> Tuple[str, bytes]:
    if source is None:
        raise MannequinSwapError(f"Missing image source for {label}.")

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise MannequinSwapError(f"{label} image does not exist: {path}")
        try:
            return path.name, path.read_bytes()
        except OSError as exc:
            raise MannequinSwapError(f"Could not read {label} image: {path}") from exc

    if hasattr(source, "getvalue"):
        data = source.getvalue()
        filename = getattr(source, "name", f"{label}.png")
        return filename, bytes(data)

    if hasattr(source, "read"):
        data = source.read()
        filename = getattr(source, "name", f"{label}.png")
        return filename, bytes(data)

    if isinstance(source, (bytes, bytearray)):
        return f"{label}.png", bytes(source)

    raise MannequinSwapError(f"Unsupported image source type for {label}.")


def load_single_image(source: Any, label: str) -> LoadedImage:
    filename, raw_bytes = read_source_bytes(source, label)
    if not raw_bytes:
        raise MannequinSwapError(f"{label} image is empty.")
    if len(raw_bytes) > 50 * 1024 * 1024:
        raise MannequinSwapError(f"{label} image exceeds the 50MB API limit.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - image parsing errors vary.
        raise MannequinSwapError(f"{label} image could not be decoded.") from exc

    return LoadedImage(
        label=label,
        filename=filename,
        image=normalized,
        png_bytes=image_to_png_bytes(normalized),
    )


def load_images(base_source: Any, front_source: Any, back_source: Any = None) -> LoadedInputs:
    base = load_single_image(base_source, "base mannequin")
    front = load_single_image(front_source, "front garment")
    back = load_single_image(back_source, "back garment") if back_source is not None else None
    return LoadedInputs(base=base, front_garment=front, back_garment=back)


def ensure_output_dir(path_value: Any) -> Path:
    output_dir = Path(path_value).expanduser()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MannequinSwapError(f"Could not create output directory: {output_dir}") from exc
    return output_dir.resolve()


def build_prompt(prompt_body: str, replacement_mode: str, image_roles: Sequence[str]) -> str:
    sections = list(image_roles)
    sections.append("")
    sections.append(prompt_body.strip())
    if replacement_mode == "full_top_replacement":
        sections.append("")
        sections.append(FULL_TOP_REPLACEMENT_APPENDIX)
    return "\n".join(section for section in sections if section is not None)


def build_garment_mask(
    base_image: Image.Image,
    replacement_mode: str,
    config: MaskConfig,
) -> Image.Image:
    width, height = base_image.size
    top = config.top
    bottom = config.bottom
    left = config.left
    right = config.right
    shoulder_flare = config.shoulder_flare

    if replacement_mode == "full_top_replacement":
        top = max(0.05, top - 0.06)
        bottom = min(0.90, bottom + 0.04)
        left = max(0.08, left - 0.05)
        right = min(0.92, right + 0.05)
        shoulder_flare = min(0.22, shoulder_flare + 0.05)

    left_px = int(width * left)
    right_px = int(width * right)
    top_px = int(height * top)
    bottom_px = int(height * bottom)
    flare_px = int(width * shoulder_flare)
    radius = max(12, int(width * 0.05))

    mask_l = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(mask_l)

    torso_box = (
        left_px,
        min(height - 1, top_px + int(height * 0.05)),
        right_px,
        bottom_px,
    )
    draw.rounded_rectangle(torso_box, radius=radius, fill=0)

    shoulder_box = (
        max(0, left_px - flare_px),
        max(0, top_px),
        min(width, right_px + flare_px),
        min(height, top_px + int(height * 0.18)),
    )
    draw.ellipse(shoulder_box, fill=0)

    left_wing = (
        max(0, left_px - int(width * 0.10)),
        max(0, top_px + int(height * 0.08)),
        min(width, left_px + int(width * 0.15)),
        min(height, top_px + int(height * 0.36)),
    )
    right_wing = (
        max(0, right_px - int(width * 0.15)),
        max(0, top_px + int(height * 0.08)),
        min(width, right_px + int(width * 0.10)),
        min(height, top_px + int(height * 0.36)),
    )
    draw.ellipse(left_wing, fill=0)
    draw.ellipse(right_wing, fill=0)

    if config.feather_radius > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(config.feather_radius))

    mask_rgba = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    mask_rgba.putalpha(mask_l)
    return mask_rgba


def editable_bbox(mask_image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    alpha = mask_image.getchannel("A")
    editable = alpha.point(lambda value: 255 if value <= 12 else 0)
    return editable.getbbox()


def build_mask_preview(base_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    preview = base_image.convert("RGBA").copy()
    alpha = mask_image.getchannel("A")
    editable_alpha = ImageOps.invert(alpha).point(lambda value: min(190, value))
    overlay = Image.new("RGBA", base_image.size, (255, 70, 70, 0))
    overlay.putalpha(editable_alpha)
    return Image.alpha_composite(preview, overlay)


def ensure_numpy() -> None:
    if np is None:
        raise MannequinSwapError(
            "numpy is required for front-view validation. Install it with `pip install numpy`."
        )


def validate_front_view(
    base_image: Image.Image,
    generated_image: Image.Image,
    mask_image: Image.Image,
    thresholds: ValidationThresholds,
) -> ValidationReport:
    ensure_numpy()

    base_rgb = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    generated_rgb = np.asarray(generated_image.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask_image.getchannel("A"), dtype=np.uint8)

    outside_mask = alpha >= 245
    inside_mask = alpha <= 10
    if outside_mask.sum() == 0 or inside_mask.sum() == 0:
        raise MannequinSwapError("The garment mask is too small or malformed for validation.")

    diff = np.abs(base_rgb - generated_rgb).mean(axis=2)
    outside_values = diff[outside_mask]
    inside_values = diff[inside_mask]

    height, width = diff.shape
    border_px = max(8, int(min(width, height) * 0.08))
    border_mask = np.zeros_like(outside_mask, dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True
    border_values = diff[border_mask & outside_mask]

    base_gray = np.asarray(base_image.convert("L"), dtype=np.float32)
    generated_gray = np.asarray(generated_image.convert("L"), dtype=np.float32)
    base_edges = np.abs(np.diff(base_gray, axis=0, prepend=base_gray[:1, :]))
    base_edges += np.abs(np.diff(base_gray, axis=1, prepend=base_gray[:, :1]))
    generated_edges = np.abs(np.diff(generated_gray, axis=0, prepend=generated_gray[:1, :]))
    generated_edges += np.abs(np.diff(generated_gray, axis=1, prepend=generated_gray[:, :1]))
    edge_diff = np.abs(base_edges - generated_edges)
    edge_values = edge_diff[outside_mask]

    inside_change_ratio = float((inside_values > 18.0).mean())
    metrics = {
        "outside_mean_diff": float(outside_values.mean()),
        "outside_p95_diff": float(np.percentile(outside_values, 95)),
        "border_mean_diff": float(border_values.mean()) if border_values.size else 0.0,
        "edge_mean_diff": float(edge_values.mean()) if edge_values.size else 0.0,
        "inside_mean_diff": float(inside_values.mean()),
        "inside_change_ratio": inside_change_ratio,
    }

    reasons: List[str] = []
    if metrics["outside_mean_diff"] > thresholds.outside_mean_max:
        reasons.append("Non-garment regions drifted too much from the locked base scene.")
    if metrics["outside_p95_diff"] > thresholds.outside_p95_max:
        reasons.append("Some preserved props or background regions changed too aggressively.")
    if metrics["border_mean_diff"] > thresholds.border_mean_max:
        reasons.append("Image framing or edge lighting drifted outside the garment region.")
    if metrics["edge_mean_diff"] > thresholds.edge_mean_max:
        reasons.append("Scene edges, shadows, or silhouette details changed too much.")
    if metrics["inside_mean_diff"] < thresholds.inside_mean_min:
        reasons.append("The original top may still be visible because the garment change is too weak.")
    if metrics["inside_change_ratio"] < thresholds.inside_change_ratio_min:
        reasons.append("The garment replacement appears incomplete inside the editable region.")

    return ValidationReport(passed=not reasons, reasons=reasons, metrics=metrics)


def get_result_field(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def extract_image_bytes(result: Any) -> bytes:
    data = getattr(result, "data", None)
    if not data:
        raise MannequinSwapError("Image API response did not include any generated image data.")

    first_item = data[0]
    encoded = get_result_field(first_item, "b64_json")
    if not encoded:
        raise MannequinSwapError("Image API response is missing `b64_json` image output.")

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MannequinSwapError("Image API returned invalid base64 output.") from exc


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    error_name = exc.__class__.__name__.lower()
    return "timeout" in error_name or "connection" in error_name


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(1.0, float(retry_after))
            except (TypeError, ValueError):
                pass
    return min(8.0, float(2**attempt))


def call_image_edit(
    client: Any,
    config: PipelineConfig,
    image_payloads: Sequence[Tuple[str, bytes]],
    prompt: str,
    mask_image: Optional[Image.Image] = None,
) -> bytes:
    mask_bytes = image_to_png_bytes(mask_image) if mask_image is not None else None
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with tempfile.TemporaryDirectory(prefix="mannequin_swap_") as temp_dir, ExitStack() as stack:
                temp_path = Path(temp_dir)
                image_files = []
                for index, (filename, image_bytes) in enumerate(image_payloads, start=1):
                    suffix = Path(filename).suffix or ".png"
                    image_path = temp_path / f"image_{index}{suffix}"
                    image_path.write_bytes(image_bytes)
                    image_files.append(stack.enter_context(image_path.open("rb")))

                request: Dict[str, Any] = {
                    "model": config.model,
                    "image": image_files if len(image_files) > 1 else image_files[0],
                    "prompt": prompt,
                    "quality": QUALITY,
                    "input_fidelity": INPUT_FIDELITY,
                    "output_format": OUTPUT_FORMAT,
                    "size": "auto",
                    "n": 1,
                }

                if mask_bytes is not None:
                    mask_path = temp_path / "edit_mask.png"
                    mask_path.write_bytes(mask_bytes)
                    request["mask"] = stack.enter_context(mask_path.open("rb"))

                result = client.images.edit(**request)
            return extract_image_bytes(result)
        except Exception as exc:  # noqa: BLE001 - SDK exception types vary by version.
            last_error = exc
            if attempt >= MAX_ATTEMPTS or not is_retryable_error(exc):
                break
            time.sleep(retry_delay_seconds(exc, attempt))

    message = str(last_error) if last_error else "Unknown image edit failure."
    raise MannequinSwapError(f"Image edit request failed: {message}")


def generate_front_view(
    client: Any,
    inputs: LoadedInputs,
    config: PipelineConfig,
    mask_image: Image.Image,
) -> ViewOutput:
    result = make_view_output("front", status=STATUS_GENERATING)
    prompt = build_prompt(
        FRONT_VIEW_PROMPT,
        config.replacement_mode,
        image_roles=(
            "Image 1 is the locked base mannequin scene and edit target.",
            "Image 2 is the front garment reference image, and it is clothing only.",
        ),
    )
    result.prompt = prompt

    front_bytes = call_image_edit(
        client=client,
        config=config,
        image_payloads=[
            (inputs.base.filename, inputs.base.png_bytes),
            (inputs.front_garment.filename, inputs.front_garment.png_bytes),
        ],
        prompt=prompt,
        mask_image=mask_image,
    )
    front_image = normalize_canvas(bytes_to_image(front_bytes), inputs.base.image.size)
    normalized_bytes = image_to_png_bytes(front_image)
    validation = validate_front_view(
        base_image=inputs.base.image,
        generated_image=front_image,
        mask_image=mask_image,
        thresholds=config.validation,
    )

    result.image_bytes = normalized_bytes
    result.validation = validation
    if validation.passed:
        result.status = STATUS_DONE
        result.message = "Front/original view accepted and locked as the anchor result."
    else:
        result.status = STATUS_NEEDS_RETRY
        result.message = "Front/original view failed validation and needs retry."
    return result


def generate_closeup_view(front_result: ViewOutput, mask_image: Image.Image) -> ViewOutput:
    result = make_view_output("closeup", status=STATUS_GENERATING)
    result.prompt = CLOSE_UP_VIEW_PROMPT

    if not front_result.image_bytes:
        result.status = STATUS_ERROR
        result.message = "Accepted front result is missing."
        return result

    front_image = bytes_to_image(front_result.image_bytes)
    bbox = editable_bbox(mask_image)
    if bbox is None:
        result.status = STATUS_ERROR
        result.message = "Garment mask could not define a torso crop."
        return result

    width, height = front_image.size
    left, top, right, bottom = bbox
    pad_x = int(width * 0.08)
    pad_top = int(height * 0.06)
    garment_height = bottom - top
    crop_bottom = top + int(garment_height * 0.78)

    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_top),
        min(width, right + pad_x),
        min(height, crop_bottom + int(height * 0.06)),
    )
    crop = front_image.crop(crop_box)
    result.image_bytes = image_to_png_bytes(crop)
    result.status = STATUS_DONE
    result.message = "Close-up derived directly from the accepted front anchor."
    return result


def generate_side_view(
    client: Any,
    inputs: LoadedInputs,
    config: PipelineConfig,
    front_result: ViewOutput,
    mask_image: Image.Image,
) -> ViewOutput:
    result = make_view_output("side", status=STATUS_GENERATING)
    prompt = build_prompt(
        SIDE_VIEW_PROMPT,
        config.replacement_mode,
        image_roles=(
            "Image 1 is the locked base mannequin scene and scene-consistency anchor.",
            "Image 2 is the accepted front-result image and garment anchor.",
        ),
    )
    result.prompt = prompt

    if not front_result.image_bytes:
        result.status = STATUS_ERROR
        result.message = "Accepted front result is missing."
        return result

    side_bytes = call_image_edit(
        client=client,
        config=config,
        image_payloads=[
            (inputs.base.filename, inputs.base.png_bytes),
            ("accepted_front_anchor.png", front_result.image_bytes),
        ],
        prompt=prompt,
        mask_image=mask_image,
    )
    side_image = normalize_canvas(bytes_to_image(side_bytes), inputs.base.image.size)
    result.image_bytes = image_to_png_bytes(side_image)
    result.status = STATUS_DONE
    result.message = "Side view generated from the accepted front anchor."
    return result


def generate_back_view(
    client: Any,
    inputs: LoadedInputs,
    config: PipelineConfig,
    front_result: ViewOutput,
    mask_image: Image.Image,
) -> ViewOutput:
    result = make_view_output("back", status=STATUS_GENERATING)
    prompt = build_prompt(
        BACK_VIEW_PROMPT,
        config.replacement_mode,
        image_roles=(
            "Image 1 is the accepted front-result image and mannequin/scene anchor.",
            "Image 2 is the back garment reference image, and it is clothing only.",
        ),
    )
    result.prompt = prompt

    if inputs.back_garment is None:
        result.status = STATUS_READY
        result.message = "Optional back garment image not supplied."
        return result

    if not front_result.image_bytes:
        result.status = STATUS_ERROR
        result.message = "Accepted front result is missing."
        return result

    back_bytes = call_image_edit(
        client=client,
        config=config,
        image_payloads=[
            ("accepted_front_anchor.png", front_result.image_bytes),
            (inputs.back_garment.filename, inputs.back_garment.png_bytes),
        ],
        prompt=prompt,
        mask_image=mask_image,
    )
    back_image = normalize_canvas(bytes_to_image(back_bytes), inputs.base.image.size)
    result.image_bytes = image_to_png_bytes(back_image)
    result.status = STATUS_DONE
    result.message = "Back view generated from the accepted front anchor plus the back reference."
    return result


def save_outputs(output_dir: Path, results: Dict[str, ViewOutput]) -> Dict[str, ViewOutput]:
    output_dir = ensure_output_dir(output_dir)
    for view in results.values():
        if view.status == STATUS_DONE and view.image_bytes:
            path = output_dir / view.filename
            try:
                path.write_bytes(view.image_bytes)
            except OSError as exc:
                raise MannequinSwapError(f"Could not write output file: {path}") from exc
            view.output_path = path.resolve()
        else:
            view.output_path = None
    return results


def block_downstream_results(results: Dict[str, ViewOutput], include_back: bool, reason: str) -> None:
    results["closeup"] = make_view_output("closeup", status=STATUS_READY, message=reason)
    results["side"] = make_view_output("side", status=STATUS_READY, message=reason)
    if include_back:
        results["back"] = make_view_output("back", status=STATUS_READY, message=reason)


def run_locked_scene_workflow(
    client: Any,
    inputs: LoadedInputs,
    config: PipelineConfig,
) -> Tuple[Dict[str, ViewOutput], Image.Image]:
    mask_image = build_garment_mask(inputs.base.image, config.replacement_mode, config.mask_config)
    results = default_results(include_back=inputs.back_garment is not None)

    front_result = generate_front_view(client, inputs, config, mask_image)
    results["front"] = front_result

    if front_result.status != STATUS_DONE:
        block_downstream_results(
            results,
            include_back=inputs.back_garment is not None,
            reason="Blocked because the original/front view failed validation.",
        )
        save_outputs(config.output_dir, results)
        return results, mask_image

    results["closeup"] = generate_closeup_view(front_result, mask_image)
    results["side"] = generate_side_view(client, inputs, config, front_result, mask_image)
    if inputs.back_garment is not None:
        results["back"] = generate_back_view(client, inputs, config, front_result, mask_image)

    save_outputs(config.output_dir, results)
    return results, mask_image


def retry_view(
    view_key: str,
    client: Any,
    inputs: LoadedInputs,
    config: PipelineConfig,
    existing_results: Dict[str, ViewOutput],
) -> Tuple[Dict[str, ViewOutput], Image.Image]:
    mask_image = build_garment_mask(inputs.base.image, config.replacement_mode, config.mask_config)
    results = dict(existing_results)

    if view_key == "front":
        results["front"] = generate_front_view(client, inputs, config, mask_image)
        if results["front"].status == STATUS_DONE:
            results["closeup"] = make_view_output(
                "closeup",
                message="Front anchor updated. Redo this section to regenerate from the new anchor.",
            )
            results["side"] = make_view_output(
                "side",
                message="Front anchor updated. Redo this section to regenerate from the new anchor.",
            )
            if inputs.back_garment is not None:
                results["back"] = make_view_output(
                    "back",
                    message="Front anchor updated. Redo this section to regenerate from the new anchor.",
                )
        else:
            block_downstream_results(
                results,
                include_back=inputs.back_garment is not None,
                reason="Blocked because the original/front view failed validation.",
            )
    elif view_key == "closeup":
        front_result = results.get("front")
        if not front_result or front_result.status != STATUS_DONE:
            raise MannequinSwapError("Close-up is blocked until the front view is accepted.")
        results["closeup"] = generate_closeup_view(front_result, mask_image)
    elif view_key == "side":
        front_result = results.get("front")
        if not front_result or front_result.status != STATUS_DONE:
            raise MannequinSwapError("Side view is blocked until the front view is accepted.")
        results["side"] = generate_side_view(client, inputs, config, front_result, mask_image)
    elif view_key == "back":
        front_result = results.get("front")
        if inputs.back_garment is None:
            raise MannequinSwapError("Back view requires an optional back garment reference image.")
        if not front_result or front_result.status != STATUS_DONE:
            raise MannequinSwapError("Back view is blocked until the front view is accepted.")
        results["back"] = generate_back_view(client, inputs, config, front_result, mask_image)
    else:
        raise MannequinSwapError(f"Unknown view key: {view_key}")

    save_outputs(config.output_dir, results)
    return results, mask_image


def fingerprint_upload(uploaded_file: Any) -> Optional[str]:
    if uploaded_file is None:
        return None
    name = getattr(uploaded_file, "name", "upload")
    payload = uploaded_file.getvalue()
    return hashlib.sha1(name.encode("utf-8") + payload).hexdigest()


def is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def reset_uploader_key(state: Dict[str, Any], slot: str) -> None:
    state[f"{slot}_uploader_key"] = f"{slot}_uploader_{uuid.uuid4().hex}"


def reset_results_for_input_change(state: Dict[str, Any], slot: str) -> None:
    include_back = state.get("has_back_input", False)
    if slot in {"base", "front"}:
        state["results"] = default_results(include_back=include_back)
        state["mask_preview"] = None
    elif slot == "back":
        results = state.get("results", default_results(include_back=include_back))
        results.pop("back", None)
        if include_back:
            results["back"] = make_view_output("back")
        state["results"] = results


def sync_input_fingerprints(state: Dict[str, Any], base_file: Any, front_file: Any, back_file: Any) -> None:
    fingerprints = {
        "base": fingerprint_upload(base_file),
        "front": fingerprint_upload(front_file),
        "back": fingerprint_upload(back_file),
    }
    state["has_back_input"] = fingerprints["back"] is not None

    for slot, new_fingerprint in fingerprints.items():
        key = f"{slot}_fingerprint"
        old_fingerprint = state.get(key)
        if old_fingerprint != new_fingerprint:
            state[key] = new_fingerprint
            reset_results_for_input_change(state, slot)


def render_status_badge(st: Any, status: str) -> None:
    css_class = status.lower().replace(" ", "-").replace("/", "-").replace("_", "-")
    st.markdown(
        f"<span class='status-badge {css_class}'>{status}</span>",
        unsafe_allow_html=True,
    )


def image_bytes_for_display(view: ViewOutput) -> Optional[Image.Image]:
    if not view.image_bytes:
        return None
    return bytes_to_image(view.image_bytes)


def format_validation_metrics(report: ValidationReport) -> List[str]:
    return [
        f"outside_mean_diff: {report.metrics.get('outside_mean_diff', 0.0):.2f}",
        f"outside_p95_diff: {report.metrics.get('outside_p95_diff', 0.0):.2f}",
        f"border_mean_diff: {report.metrics.get('border_mean_diff', 0.0):.2f}",
        f"edge_mean_diff: {report.metrics.get('edge_mean_diff', 0.0):.2f}",
        f"inside_mean_diff: {report.metrics.get('inside_mean_diff', 0.0):.2f}",
        f"inside_change_ratio: {report.metrics.get('inside_change_ratio', 0.0):.2f}",
    ]


def render_result_card(
    st: Any,
    view: ViewOutput,
    can_retry: bool,
    retry_key: str,
    download_key: str,
) -> bool:
    retry_clicked = False
    with st.container():
        st.markdown(f"**{view.label}**")
        render_status_badge(st, view.status)
        if view.message:
            st.caption(view.message)

        preview = image_bytes_for_display(view)
        if preview is not None:
            st.image(preview, use_container_width=True)
        else:
            st.info("No preview yet.")

        if view.validation is not None:
            with st.expander("Validation details", expanded=view.status == STATUS_NEEDS_RETRY):
                for line in format_validation_metrics(view.validation):
                    st.text(line)
                for reason in view.validation.reasons:
                    st.warning(reason)
                if view.validation.passed:
                    st.success("Front view passed locked-scene validation.")

        if view.output_path is not None:
            st.caption(f"Saved to: {view.output_path}")

        if view.image_bytes:
            st.download_button(
                label=f"Download {view.filename}",
                data=view.image_bytes,
                file_name=view.filename,
                mime="image/png",
                key=download_key,
            )

        if can_retry:
            retry_clicked = st.button("Redo this section", key=retry_key, use_container_width=True)

    return retry_clicked


def request_rerun(st: Any) -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def run_cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked-scene mannequin garment replacement workflow headlessly. "
            "For the interactive app, use: streamlit run mannequin_swap.py"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", help="Path to the locked base mannequin image.")
    parser.add_argument(
        "--front-garment",
        "--clothing",
        dest="front_garment",
        help="Path to the front garment reference image.",
    )
    parser.add_argument("--back-garment", help="Optional path to the back garment reference image.")
    parser.add_argument("--outdir", default="output/locked_scene_swap", help="Output directory.")
    parser.add_argument(
        "--replacement-mode",
        default="full_top_replacement",
        choices=["full_top_replacement", "standard"],
        help="Garment replacement mode.",
    )

    if not argv:
        parser.print_help()
        print("\nInteractive app: streamlit run mannequin_swap.py")
        return 0

    args = parser.parse_args(list(argv))
    if not args.base or not args.front_garment:
        parser.error("--base and --front-garment/--clothing are required in CLI mode.")

    try:
        client = build_client()
        inputs = load_images(args.base, args.front_garment, args.back_garment)
        config = PipelineConfig(
            output_dir=ensure_output_dir(args.outdir),
            replacement_mode=args.replacement_mode,
        )
        results, _ = run_locked_scene_workflow(client, inputs, config)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except MannequinSwapError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    front_result = results["front"]
    if front_result.status != STATUS_DONE:
        print("Front/original view failed validation and needs retry.", file=sys.stderr)
        if front_result.validation:
            for reason in front_result.validation.reasons:
                print(f"- {reason}", file=sys.stderr)
        return 2

    print("Generated files:")
    for key in ("front", "closeup", "side", "back"):
        view = results.get(key)
        if view and view.output_path:
            print(str(view.output_path))
    return 0


def run_streamlit_app() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise MannequinSwapError(
            "Streamlit is required for the interactive app. Install it with `pip install streamlit`."
        ) from exc

    st.set_page_config(page_title="Locked Scene Garment Swap", layout="wide")
    st.markdown(
        """
        <style>
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.65rem;
        }
        .status-badge.ready { background: #efe7c7; color: #5c4600; }
        .status-badge.generating { background: #d8e6ff; color: #1447a6; }
        .status-badge.done { background: #d6f5df; color: #15643a; }
        .status-badge.error { background: #fde2e1; color: #9b1c1c; }
        .status-badge.needs-retry { background: #ffe8cc; color: #9a4b00; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    state = st.session_state
    if "results" not in state:
        state["results"] = default_results(include_back=False)
    if "mask_preview" not in state:
        state["mask_preview"] = None
    if "last_mask_config" not in state:
        state["last_mask_config"] = MaskConfig()
    if "base_uploader_key" not in state:
        reset_uploader_key(state, "base")
    if "front_uploader_key" not in state:
        reset_uploader_key(state, "front")
    if "back_uploader_key" not in state:
        reset_uploader_key(state, "back")
    if "output_dir" not in state:
        state["output_dir"] = str(Path("output") / "locked_scene_swap")

    st.title("Locked Scene Garment Replacement")
    st.caption(
        "Preserve scene, replace garment. The base mannequin image stays locked, "
        "the front/original view is validated first, and every dependent view anchors to that accepted front result."
    )

    with st.sidebar:
        st.markdown("**Workflow Settings**")
        state["output_dir"] = st.text_input("Output directory", value=state["output_dir"])
        replacement_mode = st.selectbox(
            "Replacement mode",
            options=["full_top_replacement", "standard"],
            index=0,
            help="Use full_top_replacement for difficult garments like lace, straps, tubes, or major neckline changes.",
        )
        st.caption(f"Model: `{MODEL}`")
        if os.environ.get("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY detected.")
        else:
            st.error("OPENAI_API_KEY is missing.")

        with st.expander("Validation thresholds", expanded=False):
            outside_mean_max = st.slider("Outside mean max", 2.0, 20.0, 9.0, 0.5)
            outside_p95_max = st.slider("Outside p95 max", 10.0, 60.0, 34.0, 1.0)
            border_mean_max = st.slider("Border mean max", 2.0, 30.0, 12.0, 0.5)
            edge_mean_max = st.slider("Edge mean max", 2.0, 40.0, 20.0, 0.5)
            inside_mean_min = st.slider("Inside mean min", 1.0, 40.0, 14.0, 0.5)
            inside_change_ratio_min = st.slider("Inside change ratio min", 0.05, 0.80, 0.22, 0.01)

        validation = ValidationThresholds(
            outside_mean_max=outside_mean_max,
            outside_p95_max=outside_p95_max,
            border_mean_max=border_mean_max,
            edge_mean_max=edge_mean_max,
            inside_mean_min=inside_mean_min,
            inside_change_ratio_min=inside_change_ratio_min,
        )

    step1, step2 = st.columns(2)
    with step1:
        st.markdown("**STEP 1**")
        st.markdown("Base mannequin image")
        base_file = st.file_uploader(
            "Locked base mannequin image",
            type=["png", "jpg", "jpeg", "webp"],
            key=state["base_uploader_key"],
        )
        render_status_badge(st, STATUS_READY if base_file is None else STATUS_DONE)
        st.caption("Locked reference status")
        if st.button("Remove image", key="remove_base"):
            reset_uploader_key(state, "base")
            state["base_fingerprint"] = None
            reset_results_for_input_change(state, "base")
            request_rerun(st)
        if base_file is not None:
            st.caption("Replace image by uploading a new file.")
            st.image(base_file, use_container_width=True)

    with step2:
        st.markdown("**STEP 2**")
        st.markdown("Front garment image")
        front_file = st.file_uploader(
            "Front garment reference image",
            type=["png", "jpg", "jpeg", "webp"],
            key=state["front_uploader_key"],
        )
        render_status_badge(st, STATUS_READY if front_file is None else STATUS_DONE)
        st.caption("Ready for generation status")
        if st.button("Remove image", key="remove_front"):
            reset_uploader_key(state, "front")
            state["front_fingerprint"] = None
            reset_results_for_input_change(state, "front")
            request_rerun(st)
        if front_file is not None:
            st.caption("Replace image by uploading a new file.")
            st.image(front_file, use_container_width=True)

    st.markdown("**STEP 3**")
    st.markdown("Back garment image (optional)")
    back_cols = st.columns((2, 1))
    with back_cols[0]:
        back_file = st.file_uploader(
            "Back garment reference image",
            type=["png", "jpg", "jpeg", "webp"],
            key=state["back_uploader_key"],
        )
        if back_file is not None:
            st.caption("Used only for back output.")
            st.image(back_file, use_container_width=True)
        else:
            st.caption("Optional. Used only for back output.")
    with back_cols[1]:
        render_status_badge(st, STATUS_READY if back_file is None else STATUS_DONE)
        if st.button("Remove image", key="remove_back"):
            reset_uploader_key(state, "back")
            state["back_fingerprint"] = None
            reset_results_for_input_change(state, "back")
            request_rerun(st)

    sync_input_fingerprints(state, base_file, front_file, back_file)

    mask_config = MaskConfig(
        top=st.slider("Mask top", 0.05, 0.40, state["last_mask_config"].top, 0.01),
        bottom=st.slider("Mask bottom", 0.45, 0.90, state["last_mask_config"].bottom, 0.01),
        left=st.slider("Mask left", 0.10, 0.45, state["last_mask_config"].left, 0.01),
        right=st.slider("Mask right", 0.55, 0.90, state["last_mask_config"].right, 0.01),
        shoulder_flare=st.slider(
            "Shoulder flare",
            0.00,
            0.20,
            state["last_mask_config"].shoulder_flare,
            0.01,
        ),
        feather_radius=st.slider("Mask feather", 0, 18, state["last_mask_config"].feather_radius, 1),
    )
    state["last_mask_config"] = mask_config

    loaded_inputs: Optional[LoadedInputs] = None
    mask_image: Optional[Image.Image] = None
    mask_preview: Optional[Image.Image] = None
    load_error: Optional[str] = None

    if base_file is not None and front_file is not None:
        try:
            loaded_inputs = load_images(base_file, front_file, back_file)
            mask_image = build_garment_mask(loaded_inputs.base.image, replacement_mode, mask_config)
            mask_preview = build_mask_preview(loaded_inputs.base.image, mask_image)
            state["mask_preview"] = image_to_png_bytes(mask_preview)
        except MannequinSwapError as exc:
            load_error = str(exc)

    if load_error:
        st.error(load_error)

    if loaded_inputs is not None and mask_preview is not None:
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.markdown("**Locked base preview**")
            st.image(loaded_inputs.base.image, use_container_width=True)
        with preview_cols[1]:
            st.markdown("**Editable garment region mask**")
            st.image(mask_preview, use_container_width=True)

    controls = st.columns((1.2, 1.2, 1))
    run_pipeline = controls[0].button(
        "Run locked-scene workflow",
        use_container_width=True,
        disabled=loaded_inputs is None or not os.environ.get("OPENAI_API_KEY"),
    )
    rerun_remaining = controls[1].button(
        "Generate remaining views from accepted front",
        use_container_width=True,
        disabled=state["results"].get("front", make_view_output("front")).status != STATUS_DONE,
    )
    controls[2].caption("Front must pass validation before dependent views can run.")

    if run_pipeline and loaded_inputs is not None:
        try:
            with st.spinner("Generating original/front view, validating it, and building dependent views..."):
                client = build_client()
                config = PipelineConfig(
                    output_dir=ensure_output_dir(state["output_dir"]),
                    replacement_mode=replacement_mode,
                    mask_config=mask_config,
                    validation=validation,
                )
                results, mask_image = run_locked_scene_workflow(client, loaded_inputs, config)
                state["results"] = results
                if mask_image is not None:
                    state["mask_preview"] = image_to_png_bytes(
                        build_mask_preview(loaded_inputs.base.image, mask_image)
                    )
        except MannequinSwapError as exc:
            st.error(str(exc))

    if rerun_remaining and loaded_inputs is not None:
        try:
            with st.spinner("Generating remaining views from the accepted front anchor..."):
                client = build_client()
                config = PipelineConfig(
                    output_dir=ensure_output_dir(state["output_dir"]),
                    replacement_mode=replacement_mode,
                    mask_config=mask_config,
                    validation=validation,
                )
                results = dict(state["results"])
                mask_image = build_garment_mask(loaded_inputs.base.image, replacement_mode, mask_config)
                front_result = results.get("front")
                if not front_result or front_result.status != STATUS_DONE:
                    raise MannequinSwapError("Front view must be accepted before dependent views can run.")
                results["closeup"] = generate_closeup_view(front_result, mask_image)
                results["side"] = generate_side_view(client, loaded_inputs, config, front_result, mask_image)
                if loaded_inputs.back_garment is not None:
                    results["back"] = generate_back_view(
                        client,
                        loaded_inputs,
                        config,
                        front_result,
                        mask_image,
                    )
                save_outputs(config.output_dir, results)
                state["results"] = results
        except MannequinSwapError as exc:
            st.error(str(exc))

    st.markdown("**RESULTS**")
    current_results = state["results"]
    view_order = ["front", "closeup", "side"]
    if loaded_inputs is not None and loaded_inputs.back_garment is not None:
        view_order.append("back")

    result_columns = st.columns(2)
    pending_retries: List[str] = []

    for index, key in enumerate(view_order):
        view = current_results.get(key, make_view_output(key))
        can_retry = loaded_inputs is not None
        if key != "front":
            can_retry = can_retry and current_results.get("front", make_view_output("front")).status == STATUS_DONE
        if key == "back":
            can_retry = can_retry and loaded_inputs is not None and loaded_inputs.back_garment is not None

        with result_columns[index % 2]:
            retry_clicked = render_result_card(
                st=st,
                view=view,
                can_retry=can_retry,
                retry_key=f"retry_{key}",
                download_key=f"download_{key}",
            )
            if retry_clicked:
                pending_retries.append(key)

    for key in pending_retries:
        if loaded_inputs is None:
            st.error("Please load the required images before retrying.")
            continue
        try:
            with st.spinner(f"Redoing {VIEW_SPECS[key][0].lower()}..."):
                client = build_client()
                config = PipelineConfig(
                    output_dir=ensure_output_dir(state["output_dir"]),
                    replacement_mode=replacement_mode,
                    mask_config=mask_config,
                    validation=validation,
                )
                updated_results, mask_image = retry_view(
                    view_key=key,
                    client=client,
                    inputs=loaded_inputs,
                    config=config,
                    existing_results=current_results,
                )
                state["results"] = updated_results
                if mask_image is not None:
                    state["mask_preview"] = image_to_png_bytes(
                        build_mask_preview(loaded_inputs.base.image, mask_image)
                    )
                request_rerun(st)
        except MannequinSwapError as exc:
            st.error(str(exc))


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        if is_running_in_streamlit():
            run_streamlit_app()
            return 0
        return run_cli(list(argv or sys.argv[1:]))
    except MannequinSwapError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
