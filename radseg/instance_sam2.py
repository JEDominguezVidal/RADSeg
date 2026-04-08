from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SAM2DependencyError(RuntimeError):
    """Raised when the optional SAM2 dependency is not available."""


_SAM2_DEPENDENCY_NAMES = {
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "iopath": "iopath",
    "portalocker": "portalocker",
    "antlr4": "antlr4-python3-runtime",
}


def _missing_package_error() -> SAM2DependencyError:
    return SAM2DependencyError(
        "Could not import the official SAM2 Python package. RADSeg's local "
        "`sam2/` directory only contains configs; install the official SAM2 "
        "package in this environment, for example with "
        "`pip install --no-build-isolation -e /path/to/Grounded-SAM-2`."
    )


def _missing_dependency_error(module_name: str) -> SAM2DependencyError:
    package_name = _SAM2_DEPENDENCY_NAMES.get(module_name, module_name)
    return SAM2DependencyError(
        f"SAM2 is installed but the runtime dependency `{package_name}` is missing. "
        "Install the official SAM2 dependencies (Hydra/OmegaConf/iopath and their "
        "runtime requirements) in this environment, then retry."
    )


def _ensure_sam2_hydra_initialized() -> None:
    try:
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
    except ModuleNotFoundError as exc:
        raise _missing_dependency_error(exc.name or "hydra") from exc

    if GlobalHydra.instance().is_initialized():
        return

    try:
        initialize_config_module("sam2", version_base="1.2")
    except Exception as exc:  # pragma: no cover - depends on Hydra runtime state.
        raise SAM2DependencyError(
            "SAM2 is importable but Hydra could not initialize the `sam2` config "
            "package. This usually means the official SAM2 package is not installed "
            "correctly, or it is being shadowed by a local `sam2/` directory that "
            "only contains configs."
        ) from exc


def _normalize_sam2_config_name(sam2_config: str) -> str:
    config_path = Path(sam2_config)
    if not config_path.is_file():
        return sam2_config

    parts = config_path.parts
    if "configs" not in parts:
        return sam2_config

    config_index = parts.index("configs")
    return "/".join(parts[config_index:])


def _load_sam2_components():
    try:
        SAM2AutomaticMaskGenerator = importlib.import_module(
            "sam2.automatic_mask_generator"
        ).SAM2AutomaticMaskGenerator
        build_sam2 = importlib.import_module("sam2.build_sam").build_sam2
        SAM2ImagePredictor = importlib.import_module(
            "sam2.sam2_image_predictor"
        ).SAM2ImagePredictor
        amg_utils = importlib.import_module("sam2.utils.amg")
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name == "sam2" or missing_name.startswith("sam2."):
            raise _missing_package_error() from exc
        if missing_name in _SAM2_DEPENDENCY_NAMES:
            raise _missing_dependency_error(missing_name) from exc
        raise SAM2DependencyError(
            "SAM2 could not be imported completely. Check that the official SAM2 "
            "package and its Python dependencies are installed in this environment."
        ) from exc
    except ImportError as exc:
        raise SAM2DependencyError(
            "SAM2 could not be imported completely. Check that the official SAM2 "
            "package and its Python dependencies are installed in this environment."
        ) from exc

    _ensure_sam2_hydra_initialized()

    return {
        "SAM2AutomaticMaskGenerator": SAM2AutomaticMaskGenerator,
        "SAM2ImagePredictor": SAM2ImagePredictor,
        "build_sam2": build_sam2,
        "calculate_stability_score": amg_utils.calculate_stability_score,
        "rle_to_mask": amg_utils.rle_to_mask,
    }


@dataclass(slots=True)
class SAM2PromptSpec:
    proposal_class_index: int
    proposal_class_name: str
    seed_point: tuple[int, int]
    point_coords: list[tuple[int, int]]
    point_labels: list[int]
    box_xyxy: tuple[int, int, int, int]
    mask_input: np.ndarray
    source: str = "radio-prompts"
    component_id: int | None = None
    seed_rank: int = 0


@dataclass(slots=True)
class SAM2CandidatePrediction:
    mask: np.ndarray
    mask_logits: np.ndarray | None
    low_res_logits: np.ndarray | None
    predicted_iou: float
    stability_score: float
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    source: str
    seed_point: tuple[int, int] | None
    proposal_class_index: int | None
    proposal_class_name: str | None
    point_coords: list[tuple[int, int]]
    point_labels: list[int]
    crop_box_xyxy: tuple[int, int, int, int] | None = None


def _mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class SAM2InstanceHelper:
    def __init__(
        self,
        sam2_config: str,
        sam2_ckpt: str,
        device: str,
        build_amg: bool = False,
        amg_points_per_side: int = 16,
        amg_crop_n_layers: int = 0,
        amg_points_per_batch: int = 64,
        amg_pred_iou_thresh: float = 0.7,
        amg_stability_score_thresh: float = 0.9,
        amg_box_nms_thresh: float = 0.7,
        amg_crop_nms_thresh: float = 0.7,
        amg_min_mask_region_area: int = 50,
    ):
        components = _load_sam2_components()
        checkpoint_path = Path(sam2_ckpt)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {sam2_ckpt}. "
                "Provide --sam2-ckpt with a valid checkpoint path."
            )
        resolved_config_name = _normalize_sam2_config_name(sam2_config)

        build_sam2 = components["build_sam2"]
        SAM2ImagePredictor = components["SAM2ImagePredictor"]
        SAM2AutomaticMaskGenerator = components["SAM2AutomaticMaskGenerator"]

        self._calculate_stability_score = components["calculate_stability_score"]
        self._rle_to_mask = components["rle_to_mask"]

        try:
            self.model = build_sam2(
                config_file=resolved_config_name,
                ckpt_path=str(checkpoint_path),
                device=device,
            )
        except Exception as exc:
            if type(exc).__name__ == "MissingConfigException":
                raise RuntimeError(
                    f"SAM2 config could not be resolved: {sam2_config}. Pass a Hydra "
                    "config name like `configs/sam2.1/sam2.1_hiera_s.yaml`, or a file "
                    "path located under a `sam2/configs/...` directory."
                ) from exc
            raise RuntimeError(
                "Failed to build the SAM2 model. Check that --sam2-config and "
                "--sam2-ckpt match the same SAM2 variant and that the checkpoint "
                "is not corrupted."
            ) from exc

        self.predictor = SAM2ImagePredictor(self.model)
        self.amg = None
        if build_amg:
            self.amg = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=amg_points_per_side,
                points_per_batch=amg_points_per_batch,
                pred_iou_thresh=amg_pred_iou_thresh,
                stability_score_thresh=amg_stability_score_thresh,
                box_nms_thresh=amg_box_nms_thresh,
                crop_n_layers=amg_crop_n_layers,
                crop_nms_thresh=amg_crop_nms_thresh,
                min_mask_region_area=amg_min_mask_region_area,
                output_mode="binary_mask",
                multimask_output=True,
            )

        self.device = device
        self.mask_input_size = tuple(self.model.sam_prompt_encoder.mask_input_size)
        self.image_rgb: np.ndarray | None = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("SAM2 expects image_rgb in HxWx3 RGB format.")
        image_rgb = np.asarray(image_rgb)
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        self.image_rgb = image_rgb
        self.predictor.set_image(image_rgb)

    def generate_from_prompts(
        self,
        prompt_specs: list[SAM2PromptSpec],
        multimask_output: bool = True,
    ) -> list[SAM2CandidatePrediction]:
        if not prompt_specs:
            return []
        if self.image_rgb is None:
            raise RuntimeError("Call set_image() before generating SAM2 candidates.")

        batch_size = len(prompt_specs)
        max_points = max(len(spec.point_coords) for spec in prompt_specs)

        point_coords = np.zeros((batch_size, max_points, 2), dtype=np.float32)
        point_labels = np.full((batch_size, max_points), -1, dtype=np.int32)
        boxes = np.zeros((batch_size, 4), dtype=np.float32)
        mask_inputs = np.zeros(
            (batch_size, 1, self.mask_input_size[0], self.mask_input_size[1]),
            dtype=np.float32,
        )

        for index, spec in enumerate(prompt_specs):
            num_points = len(spec.point_coords)
            point_coords[index, :num_points] = np.asarray(spec.point_coords, dtype=np.float32)
            point_labels[index, :num_points] = np.asarray(spec.point_labels, dtype=np.int32)
            boxes[index] = np.asarray(spec.box_xyxy, dtype=np.float32)
            if spec.mask_input.shape != mask_inputs[index].shape:
                raise ValueError(
                    "Prompt mask_input shape does not match SAM2 prompt encoder size: "
                    f"expected {mask_inputs[index].shape}, got {spec.mask_input.shape}."
                )
            mask_inputs[index] = spec.mask_input.astype(np.float32, copy=False)

        masks, predicted_ious, low_res_logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
            mask_input=mask_inputs,
            multimask_output=multimask_output,
            return_logits=True,
            normalize_coords=False,
        )

        masks = np.asarray(masks, dtype=np.float32)
        predicted_ious = np.asarray(predicted_ious, dtype=np.float32)
        low_res_logits = np.asarray(low_res_logits, dtype=np.float32)

        if masks.ndim == 3:
            masks = masks[None, ...]
        if predicted_ious.ndim == 1:
            predicted_ious = predicted_ious[None, ...]
        if low_res_logits.ndim == 3:
            low_res_logits = low_res_logits[None, ...]

        stability_scores = self._calculate_stability_score(
            torch.from_numpy(masks.reshape(-1, masks.shape[-2], masks.shape[-1])),
            self.predictor.mask_threshold,
            1.0,
        )
        stability_scores = (
            stability_scores.detach()
            .cpu()
            .numpy()
            .reshape(masks.shape[0], masks.shape[1])
            .astype(np.float32)
        )

        predictions: list[SAM2CandidatePrediction] = []
        for batch_index, spec in enumerate(prompt_specs):
            for mask_index in range(masks.shape[1]):
                mask_logits = masks[batch_index, mask_index]
                binary_mask = mask_logits > self.predictor.mask_threshold
                predictions.append(
                    SAM2CandidatePrediction(
                        mask=binary_mask.astype(bool, copy=False),
                        mask_logits=mask_logits.astype(np.float32, copy=False),
                        low_res_logits=low_res_logits[batch_index, mask_index].astype(
                            np.float32,
                            copy=False,
                        ),
                        predicted_iou=float(predicted_ious[batch_index, mask_index]),
                        stability_score=float(stability_scores[batch_index, mask_index]),
                        bbox_xyxy=_mask_to_box(binary_mask),
                        area=int(binary_mask.sum()),
                        source=spec.source,
                        seed_point=tuple(int(v) for v in spec.seed_point),
                        proposal_class_index=int(spec.proposal_class_index),
                        proposal_class_name=spec.proposal_class_name,
                        point_coords=[tuple(int(v) for v in point) for point in spec.point_coords],
                        point_labels=[int(v) for v in spec.point_labels],
                    )
                )

        return predictions

    def generate_amg_candidates(self) -> list[SAM2CandidatePrediction]:
        if self.amg is None:
            return []
        if self.image_rgb is None:
            raise RuntimeError("Call set_image() before generating SAM2 AMG candidates.")

        mask_data = self.amg._generate_masks(self.image_rgb)
        candidates: list[SAM2CandidatePrediction] = []
        for index, rle in enumerate(mask_data["rles"]):
            mask = self._rle_to_mask(rle).astype(bool)
            point_coords = mask_data["points"][index].tolist()
            crop_box = mask_data["crop_boxes"][index].tolist()
            bbox = mask_data["boxes"][index].tolist()
            candidates.append(
                SAM2CandidatePrediction(
                    mask=mask,
                    mask_logits=None,
                    low_res_logits=None,
                    predicted_iou=float(mask_data["iou_preds"][index]),
                    stability_score=float(mask_data["stability_score"][index]),
                    bbox_xyxy=(
                        int(round(bbox[0])),
                        int(round(bbox[1])),
                        int(round(bbox[2])),
                        int(round(bbox[3])),
                    ),
                    area=int(mask.sum()),
                    source="sam2-amg",
                    seed_point=(
                        int(round(point_coords[0])),
                        int(round(point_coords[1])),
                    ),
                    proposal_class_index=None,
                    proposal_class_name=None,
                    point_coords=[
                        (
                            int(round(point_coords[0])),
                            int(round(point_coords[1])),
                        )
                    ],
                    point_labels=[1],
                    crop_box_xyxy=(
                        int(round(crop_box[0])),
                        int(round(crop_box[1])),
                        int(round(crop_box[2])),
                        int(round(crop_box[3])),
                    ),
                )
            )
        return candidates

    @staticmethod
    def candidate_to_metadata(candidate: SAM2CandidatePrediction) -> dict[str, Any]:
        return {
            "source": candidate.source,
            "seed_point": candidate.seed_point,
            "predicted_iou": candidate.predicted_iou,
            "stability_score": candidate.stability_score,
            "bbox_xyxy": list(candidate.bbox_xyxy),
            "area": candidate.area,
            "proposal_class_index": candidate.proposal_class_index,
            "proposal_class_name": candidate.proposal_class_name,
            "point_coords": [list(point) for point in candidate.point_coords],
            "point_labels": list(candidate.point_labels),
            "crop_box_xyxy": (
                list(candidate.crop_box_xyxy)
                if candidate.crop_box_xyxy is not None
                else None
            ),
        }
