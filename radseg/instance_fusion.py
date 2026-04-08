from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from skimage.feature import peak_local_max

from radseg.instance_sam2 import SAM2CandidatePrediction, SAM2PromptSpec


DEFAULT_SCORE_WEIGHTS = {
    "radio_mean": 0.6,
    "class_margin": 0.2,
    "sam2_iou": 0.1,
    "sam2_stability": 0.1,
}


@dataclass(slots=True)
class ScoredInstanceCandidate:
    class_index: int
    class_name: str
    score: float
    radio_mean: float
    class_margin: float
    sam2_predicted_iou: float
    sam2_stability_score: float
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    source: str
    seed_point: tuple[int, int] | None
    proposal_class_index: int | None
    proposal_class_name: str | None
    point_coords: list[tuple[int, int]]
    point_labels: list[int]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _resize_mask_prompt(mask: np.ndarray, mask_input_size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(
        tensor,
        size=mask_input_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).numpy()


def _build_seed_mask_prompt(
    class_scores: np.ndarray,
    component_mask: np.ndarray,
    seed_point: tuple[int, int],
    mask_input_size: tuple[int, int],
) -> np.ndarray:
    mask = component_mask.astype(np.float32, copy=False)
    if mask.sum() == 0:
        return np.zeros((1, mask_input_size[0], mask_input_size[1]), dtype=np.float32)

    bbox = _bbox_from_mask(component_mask)
    width = max(1, bbox[2] - bbox[0] + 1)
    height = max(1, bbox[3] - bbox[1] + 1)
    sigma = max(4.0, 0.25 * max(width, height))

    yy, xx = np.indices(component_mask.shape, dtype=np.float32)
    seed_x = float(seed_point[0])
    seed_y = float(seed_point[1])
    gaussian = np.exp(-((xx - seed_x) ** 2 + (yy - seed_y) ** 2) / (2.0 * sigma**2))

    prompt_prob = class_scores.astype(np.float32, copy=False) * mask * gaussian
    max_value = float(prompt_prob.max())
    if max_value > 0.0:
        prompt_prob = prompt_prob / max_value

    prompt_prob = _resize_mask_prompt(prompt_prob, mask_input_size)
    prompt_logits = np.clip((prompt_prob * 2.0 - 1.0) * 8.0, -8.0, 8.0)
    return prompt_logits.astype(np.float32, copy=False)


def _component_peak_coords(
    class_scores: np.ndarray,
    component_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    seed_thresh: float,
    max_seeds_per_component: int,
) -> list[tuple[int, int]]:
    class_scores = class_scores.astype(np.float32, copy=False)
    width = max(1, bbox_xyxy[2] - bbox_xyxy[0] + 1)
    height = max(1, bbox_xyxy[3] - bbox_xyxy[1] + 1)
    min_distance = max(4, min(width, height) // 6)

    peaks = peak_local_max(
        class_scores,
        min_distance=min_distance,
        threshold_abs=seed_thresh,
        num_peaks=max_seeds_per_component,
        exclude_border=False,
        labels=component_mask.astype(np.uint8, copy=False),
    )
    coords = [(int(col), int(row)) for row, col in peaks.tolist()]
    if coords:
        return coords

    ys, xs = np.nonzero(component_mask)
    if ys.size == 0 or xs.size == 0:
        return []
    component_scores = class_scores[component_mask]
    best_index = int(component_scores.argmax())
    return [(int(xs[best_index]), int(ys[best_index]))]


def build_radio_prompt_specs(
    seg_probs_np: np.ndarray,
    seg_pred_np: np.ndarray,
    class_index_to_name: dict[int, str],
    mask_input_size: tuple[int, int],
    seed_thresh: float,
    max_seeds_per_component: int,
    min_area: int,
) -> list[SAM2PromptSpec]:
    prompt_specs: list[SAM2PromptSpec] = []
    num_classes = int(seg_probs_np.shape[0])
    seg_probs_np = seg_probs_np.astype(np.float32, copy=False)

    for class_index in range(1, num_classes):
        class_scores = seg_probs_np[class_index]
        support_mask = np.logical_and(seg_pred_np == class_index, class_scores >= seed_thresh)
        labeled_mask = measure.label(support_mask.astype(np.uint8), connectivity=2)
        num_components = int(labeled_mask.max())

        for component_id in range(1, num_components + 1):
            component_mask = labeled_mask == component_id
            area = int(component_mask.sum())
            if area < min_area:
                continue

            bbox_xyxy = _bbox_from_mask(component_mask)
            seed_points = _component_peak_coords(
                class_scores,
                component_mask,
                bbox_xyxy,
                seed_thresh,
                max_seeds_per_component,
            )
            if not seed_points:
                continue

            for seed_rank, seed_point in enumerate(seed_points):
                negative_points = [
                    other_point
                    for other_point in seed_points
                    if other_point != seed_point
                ]
                negative_points.sort(
                    key=lambda point: (point[0] - seed_point[0]) ** 2
                    + (point[1] - seed_point[1]) ** 2
                )
                negative_points = negative_points[:2]

                point_coords = [seed_point] + negative_points
                point_labels = [1] + [0] * len(negative_points)
                mask_input = _build_seed_mask_prompt(
                    class_scores,
                    component_mask,
                    seed_point,
                    mask_input_size,
                )

                prompt_specs.append(
                    SAM2PromptSpec(
                        proposal_class_index=class_index,
                        proposal_class_name=class_index_to_name[class_index],
                        seed_point=seed_point,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box_xyxy=bbox_xyxy,
                        mask_input=mask_input,
                        source="radio-prompts",
                        component_id=component_id,
                        seed_rank=seed_rank,
                    )
                )

    return prompt_specs


def score_instance_candidates(
    raw_candidates: list[SAM2CandidatePrediction],
    seg_probs_np: np.ndarray,
    class_index_to_name: dict[int, str],
    min_area: int,
    radio_mean_thresh: float = 0.35,
    class_margin_thresh: float = 0.08,
    score_weights: dict[str, float] | None = None,
) -> list[ScoredInstanceCandidate]:
    if score_weights is None:
        score_weights = DEFAULT_SCORE_WEIGHTS

    scored_candidates: list[ScoredInstanceCandidate] = []
    num_classes = int(seg_probs_np.shape[0])
    seg_probs_np = seg_probs_np.astype(np.float32, copy=False)

    for candidate in raw_candidates:
        mask = candidate.mask.astype(bool, copy=False)
        area = int(mask.sum())
        if area < min_area:
            continue

        class_means = seg_probs_np[:, mask].mean(axis=1)
        foreground_scores = class_means[1:num_classes]
        if foreground_scores.size == 0:
            continue

        order = np.argsort(foreground_scores)[::-1]
        top1_class_index = int(order[0] + 1)
        top1_score = float(class_means[top1_class_index])
        top2_score = float(class_means[int(order[1] + 1)]) if order.size > 1 else 0.0
        class_margin = top1_score - top2_score

        if candidate.proposal_class_index is None:
            class_index = top1_class_index
            radio_mean = top1_score
            class_name = class_index_to_name[class_index]
        else:
            class_index = int(candidate.proposal_class_index)
            if top1_class_index != class_index:
                continue
            radio_mean = float(class_means[class_index])
            class_name = candidate.proposal_class_name or class_index_to_name[class_index]

        if radio_mean < radio_mean_thresh or class_margin < class_margin_thresh:
            continue

        score = (
            score_weights["radio_mean"] * radio_mean
            + score_weights["class_margin"] * class_margin
            + score_weights["sam2_iou"] * float(candidate.predicted_iou)
            + score_weights["sam2_stability"] * float(candidate.stability_score)
        )

        scored_candidates.append(
            ScoredInstanceCandidate(
                class_index=class_index,
                class_name=class_name,
                score=float(score),
                radio_mean=float(radio_mean),
                class_margin=float(class_margin),
                sam2_predicted_iou=float(candidate.predicted_iou),
                sam2_stability_score=float(candidate.stability_score),
                area=area,
                bbox_xyxy=_bbox_from_mask(mask),
                mask=mask,
                source=candidate.source,
                seed_point=candidate.seed_point,
                proposal_class_index=candidate.proposal_class_index,
                proposal_class_name=candidate.proposal_class_name,
                point_coords=[tuple(point) for point in candidate.point_coords],
                point_labels=[int(value) for value in candidate.point_labels],
            )
        )

    return scored_candidates


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _mask_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    smaller_area = float(min(mask_a.sum(), mask_b.sum()))
    if smaller_area <= 0.0:
        return 0.0
    return intersection / smaller_area


def select_instances_with_nms(
    candidates: list[ScoredInstanceCandidate],
    nms_iou_thresh: float,
    containment_thresh: float = 0.9,
) -> list[ScoredInstanceCandidate]:
    selected: list[ScoredInstanceCandidate] = []
    for class_index in sorted({candidate.class_index for candidate in candidates}):
        class_candidates = [candidate for candidate in candidates if candidate.class_index == class_index]
        class_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        kept: list[ScoredInstanceCandidate] = []

        for candidate in class_candidates:
            should_suppress = False
            for existing in kept:
                if _mask_iou(candidate.mask, existing.mask) >= nms_iou_thresh:
                    should_suppress = True
                    break
                if _mask_containment(candidate.mask, existing.mask) >= containment_thresh:
                    should_suppress = True
                    break
            if not should_suppress:
                kept.append(candidate)

        selected.extend(kept)

    selected.sort(key=lambda candidate: candidate.score, reverse=True)
    return selected


def rasterize_instance_index(
    candidates: list[ScoredInstanceCandidate],
    image_shape: tuple[int, int],
) -> np.ndarray:
    instance_index = np.zeros(image_shape, dtype=np.uint16)
    for instance_id, candidate in enumerate(candidates, start=1):
        writable = np.logical_and(candidate.mask, instance_index == 0)
        instance_index[writable] = instance_id
    return instance_index


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict[str, list[int] | list[int]]:
    flat = np.asfortranarray(mask.astype(np.uint8)).reshape(-1, order="F")
    counts: list[int] = []
    last_value = 0
    run_length = 0
    for value in flat.tolist():
        if value == last_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            last_value = value
    counts.append(run_length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def serialize_instances(
    candidates: list[ScoredInstanceCandidate],
) -> list[dict[str, object]]:
    instances: list[dict[str, object]] = []
    for instance_id, candidate in enumerate(candidates, start=1):
        instances.append(
            {
                "instance_id": instance_id,
                "class_index": candidate.class_index,
                "class_name": candidate.class_name,
                "score": candidate.score,
                "radio_mean": candidate.radio_mean,
                "class_margin": candidate.class_margin,
                "sam2_predicted_iou": candidate.sam2_predicted_iou,
                "sam2_stability_score": candidate.sam2_stability_score,
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "area": candidate.area,
                "segmentation_rle": mask_to_uncompressed_rle(candidate.mask),
                "source": candidate.source,
                "seed_point": list(candidate.seed_point) if candidate.seed_point is not None else None,
            }
        )
    return instances
