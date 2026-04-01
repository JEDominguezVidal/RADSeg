import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

from radseg.radseg import RADSegEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal CLI demo for RADSeg on a single image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--classes",
        required=True,
        help="Comma-separated list of classes, for example: sky,road,car",
    )
    parser.add_argument(
        "--model-version",
        default="c-radio_v3-b",
        help="RADIO model version to load.",
    )
    parser.add_argument(
        "--lang-model",
        default="siglip2",
        help="Language adaptor to use.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/minimal_demo",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results with matplotlib after saving them.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Generate labeled mask outputs and region tables in segmentation mode.",
    )
    parser.add_argument(
        "--label-min-area",
        type=int,
        default=500,
        help="Minimum connected-region area in pixels required to draw a label.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Measure and report per-stage execution times.",
    )
    parser.add_argument(
        "--heatmaps",
        action="store_true",
        help="Generate per-class heatmaps instead of final segmentation outputs.",
    )
    parser.add_argument(
        "--sam-refinement",
        action="store_true",
        help="Enable SAM refinement for final segmentation mode.",
    )
    parser.add_argument(
        "--sam-ckpt",
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint used when --sam-refinement is enabled.",
    )
    parser.add_argument(
        "--prediction-thresh",
        type=float,
        default=0.0,
        help="Prediction threshold for final segmentation mode.",
    )
    parser.add_argument(
        "--slide-crop",
        type=int,
        default=336,
        help="Sliding window crop size. Set 0 to disable sliding window.",
    )
    parser.add_argument(
        "--slide-stride",
        type=int,
        default=224,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--scra-scaling",
        type=float,
        default=10.0,
        help="Self-Correlating Recursive Attention scaling.",
    )
    parser.add_argument(
        "--scga-scaling",
        type=float,
        default=10.0,
        help="Self-Correlating Global Aggregation scaling.",
    )
    return parser.parse_args()


def parse_classes(classes_arg: str) -> list[str]:
    classes = [item.strip() for item in classes_arg.split(",")]
    classes = [item for item in classes if item]
    if not classes:
        raise ValueError("No valid classes were provided after parsing --classes.")
    return classes


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path: Path, device: str) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = tensor.float() / 255.0
    return image, tensor


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "class"


def build_palette(num_colors: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(num_colors, 1))
    colors = []
    for idx in range(max(num_colors, 1)):
        rgb = cmap(idx)[:3]
        colors.append((np.array(rgb) * 255).astype(np.uint8))
    return np.stack(colors, axis=0)


def apply_heatmap(values: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val > min_val:
        values = (values - min_val) / (max_val - min_val)
    else:
        values = np.zeros_like(values)
    colored = plt.get_cmap(cmap_name)(values)[..., :3]
    return (colored * 255).astype(np.uint8)


def blend_overlay(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    blended = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * overlay_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_metadata(output_dir: Path, metadata: dict):
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def get_label_font(image_shape: tuple[int, int, int]) -> ImageFont.ImageFont:
    size = max(10, min(image_shape[0], image_shape[1]) // 35)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def get_center_location(mask: np.ndarray) -> tuple[int, int]:
    loc = np.argwhere(mask == 1)
    if loc.size == 0:
        return (0, 0)

    loc_sort = np.array(sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
    y_list = loc_sort[:, 0]
    unique, counts = np.unique(y_list, return_counts=True)
    y_loc = unique[counts.argmax()]
    y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
    center_num = len(y_most_freq_loc) // 2
    x = int(y_most_freq_loc[center_num][1])
    y = int(y_most_freq_loc[center_num][0])
    return (x, y)


def extract_labeled_regions(
    seg_pred_np: np.ndarray,
    class_index_to_name: dict[int, str],
    palette: np.ndarray,
    min_area: int,
) -> list[dict]:
    regions = []
    region_id = 1

    for class_index in sorted(np.unique(seg_pred_np).tolist()):
        if class_index == 0:
            continue

        class_mask = (seg_pred_np == class_index).astype(np.uint8)
        labeled_mask = measure.label(class_mask, connectivity=2)
        num_regions = int(labeled_mask.max())

        for region_label in range(1, num_regions + 1):
            region_mask = labeled_mask == region_label
            pixel_area = int(region_mask.sum())
            if pixel_area < min_area:
                continue

            ys, xs = np.nonzero(region_mask)
            center_x, center_y = get_center_location(region_mask.astype(np.uint8))
            color_rgb = [int(channel) for channel in palette[class_index].tolist()]

            regions.append(
                {
                    "region_id": region_id,
                    "class_index": int(class_index),
                    "class_name": class_index_to_name[int(class_index)],
                    "pixel_area": pixel_area,
                    "bbox_xmin": int(xs.min()),
                    "bbox_ymin": int(ys.min()),
                    "bbox_xmax": int(xs.max()),
                    "bbox_ymax": int(ys.max()),
                    "center_x": int(center_x),
                    "center_y": int(center_y),
                    "color_rgb": color_rgb,
                }
            )
            region_id += 1

    return regions


def draw_labeled_regions(image_rgb: np.ndarray, regions: list[dict]) -> np.ndarray:
    labeled_image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(labeled_image)
    font = get_label_font(image_rgb.shape)

    for region in regions:
        text = region["class_name"]
        center_x = region["center_x"]
        center_y = region["center_y"]
        color = tuple(region["color_rgb"])

        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding_x = 6
        padding_y = 4
        box_width = text_width + 2 * padding_x
        box_height = text_height + 2 * padding_y

        x0 = max(0, min(image_rgb.shape[1] - box_width, center_x - box_width // 2))
        y0 = max(0, min(image_rgb.shape[0] - box_height, center_y - box_height // 2))
        x1 = x0 + box_width
        y1 = y0 + box_height

        draw.rectangle((x0, y0, x1, y1), fill=color, outline=(0, 0, 0), width=1)
        draw.text(
            (x0 + padding_x, y0 + padding_y),
            text,
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    return np.array(labeled_image)


def save_regions_csv(output_dir: Path, regions: list[dict]):
    fieldnames = [
        "region_id",
        "class_index",
        "class_name",
        "pixel_area",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "center_x",
        "center_y",
        "color_rgb",
    ]
    with (output_dir / "regions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for region in regions:
            row = dict(region)
            row["color_rgb"] = ",".join(str(channel) for channel in row["color_rgb"])
            writer.writerow(row)


def save_regions_json(output_dir: Path, regions: list[dict]):
    (output_dir / "regions.json").write_text(json.dumps(regions, indent=2))


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fps_from_seconds(elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return float("inf")
    return 1.0 / elapsed_seconds


def measure_stage(stage_name: str, timings: dict | None, fn, *args, **kwargs):
    if timings is None:
        return fn(*args, **kwargs)
    sync_cuda()
    start = perf_counter()
    result = fn(*args, **kwargs)
    sync_cuda()
    timings[stage_name] = perf_counter() - start
    return result


def format_fps(elapsed_seconds: float) -> str:
    fps = fps_from_seconds(elapsed_seconds)
    return "inf" if fps == float("inf") else f"{fps:.2f}"


def print_timing_summary(timings: dict):
    print("\n" + "=" * 60)
    print("RADSEG TIMING SUMMARY")
    print("=" * 60)
    for stage_name, elapsed in timings.items():
        print(f"{stage_name:<28}: {elapsed:.4f} s ({format_fps(elapsed)} FPS)")
    print(
        "\nBenchmark note: 'total_execution' is the most useful number to compare end-to-end throughput between PCs."
    )
    total = timings.get("total_execution")
    if total is not None:
        print(f"Per-image latency (RADSeg): {total * 1000:.2f} ms")
        print(f"Equivalent throughput      : {format_fps(total)} FPS")
    print("=" * 60 + "\n")


def create_execution_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    execution_name = f"execution_{timestamp}"

    for index in range(100):
        suffix = "" if index == 0 else f"_{index:02d}"
        execution_dir = base_output_dir / f"{execution_name}{suffix}"
        try:
            execution_dir.mkdir(parents=True, exist_ok=False)
            return execution_dir
        except FileExistsError:
            continue

    raise RuntimeError(
        f"Could not create a unique execution directory under {base_output_dir}"
    )


@torch.inference_mode()
def run_mask_mode(args, classes: list[str], device: str, output_dir: Path):
    timings = {} if args.timings else None
    sync_cuda()
    total_start = perf_counter() if args.timings else None

    if args.sam_refinement and not Path(args.sam_ckpt).is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {args.sam_ckpt}. Disable --sam-refinement or provide a valid path."
        )

    image_pil, tensor_image = measure_stage(
        "load_image", timings, load_image, Path(args.image), device
    )
    image_np = np.array(image_pil)

    encoder = measure_stage(
        "create_encoder",
        timings,
        RADSegEncoder,
        device=device,
        model_version=args.model_version,
        lang_model=args.lang_model,
        predict=True,
        classes=classes,
        prediction_thresh=args.prediction_thresh,
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
        sam_refinement=args.sam_refinement,
        sam_ckpt=args.sam_ckpt,
    )

    seg_probs, seg_pred = measure_stage(
        "model_inference",
        timings,
        encoder.encode_image_to_feat_map,
        tensor_image,
        orig_img_size=image_np.shape[:2],
        return_preds=True,
    )

    def postprocess_mask():
        seg_probs_np = seg_probs.squeeze(0).detach().cpu().numpy()
        seg_pred_np = seg_pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        palette = build_palette(len(classes) + 1)
        color_mask = palette[seg_pred_np]
        overlay = blend_overlay(image_np, color_mask)
        class_index_to_name = {0: "__background__"} | {
            i + 1: class_name for i, class_name in enumerate(classes)
        }

        labeled_regions = []
        mask_color_labeled = None
        overlay_labeled = None
        if args.show_labels:
            labeled_regions = extract_labeled_regions(
                seg_pred_np,
                class_index_to_name,
                palette,
                min_area=args.label_min_area,
            )
            mask_color_labeled = draw_labeled_regions(color_mask, labeled_regions)
            overlay_labeled = draw_labeled_regions(overlay, labeled_regions)

        return (
            seg_probs_np,
            seg_pred_np,
            color_mask,
            overlay,
            class_index_to_name,
            labeled_regions,
            mask_color_labeled,
            overlay_labeled,
        )

    (
        seg_probs_np,
        seg_pred_np,
        color_mask,
        overlay,
        class_index_to_name,
        labeled_regions,
        mask_color_labeled,
        overlay_labeled,
    ) = measure_stage(
        "postprocess_mask", timings, postprocess_mask
    )

    outputs = [
        "input.png",
        "mask_index.png",
        "mask_color.png",
        "overlay.png",
        "seg_probs.npy",
    ]
    if args.show_labels:
        outputs.extend(
            [
                "mask_color_labeled.png",
                "overlay_labeled.png",
                "regions.csv",
                "regions.json",
            ]
        )

    metadata = {
        "mode": "mask",
        "image": str(Path(args.image)),
        "classes": classes,
        "class_indices": {str(index): name for index, name in class_index_to_name.items()},
        "model_version": args.model_version,
        "lang_model": args.lang_model,
        "device": device,
        "sam_refinement": args.sam_refinement,
        "sam_ckpt": args.sam_ckpt if args.sam_refinement else None,
        "show_labels": args.show_labels,
        "label_min_area": args.label_min_area if args.show_labels else None,
        "num_labeled_regions": len(labeled_regions),
        "prediction_thresh": args.prediction_thresh,
        "slide_crop": args.slide_crop,
        "slide_stride": args.slide_stride,
        "scra_scaling": args.scra_scaling,
        "scga_scaling": args.scga_scaling,
        "outputs": outputs,
    }
    if timings is not None:
        metadata["timings_seconds"] = dict(timings)

    def save_outputs():
        image_pil.save(output_dir / "input.png")
        Image.fromarray(seg_pred_np, mode="L").save(output_dir / "mask_index.png")
        Image.fromarray(color_mask).save(output_dir / "mask_color.png")
        Image.fromarray(overlay).save(output_dir / "overlay.png")
        if args.show_labels:
            Image.fromarray(mask_color_labeled).save(output_dir / "mask_color_labeled.png")
            Image.fromarray(overlay_labeled).save(output_dir / "overlay_labeled.png")
            save_regions_csv(output_dir, labeled_regions)
            save_regions_json(output_dir, labeled_regions)
        np.save(output_dir / "seg_probs.npy", seg_probs_np)
        if timings is not None:
            metadata["timings_fps"] = {
                stage_name: fps_from_seconds(elapsed)
                for stage_name, elapsed in metadata["timings_seconds"].items()
            }
        save_metadata(output_dir, metadata)

    measure_stage("save_outputs", timings, save_outputs)

    if timings is not None:
        sync_cuda()
        timings["total_execution"] = perf_counter() - total_start
        metadata["timings_seconds"] = dict(timings)
        metadata["timings_fps"] = {
            stage_name: fps_from_seconds(elapsed)
            for stage_name, elapsed in metadata["timings_seconds"].items()
        }
        save_metadata(output_dir, metadata)
        print_timing_summary(timings)

    if args.show:
        panels = [
            ("Input", image_np),
            ("Segmentation", color_mask),
            ("Overlay", overlay),
        ]
        if args.show_labels:
            panels.extend(
                [
                    ("Segmentation Labeled", mask_color_labeled),
                    ("Overlay Labeled", overlay_labeled),
                ]
            )

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        axes = np.atleast_1d(axes)
        for ax, (title, image) in zip(axes, panels):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


@torch.inference_mode()
def run_heatmap_mode(args, classes: list[str], device: str, output_dir: Path):
    timings = {} if args.timings else None
    sync_cuda()
    total_start = perf_counter() if args.timings else None

    if args.sam_refinement:
        raise ValueError("--sam-refinement is only supported in final segmentation mode, not with --heatmaps.")

    image_pil, tensor_image = measure_stage(
        "load_image", timings, load_image, Path(args.image), device
    )
    image_np = np.array(image_pil)

    encoder = measure_stage(
        "create_encoder",
        timings,
        RADSegEncoder,
        device=device,
        model_version=args.model_version,
        lang_model=args.lang_model,
        predict=False,
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
    )

    feat_map = measure_stage(
        "compute_feature_map",
        timings,
        encoder.encode_image_to_feat_map,
        tensor_image,
        orig_img_size=image_np.shape[:2],
    )
    aligned_feats = measure_stage(
        "align_features",
        timings,
        encoder.align_spatial_features_with_language,
        feat_map,
        onehot=False,
    )
    prompt_embeds = measure_stage(
        "encode_labels", timings, encoder.encode_labels, classes
    )

    def compute_similarity():
        _, channels, height, width = aligned_feats.shape
        aligned_feats_flat = aligned_feats.permute(0, 2, 3, 1).reshape(-1, channels)
        vec1 = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
        vec2 = aligned_feats_flat / aligned_feats_flat.norm(dim=-1, keepdim=True)
        sim = vec1 @ vec2.t()
        sim = sim.reshape(len(classes), height, width)
        sim = F.interpolate(
            sim.unsqueeze(0),
            size=image_np.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return sim

    sim = measure_stage("compute_similarity", timings, compute_similarity)

    def postprocess_heatmaps():
        output_items = ["input.png"]
        processed_items = []
        for index, class_name in enumerate(classes):
            heatmap = sim[index].detach().cpu().numpy()
            heatmap_rgb = apply_heatmap(heatmap)
            overlay = blend_overlay(image_np, heatmap_rgb)
            stem = f"{index + 1:02d}_{sanitize_name(class_name)}"
            output_items.extend(
                [
                    f"{stem}_heatmap.png",
                    f"{stem}_overlay.png",
                    f"{stem}_scores.npy",
                ]
            )
            processed_items.append((stem, heatmap, heatmap_rgb, overlay))
        return output_items, processed_items

    outputs, processed_heatmaps = measure_stage(
        "postprocess_heatmaps", timings, postprocess_heatmaps
    )

    metadata = {
        "mode": "heatmaps",
        "image": str(Path(args.image)),
        "classes": classes,
        "model_version": args.model_version,
        "lang_model": args.lang_model,
        "device": device,
        "slide_crop": args.slide_crop,
        "slide_stride": args.slide_stride,
        "scra_scaling": args.scra_scaling,
        "scga_scaling": args.scga_scaling,
        "outputs": outputs,
    }
    if timings is not None:
        metadata["timings_seconds"] = dict(timings)

    def save_outputs():
        image_pil.save(output_dir / "input.png")
        for stem, heatmap, heatmap_rgb, overlay in processed_heatmaps:
            Image.fromarray(heatmap_rgb).save(output_dir / f"{stem}_heatmap.png")
            Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
            np.save(output_dir / f"{stem}_scores.npy", heatmap)
        if timings is not None:
            metadata["timings_fps"] = {
                stage_name: fps_from_seconds(elapsed)
                for stage_name, elapsed in metadata["timings_seconds"].items()
            }
        save_metadata(output_dir, metadata)

    measure_stage("save_outputs", timings, save_outputs)

    if timings is not None:
        sync_cuda()
        timings["total_execution"] = perf_counter() - total_start
        metadata["timings_seconds"] = dict(timings)
        metadata["timings_fps"] = {
            stage_name: fps_from_seconds(elapsed)
            for stage_name, elapsed in metadata["timings_seconds"].items()
        }
        save_metadata(output_dir, metadata)
        print_timing_summary(timings)

    if args.show:
        cols = 2
        rows = len(classes)
        fig, axes = plt.subplots(rows, cols, figsize=(10, max(4, 4 * rows)))
        axes = np.atleast_2d(axes)
        for index, class_name in enumerate(classes):
            heatmap = sim[index].detach().cpu().numpy()
            heatmap_rgb = apply_heatmap(heatmap)
            overlay = blend_overlay(image_np, heatmap_rgb)
            axes[index, 0].imshow(heatmap_rgb)
            axes[index, 0].set_title(f"{class_name} heatmap")
            axes[index, 1].imshow(overlay)
            axes[index, 1].set_title(f"{class_name} overlay")
            axes[index, 0].axis("off")
            axes[index, 1].axis("off")
        plt.tight_layout()
        plt.show()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.show_labels and args.heatmaps:
        raise ValueError("--show-labels is only supported in final segmentation mode, not with --heatmaps.")
    if args.label_min_area < 0:
        raise ValueError("--label-min-area must be zero or a positive integer.")

    classes = parse_classes(args.classes)
    device = resolve_device(args.device)

    base_output_dir = Path(args.output_dir)
    output_dir = create_execution_dir(base_output_dir)

    if args.heatmaps:
        run_heatmap_mode(args, classes, device, output_dir)
    else:
        run_mask_mode(args, classes, device, output_dir)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
