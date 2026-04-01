import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    if args.sam_refinement and not Path(args.sam_ckpt).is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {args.sam_ckpt}. Disable --sam-refinement or provide a valid path."
        )

    image_pil, tensor_image = load_image(Path(args.image), device)
    image_np = np.array(image_pil)

    encoder = RADSegEncoder(
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

    seg_probs, seg_pred = encoder.encode_image_to_feat_map(
        tensor_image,
        orig_img_size=image_np.shape[:2],
        return_preds=True,
    )

    seg_probs_np = seg_probs.squeeze(0).detach().cpu().numpy()
    seg_pred_np = seg_pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    palette = build_palette(len(classes) + 1)
    color_mask = palette[seg_pred_np]
    overlay = blend_overlay(image_np, color_mask)

    image_pil.save(output_dir / "input.png")
    Image.fromarray(seg_pred_np, mode="L").save(output_dir / "mask_index.png")
    Image.fromarray(color_mask).save(output_dir / "mask_color.png")
    Image.fromarray(overlay).save(output_dir / "overlay.png")
    np.save(output_dir / "seg_probs.npy", seg_probs_np)

    metadata = {
        "mode": "mask",
        "image": str(Path(args.image)),
        "classes": classes,
        "class_indices": {"0": "__background__"} | {str(i + 1): cls for i, cls in enumerate(classes)},
        "model_version": args.model_version,
        "lang_model": args.lang_model,
        "device": device,
        "sam_refinement": args.sam_refinement,
        "sam_ckpt": args.sam_ckpt if args.sam_refinement else None,
        "prediction_thresh": args.prediction_thresh,
        "slide_crop": args.slide_crop,
        "slide_stride": args.slide_stride,
        "scra_scaling": args.scra_scaling,
        "scga_scaling": args.scga_scaling,
        "outputs": [
            "input.png",
            "mask_index.png",
            "mask_color.png",
            "overlay.png",
            "seg_probs.npy",
        ],
    }
    save_metadata(output_dir, metadata)

    if args.show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_np)
        axes[0].set_title("Input")
        axes[1].imshow(color_mask)
        axes[1].set_title("Segmentation")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


@torch.inference_mode()
def run_heatmap_mode(args, classes: list[str], device: str, output_dir: Path):
    if args.sam_refinement:
        raise ValueError("--sam-refinement is only supported in final segmentation mode, not with --heatmaps.")

    image_pil, tensor_image = load_image(Path(args.image), device)
    image_np = np.array(image_pil)

    encoder = RADSegEncoder(
        device=device,
        model_version=args.model_version,
        lang_model=args.lang_model,
        predict=False,
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
    )

    feat_map = encoder.encode_image_to_feat_map(
        tensor_image,
        orig_img_size=image_np.shape[:2],
    )
    aligned_feats = encoder.align_spatial_features_with_language(feat_map, onehot=False)
    prompt_embeds = encoder.encode_labels(classes)

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

    image_pil.save(output_dir / "input.png")

    outputs = ["input.png"]
    for index, class_name in enumerate(classes):
        heatmap = sim[index].detach().cpu().numpy()
        heatmap_rgb = apply_heatmap(heatmap)
        overlay = blend_overlay(image_np, heatmap_rgb)
        stem = f"{index + 1:02d}_{sanitize_name(class_name)}"
        Image.fromarray(heatmap_rgb).save(output_dir / f"{stem}_heatmap.png")
        Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
        np.save(output_dir / f"{stem}_scores.npy", heatmap)
        outputs.extend(
            [
                f"{stem}_heatmap.png",
                f"{stem}_overlay.png",
                f"{stem}_scores.npy",
            ]
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
    save_metadata(output_dir, metadata)

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
