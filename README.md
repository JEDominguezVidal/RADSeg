<h1 align="center">RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models</h1>

<p align="center">
  <a href="https://oasisartisan.github.io/"><strong>Omar Alama</strong></a>*
  ·
  <a href="https://www.linkedin.com/in/darshil-jariwala"><strong>Darshil Jariwala</strong></a>*
  ·
  <a href="https://avigyanbh.github.io/"><strong>Avigyan Bhattacharya</strong></a>*
  <br>
  <a href="https://seungchan-kim.github.io/"><strong>Seungchan Kim</strong></a>
  ·
  <a href="https://theairlab.org/team/wenshan/"><strong>Wenshan Wang</strong></a>
  ·
  <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
  </br>
  <sub><i>*Equal Contribution</i></sub>
</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2511.19704">Paper</a> | <a href="https://radseg-ovss.github.io/">Project Page</a> | <a href="https://huggingface.co/spaces/theairlabcmu/RADSeg">Demo</a></h3>
  
  <div align="center"></div>

![RADSeg overview](assets/abstract_figure.jpg)

**RADSeg** is a framework leveraging a single agglomerative vision foundation model **RADIO** to improve zero-shot Open-Vocabulary Semantic Segmentation (OVSS) in 2D and 3D ! Remarkably, RADSeg-base (105M) outperforms previous combinations of huge vision models (850-1350M) in mIoU, achieving state-of-the-art accuracy with substantially lower computational and memory cost.

**Key Features:**
- **Unified Backbone**: Stop cascading multiple heavy foundation models. RADSeg Unlocks efficient OVSS with RADIO.
- **Efficiency**: 3.95x faster inference and 2.5x fewer parameters than comparable state-of-the-art methods.
- **Performance**: Significant mIoU improvements (6-30% on base ViT class) across benchmarks.

[Try our demo !](https://huggingface.co/spaces/theairlabcmu/RADSeg)

## Environment Setup

RADSeg is currently set up around Python 3.11, PyTorch 2.4.0, torchvision 0.19.0 and CUDA 12.1. The commands below assume you are running from the repository root unless a section explicitly changes directory.

### Option 1: Conda
Create the original conda environment and install the base dependencies:
```bash
conda env create -f environment.yml
conda activate radseg
```

### Option 2: Virtual Environment (`venv`)
If you prefer a standard virtual environment instead of conda, first make sure Python 3.11 and the `venv` module are installed on your system. Then create and activate the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` mirrors the base dependencies from `environment.yml` and targets the same CUDA 12.1 wheel index used by the conda setup. If you need a CPU-only install or a different CUDA version, adjust the first lines of `requirements.txt` to use the matching PyTorch wheel index from the [official PyTorch installation matrix](https://docs.pytorch.org/get-started/previous-versions/).

Quick verification:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Important notes for either setup path:
- The project is not packaged as an installable wheel, so run the commands below from the repository root unless the README tells you to `cd` elsewhere.
- The first time you load RADSeg, `torch.hub` will download the RADIO backbone and adaptor weights from `NVlabs/RADIO`, so internet access is required on first use.
- SAM refinement is optional, but if you enable `sam_refinement` or `--sam_refine`, you must separately download the `sam_vit_h_4b8939.pth` checkpoint from the [official Segment Anything release](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). The default config assumes a local file with that name, so either place it in your working directory or update the `sam_ckpt` path in your config/code.

Additional dependencies for 2D evaluation:
1. Activate the conda environment or the virtual environment created above, then install the OpenMMLab dependencies inside that same environment:
   ```bash
   python -m pip install mmengine==0.10.1
   python -m pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
   python -m pip install mmsegmentation==1.2.2
   ```
   These packages are intentionally not included in `requirements.txt`, because `mmcv` must match the exact PyTorch/CUDA combination you installed.

2. **MMSegmentation Compatibility:** 
   In `{site-packages-path}/mmseg/__init__.py`, you may need to update the `mmcv` version check (Approved by original mmcv author). Change:
   ```python
   assert (mmcv_min_version <= mmcv_version < mmcv_max_version)
   ```
   to:
   ```python
   assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)
   ```

Additional dependencies for 3D evaluation:
Initialize the RayFronts submodule first:
```bash
git submodule update --init --recursive
```

Then follow the minimal setup instructions of [RayFronts Environment Setup](https://github.com/RayFronts/RayFronts?tab=readme-ov-file#environment-setup). The base `requirements.txt` in this repository covers RADSeg itself, but not the full RayFronts 3D evaluation stack.

## Quickstart

### Torch Hub
You can easily load RADSeg using Torch Hub for integration into your own projects:

```python
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

with torch.inference_mode():
    # Define labels for zero-shot segmentation
    labels = ["sky", "grass", "sheep", "mountain"]
    
    # Load RADSeg model
    model = torch.hub.load(
        'RADSeg-OVSS/RADSeg', 'radseg_encoder',  model_version="c-radio_v3-l",  lang_model="siglip2",
         device="cuda",
         predict=True, # Set to false to return 
         sam_refinement=False, # Set to true for RADSeg+
         classes=labels)

    # Prepare image
    img = Image.open('assets/example1.jpg').convert('RGB')
    img_tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')

    seg_probs = model.encode_image_to_feat_map(img_tensor) # [1, len(labels)+1, H, W]

    for i in range(len(labels)):
        plt.subplot(2, 2, i+1)
        plt.imshow(seg_probs[0, i+1].cpu())
        plt.title(labels[i])
    plt.show()
```

The first call to `torch.hub.load(...)` will download the RADIO weights and adaptors if they are not already cached locally.

### Gradio Demo
To test RADSeg on your own images using an interactive Gradio interface (Available online [here](https://huggingface.co/spaces/theairlabcmu/RADSeg)):

1. **Activate the environment you created in the setup section**:
   ```bash
   conda activate radseg
   # or
   source .venv/bin/activate
   ```

2. **Run the app from the repository root**:
   ```bash
   python radseg_demo.py
   ```
This will launch an interface where you can upload images, add custom text prompts, and adjust model parameters

### Minimal CLI Demo
For headless usage without Gradio, use the minimal command-line demo:

```bash
python radseg_minimal_demo.py \
  --image assets/example1.jpg \
  --classes sky,road,car
```

This mode saves a final segmentation mask, a colorized mask, an overlay, the raw probability tensor, and metadata to `outputs/minimal_demo/` by default.

To generate per-class heatmaps instead of the final segmentation outputs, add `--heatmaps`:

```bash
python radseg_minimal_demo.py \
  --image assets/example1.jpg \
  --classes sky,road,car \
  --heatmaps
```

Useful options:
- `--model-version c-radio_v3-b` or another RADIO variant supported by the upstream model hub.
- `--lang-model siglip2` to choose the language adaptor.
- `--output-dir outputs/my_run` to choose where results are saved.
- `--show` to display the saved results with `matplotlib`.
- `--sam-refinement --sam-ckpt sam_vit_h_4b8939.pth` to enable SAM refinement in final segmentation mode.

The first run will still download the RADIO weights if they are not already cached locally.

## 2D Evaluation

### Dataset Preparation
Please follow the [MMSegmentation data preparation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and process the 5 2D datasets.

### Running Evaluation 
To evaluate RADSeg on a specific 2D dataset, activate your environment, switch to the `evaluation/2d` directory and run:

```bash
cd evaluation/2d
python eval.py \
  --config configs/mid_res_configs/cfg_voc20.py \
  --model_version c-radio_v3-b \
  --lang_model siglip2 \
  --scra_scaling 10.0 \
  --scga_scaling 10.0 \
  --work-dir ./work_logs/ \
  --sam_refine
```

Arguments:
- `--config`: Path to the mmseg config file.
- `--model_version`: RADIO model version (e.g., `c-radio_v3-b`).
- `--lang_model`: Language model to use (e.g., `siglip2`).
- `--scra_scaling`: Scaling factor for Self-Correlating Recursive Attention (SCRA).
- `--scga_scaling`: Scaling factor for Self-Correlating Global Aggregation (SCGA).
- `--sam_refine`: Enable RADIO-SAM mask refinement for RADSeg+ performance (include flag to enable).

If you enable `--sam_refine`, make sure the `sam_ckpt` path in your config points to a valid `sam_vit_h_4b8939.pth` checkpoint file.

To run evaluation across multiple resolutions and configs as defined in `eval_all.py`:

```bash
cd evaluation/2d
python eval_all.py
```
This script iterates over defined configurations (Low Resolution, Mid Resolution and High Resolution) and runs the evaluation automatically.

## 3D Evaluation

Before running 3D evaluation commands, make sure the RayFronts submodule has been initialized with `git submodule update --init --recursive`.

### Dataset Preparation
Please follow the guidelines and dataset download links provided by [RayFronts Datasets](https://github.com/RayFronts/RayFronts/tree/main/rayfronts/datasets#datasets--data-sourcesstreams) to process and prepare the 3 datasets (Replica - NiceReplica version, ScanNet, ScanNet++) used for 3D evaluation.

### Running Evaluation 

To evaluate RADSeg on a specific 3D dataset, switch to the `evaluation/3d` directory and run:

```bash
cd evaluation/3d
PYTHONPATH="../../:$PYTHONPATH" python RayFronts/scripts/semseg_eval.py \
  --config-dir ./configs/ \
  --config-name replica_radseg.yaml \
  dataset.path="path/to/your/dataset"
```

**Available Config Files:**
- `replica_radseg.yaml` - For Replica dataset (NiceReplica version)
- `scannet_radseg.yaml` - For ScanNet dataset
- `scannetpp_radseg.yaml` - For ScanNet++ dataset

**Running Multiple Scenes:**

The config files support Hydra sweeper for running multiple scenes in parallel. To run all scenes defined in the config's `hydra.sweeper.params.dataset.scene_name` list, simply run the command with an added argument `--multirun`:

```bash
PYTHONPATH="../../:$PYTHONPATH" python RayFronts/scripts/semseg_eval.py \
  --config-dir ./configs/ \
  --config-name scannet_radseg.yaml \
  dataset.path="path/to/scannet" --multirun
```

**Results:**

Evaluation results will be saved in the directory specified by `eval_out` in your config file (default: `eval_out/radseg/`). The results include per-scene metrics and aggregated statistics.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{alama2025radseg,
  title={RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models},
  author={Alama, Omar and Jariwala, Darshil and Bhattacharya, Avigyan and Kim, Seungchan and Wang, Wenshan and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2511.19704},
  year={2025}
}
```

## Acknowledgements

This codebase is built upon [AM-RADIO](https://github.com/NVlabs/RADIO), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Trident](https://github.com/YuHengsss/Trident), and [RayFronts](https://github.com/RayFronts/RayFronts). We thank the authors for their open-source contributions.
