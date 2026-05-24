# 11763 — Medical Image Processing: Final Project

**Author:** Marta González Juan
**Course:** Medical Image Processing (11763)
**Dataset:** Dynamic brain PET study (BRAIN DINAMIC COLINA) + MR image (AX 3D T1)

## Project structure

- `01_dicom_loading.ipynb` — DICOM loading, PET rearrangement, visualization and animations
- `02_coregistration.ipynb` — 3D rigid coregistration of PET to MR using SimpleITK
- `03_segmentation.ipynb` — 3D semi-automatic tumor segmentation with nnInteractive, validated against the coregistered PET
- `utils.py` — all helper functions used across notebooks
- `requirements.txt` — Python dependencies

## How to run

Run the notebooks in order: 01 → 02 → 03.
Notebook 02 saves the coregistered PET to `Data/outputs/`, which notebook 03 reads for cross-modal validation.

## Requirements

- **GPU recommended for notebook 03** (nnInteractive runs on CUDA; tested on an NVIDIA RTX 5080).
- CPU is sufficient for notebooks 01 and 02.

### Install

```bash
pip install -r requirements.txt
```

PyTorch needs to be installed with the matching CUDA build for your GPU. For Blackwell GPUs (RTX 50xx, CUDA 12.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For other GPUs, see https://pytorch.org/get-started/locally/.

### Model weights

The nnInteractive checkpoint (~400 MB) is downloaded automatically from HuggingFace the first time `segment_with_nninteractive()` is called. It is cached under `Data/models/nnInteractive_v1.0/`.

## Data

Place the DICOM folders in `Data/FORISI/`:

- `15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916/` (MR)
- `02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916/` (dynamic PET)

## Outputs

Generated files are saved under `Data/outputs/`:

- GIFs from notebooks 01 and 02 (dynamic PET animation, rotating MIP, MR+PET fusion)
- `pet_registered.npy` and `pet_last_frame_registered.npy` from notebook 02 (consumed by notebook 03)
- The tumour mask from notebook 03 (if you decide to save it)