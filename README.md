# 11763 — Medical Image Processing: Final Project

**Author:** Marta González Juan  
**Course:** Medical Image Processing (11763)  
**Dataset:** Dynamic brain PET study (BRAIN DINAMIC COLINA) + MR image (AX 3D T1)

## Project Structure

- `01_dicom_loading.ipynb` — DICOM loading, PET rearrangement, visualization and animations
- `02_coregistration.ipynb` — 3D rigid coregistration of PET average to MR using SimpleITK
- `03_segmentation.ipynb` — 3D tumour segmentation using AI model (in progress)
- `utils.py` — all helper functions used across notebooks

## How to run

Run the notebooks in order: 01 - 02 - 03

### Dependencies

```bash
pip install pydicom SimpleITK numpy scipy matplotlib pillow
```

### Data

Place the DICOM files in `Data/FORISI/`:

## Outputs

Generated GIFs are saved to `Data/outputs/`.
