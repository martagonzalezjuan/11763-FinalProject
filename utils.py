import os
import math
import pydicom
import numpy as np
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import SimpleITK as sitk
from IPython.display import Image, display
import torch
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

#  Paths 
DATA_DIR = Path("Data/FORISI")
PET_PATH = DATA_DIR / "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"
MR_PATH  = DATA_DIR / "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916"

# =============================================================================
# LOADING
# =============================================================================

def get_voxel_spacing(dcm):
    """
    Extract voxel spacing (dz, dy, dx) from DICOM headers.
    Returns: (dz, dy, dx) tuple in mm
    """
    try:
        pixel_spacing = dcm.PixelSpacing
        dy = float(pixel_spacing[0])
        dx = float(pixel_spacing[1])
        # Prefer SpacingBetweenSlices over SliceThickness for inter-slice gap
        if hasattr(dcm, 'SpacingBetweenSlices') and dcm.SpacingBetweenSlices:
            dz = float(dcm.SpacingBetweenSlices)
        else:
            dz = float(dcm.SliceThickness)
    except Exception:
        dz, dy, dx = 1.0, 1.0, 1.0
    return (dz, dy, dx)

def interpolate_to_isotropic(volume, voxel_spacing):
    """Interpolate volume to isotropic voxel spacing for visualization."""
    dz, dy, dx = voxel_spacing
    target_spacing = min(dz, dy, dx)
    if abs(dz - dy) < 0.01 and abs(dy - dx) < 0.01:
        return volume
    nz, ny, nx = volume.shape
    zoom_factors = [dz / target_spacing, dy / target_spacing, dx / target_spacing]
    return scipy.ndimage.zoom(volume, zoom_factors, order=1)

def load_dicom(path):
    """Load a single DICOM file."""
    return pydicom.dcmread(path)

def rearrange_pet(pet_dcm):
    """
    Rearrange PET pixel array from (n_frames*n_slices, H, W)
    into (n_frames, n_slices, H, W) using timing headers.
    """
    raw = pet_dcm.pixel_array

    frame_times     = list(pet_dcm[0x0055, 0x1001].value)
    frame_durations = list(pet_dcm[0x0055, 0x1004].value)

    n_timeframes = len(frame_times)
    n_slices     = raw.shape[0] // n_timeframes

    pet_4d = raw.reshape(n_timeframes, n_slices, raw.shape[1], raw.shape[2])

    voxel_spacing = get_voxel_spacing(pet_dcm)

    metadata = {
        "n_timeframes": n_timeframes,
        "n_slices": n_slices,
        "frame_times_s":     list(frame_times),
        "frame_durations_s": [d / 1000 for d in frame_durations],
        "voxel_spacing": voxel_spacing
    }
    return pet_4d, metadata

# =============================================================================
# PLANE EXTRACTION
# =============================================================================

def median_axial_plane(volume):
    return volume[volume.shape[0] // 2, :, :]

def median_coronal_plane(volume):
    return np.flipud(volume[:, volume.shape[1] // 2, :])

def median_sagittal_plane(volume):
    return np.flipud(volume[:, :, volume.shape[2] // 2])

def MIP_sagittal_plane(volume):
    return np.flipud(np.max(volume, axis=2))

def MIP_coronal_plane(volume):
    return np.flipud(np.max(volume, axis=1))

def rotate_on_axial_plane(volume, angle_in_degrees):
    return scipy.ndimage.rotate(volume, angle_in_degrees, axes=(1, 2), reshape=False)

# =============================================================================
# VISUALIZATION
# =============================================================================

def show_three_planes(volume, title="", cmap="bone", voxel_spacing=(1.0, 1.0, 1.0)):
    """Show axial, coronal and sagittal middle slices with correct aspect ratio."""
    volume_iso = interpolate_to_isotropic(volume, voxel_spacing)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title)
    axes[0].imshow(median_axial_plane(volume_iso),    cmap=cmap, aspect='equal')
    axes[0].set_title("Axial");    axes[0].axis("off")
    axes[1].imshow(median_coronal_plane(volume_iso),  cmap=cmap, aspect='equal')
    axes[1].set_title("Coronal");  axes[1].axis("off")
    axes[2].imshow(median_sagittal_plane(volume_iso), cmap=cmap, aspect='equal')
    axes[2].set_title("Sagittal"); axes[2].axis("off")
    plt.tight_layout()
    plt.show()

def apply_cmap(img, cmap_name='bone'):
    norm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
    return matplotlib.colormaps[cmap_name](norm(img))

def display_gif(path):
    display(Image(filename=path))

# =============================================================================
# ANIMATION
# =============================================================================

def save_gif_three_planes(pet_4d, voxel_spacing=(1.0, 1.0, 1.0), cmap="hot",
                           output_path="outputs/pet_animation.gif", interval=200):
    """
    GIF animating the 3 median planes across all time frames.
    pet_4d shape: (n_frames, n_slices, H, W)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img_min = np.amin(pet_4d)
    img_max = np.amax(pet_4d)
    cm = matplotlib.colormaps[cmap]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("3 Median Planes — Dynamic PET")
    axes[0].set_title("Axial");    axes[0].axis("off")
    axes[1].set_title("Coronal");  axes[1].axis("off")
    axes[2].set_title("Sagittal"); axes[2].axis("off")

    frames = []
    for t in range(pet_4d.shape[0]):
        vol_iso = interpolate_to_isotropic(pet_4d[t], voxel_spacing)
        norm = matplotlib.colors.Normalize(vmin=img_min, vmax=img_max)
        fa = axes[0].imshow(median_axial_plane(vol_iso),    cmap=cm, norm=norm, animated=True)
        fc = axes[1].imshow(median_coronal_plane(vol_iso),  cmap=cm, norm=norm, animated=True)
        fs = axes[2].imshow(median_sagittal_plane(vol_iso), cmap=cm, norm=norm, animated=True)
        frames.append([fa, fc, fs])

    anim = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)
    anim.save(output_path, writer='pillow')
    print(f"GIF saved to {output_path}")
    plt.close()

# =============================================================================
# COREGISTRATION using SimpleITK 
# =============================================================================

def volume_to_sitk(volume, voxel_spacing):
    """
    Convert a numpy volume to a SimpleITK image with correct physical spacing.
    voxel_spacing: (dz, dy, dx) in mm
    SimpleITK expects spacing as (dx, dy, dz)
    """
    
    dz, dy, dx = voxel_spacing
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing((dx, dy, dz))   # SimpleITK: (x, y, z) order
    return img

def sitk_to_volume(sitk_img):
    """Convert SimpleITK image back to numpy array"""

    return sitk.GetArrayFromImage(sitk_img).astype(np.float32)

def coregister_volumes_sitk(ref_volume, inp_volume,
                             ref_spacing, inp_spacing,
                             n_iterations=300, verbose=True):
    """
    Coregister inp_volume (PET) to ref_volume (MR) using SimpleITK.

    Key improvements over PyElastix approach:
    - Uses PHYSICAL space (mm) not pixel space so different voxel sizes are handled correctly
    - Mattes Mutual Information built-in (no compilation issues)
    - Multi-resolution pyramid 
    - CenteredTransformInitializer aligns image centers before optimization

    Params
    ----------
    ref_volume   : np.ndarray (nz, ny, nx)  MR, fixed image
    inp_volume   : np.ndarray (nz, ny, nx)  PET average, moving image
    ref_spacing  : (dz, dy, dx) in mm for MR
    inp_spacing  : (dz, dy, dx) in mm for PET
    n_iterations : max iterations per resolution level
    verbose      : print progress

    Returns
    -------
    registered_volume : np.ndarray PET resampled into MR space
    final_transform   : SimpleITK transform object (for inspection/reuse)
    """

    # build SimpleITK images with correct physical spacing
    fixed  = volume_to_sitk(ref_volume, ref_spacing)
    moving = volume_to_sitk(inp_volume, inp_spacing)

    if verbose:
        print("=== SimpleITK Rigid Registration ===")
        print(f"  Fixed  (MR)  : shape={ref_volume.shape}, spacing={ref_spacing}")
        print(f"  Moving (PET) : shape={inp_volume.shape}, spacing={inp_spacing}")

    # registration framework
    R = sitk.ImageRegistrationMethod()

    # similarity metric: Mattes Mutual Information (optimal for multi-modal)
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25, seed=42)  

    # interpolator for the moving image
    R.SetInterpolator(sitk.sitkLinear)

    # optimizer: gradient descent with adaptive learning rate
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=n_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    R.SetOptimizerScalesFromPhysicalShift()

    # multi-resolution pyramid
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # initial transform: align centers of mass 
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS  # align by moments of inertia
    )
    R.SetInitialTransform(initial_transform, inPlace=False)

    # to monitor optimization progress 
    if verbose:
        def iteration_callback():
            level = R.GetCurrentLevel()
            iter_  = R.GetOptimizerIteration()
            metric = R.GetMetricValue()
            if iter_ % 50 == 0:
                print(f"  Level {level} | Iter {iter_:4d} | Metric: {metric:.5f}")
        R.AddCommand(sitk.sitkIterationEvent, iteration_callback)

    # run registration
    if verbose:
        print("\nRunning registration...")
    final_transform = R.Execute(
        sitk.Cast(fixed,  sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32)
    )

    if verbose:
        print(f"\nRegistration complete.")
        print(f"  Final metric value : {R.GetMetricValue():.5f}")
        print(f"  Optimizer stop cond: {R.GetOptimizerStopConditionDescription()}")

    # resample moving image into fixed image space
    registered_sitk = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,                    # default value for voxels outside FOV
        moving.GetPixelID()
    )

    registered_volume = sitk_to_volume(registered_sitk)
    return registered_volume, final_transform


def print_transform_params(transform):
    """Print the 6 rigid transformation parameters (3 rotations + 3 translations)."""
    params = transform.GetParameters()
    print("=== Rigid Transform Parameters ===")
    print(f"  Rotation    (rad): Rx={params[0]:.4f}, Ry={params[1]:.4f}, Rz={params[2]:.4f}")
    print(f"  Translation  (mm): tx={params[3]:.2f}, ty={params[4]:.2f}, tz={params[5]:.2f}")
    angles_deg = [math.degrees(p) for p in params[:3]]
    print(f"  Rotation    (deg): Rx={angles_deg[0]:.2f}°, Ry={angles_deg[1]:.2f}°, Rz={angles_deg[2]:.2f}°")


# =============================================================================
# QUANTITATIVE EVALUATION of coregistration
# =============================================================================

# normalize to [0,1]
def norm01(v):
        v = v.astype(np.float64)
        return (v - v.min()) / (v.max() - v.min() + 1e-10)

def mutual_information(a, b, bins=64):
        hist2d, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
        p_xy = hist2d / hist2d.sum()
        p_x  = p_xy.sum(axis=1, keepdims=True)
        p_y  = p_xy.sum(axis=0, keepdims=True)
        mask = p_xy > 0
        mi   = np.sum(p_xy[mask] * np.log(p_xy[mask] / (p_x * p_y + 1e-10)[mask]))
        return mi

def normalized_cross_correlation(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return np.sum(a * b) / (denom + 1e-10)

def evaluate_coregistration(mr_volume, pet_before, pet_after,
                             mr_spacing, pet_spacing):
    """
    Quantitative evaluation of coregistration quality.
    
    Computes Mutual Information and Normalized Cross-Correlation
    before and after registration to quantify improvement.
    
    Note: comparison is done after resampling PET to MR grid.
    """
    # resample PET (before) to MR grid for fair comparison
    zoom_factors = [mr_volume.shape[i] / pet_before.shape[i] for i in range(3)]
    pet_before_resampled = scipy.ndimage.zoom(pet_before, zoom_factors, order=1)

    mr_n  = norm01(mr_volume)
    pb_n  = norm01(pet_before_resampled)
    pa_n  = norm01(pet_after)

    mi_before  = mutual_information(mr_n, pb_n)
    mi_after   = mutual_information(mr_n, pa_n)
    ncc_before = normalized_cross_correlation(mr_n, pb_n)
    ncc_after  = normalized_cross_correlation(mr_n, pa_n)

    print("=== Coregistration Evaluation ===")
    print(f"  Mutual Information   — Before: {mi_before:.4f}  |  After: {mi_after:.4f}  |  Δ = {mi_after - mi_before:+.4f}")
    print(f"  Norm. Cross-Corr.    — Before: {ncc_before:.4f}  |  After: {ncc_after:.4f}  |  Δ = {ncc_after - ncc_before:+.4f}")

    improvement_mi  = (mi_after  - mi_before)  / (abs(mi_before)  + 1e-10) * 100
    improvement_ncc = (ncc_after - ncc_before) / (abs(ncc_before) + 1e-10) * 100
    print(f"\n  MI improvement:  {improvement_mi:+.1f}%")
    print(f"  NCC improvement: {improvement_ncc:+.1f}%")

    return {
        'MI_before': mi_before,   'MI_after': mi_after,
        'NCC_before': ncc_before, 'NCC_after': ncc_after,
    }


def plot_coregistration_checkerboard(mr_volume, pet_registered, patch_size=32):
    """
    Checkerboard visualization alternating MR and PET patches to assess coregistration quality
    if anatomical borders are continuous across patch boundaries → good registration.
    
    Shows all 3 median planes
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Coregistration Quality — Checkerboard (MR vs PET registered)", fontsize=14)
    
    planes = [
        ("Axial",    mr_volume[mr_volume.shape[0]//2],                        pet_registered[pet_registered.shape[0]//2]),
        ("Coronal",  np.flipud(mr_volume[:, mr_volume.shape[1]//2, :]),        np.flipud(pet_registered[:, pet_registered.shape[1]//2, :])),
        ("Sagittal", np.flipud(mr_volume[:, :, mr_volume.shape[2]//2]),        np.flipud(pet_registered[:, :, pet_registered.shape[2]//2])),
    ]
    
    for row, (plane_name, mr_slice, pet_slice) in enumerate(planes):
        # Normalize both to [0,1]
        mr_n  = (mr_slice  - mr_slice.min())  / (mr_slice.max()  - mr_slice.min()  + 1e-10)
        pet_n = (pet_slice - pet_slice.min()) / (pet_slice.max() - pet_slice.min() + 1e-10)
        
        # Build checkerboard
        checker = np.zeros_like(mr_n)
        for i in range(0, mr_n.shape[0], patch_size):
            for j in range(0, mr_n.shape[1], patch_size):
                if (i // patch_size + j // patch_size) % 2 == 0:
                    checker[i:i+patch_size, j:j+patch_size] = mr_n[i:i+patch_size, j:j+patch_size]
                else:
                    checker[i:i+patch_size, j:j+patch_size] = pet_n[i:i+patch_size, j:j+patch_size]
        
        axes[row, 0].imshow(mr_n,    cmap='bone'); axes[row, 0].set_title(f"MR — {plane_name}");              axes[row, 0].axis('off')
        axes[row, 1].imshow(pet_n,   cmap='hot');  axes[row, 1].set_title(f"PET registered — {plane_name}");  axes[row, 1].axis('off')
        axes[row, 2].imshow(checker, cmap='gray'); axes[row, 2].set_title(f"Checkerboard — {plane_name}");    axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# ALPHA FUSION VISUALIZATION
# =============================================================================

def show_three_planes_fusion(vol_ref, vol_inp, voxel_spacing=(1,1,1), alpha=0.3,
                              cmap_ref='bone', cmap_inp='hot', title="MR + PET Fusion"):
    """
    Alpha fusion in 3 median planes.
    Both volumes are in the same physical space.
    """
    vol_ref_iso = interpolate_to_isotropic(vol_ref, voxel_spacing)
    vol_inp_iso = interpolate_to_isotropic(vol_inp, voxel_spacing)

    # Volumes are already in same space, only minor shape differences from isotropic interp
    if vol_inp_iso.shape != vol_ref_iso.shape:
        zoom_factors = [vol_ref_iso.shape[i] / vol_inp_iso.shape[i] for i in range(3)]
        vol_inp_iso = scipy.ndimage.zoom(vol_inp_iso, zoom_factors, order=1)

    planes = {
        'Axial':    (median_axial_plane(vol_ref_iso),    median_axial_plane(vol_inp_iso)),
        'Coronal':  (median_coronal_plane(vol_ref_iso),  median_coronal_plane(vol_inp_iso)),
        'Sagittal': (median_sagittal_plane(vol_ref_iso), median_sagittal_plane(vol_inp_iso)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'{title} (α={alpha})')

    for ax, (plane_name, (ref_slice, inp_slice)) in zip(axes, planes.items()):
        ref_c = apply_cmap(ref_slice, cmap_ref)
        inp_c = apply_cmap(inp_slice, cmap_inp)
        fused = (1 - alpha) * ref_c + alpha * inp_c
        ax.imshow(fused, aspect='equal')
        ax.set_title(plane_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def save_gif_alpha_fusion_mip(volume_ref, volume_inp,
                               ref_spacing=(1.0, 1.0, 1.0),
                               inp_spacing=(1.0, 1.0, 1.0),
                               output_path="outputs/fusion_animation.gif",
                               n_angles=36, interval=100, alpha=0.3):
    """
    GIF of rotating MIP with alpha fusion of reference + registered input.
 
    Params
    ----------
    ref_spacing : (dz,dy,dx) of reference (MR), used to set aspect ratio
    inp_spacing : (dz,dy,dx) of input: after registration, PET is in MR space,
                  so pass ref_spacing for both.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dz, dy, dx = ref_spacing

    ref_min, ref_max = np.amin(volume_ref), np.amax(volume_ref)
    inp_min, inp_max = np.amin(volume_inp), np.amax(volume_inp)

    cm_ref = matplotlib.colormaps["bone"]
    cm_inp = matplotlib.colormaps["hot"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_title("MR (reference)");    axes[0].axis("off")
    axes[1].set_title("PET (coregistered)"); axes[1].axis("off")
    axes[2].set_title(f"Alpha Fusion (α={alpha})"); axes[2].axis("off")
    fig.suptitle("Rotating MIP — Coronal-Sagittal Plane")

    frames = []
    for angle in np.linspace(0, 360 * (n_angles - 1) / n_angles, num=n_angles):
        rot_ref = rotate_on_axial_plane(volume_ref, angle)
        rot_inp = rotate_on_axial_plane(volume_inp, angle)

        # Project along axis=1: sweeps coronal to sagittal as angle changes
        mip_ref = MIP_coronal_plane(rot_ref)
        mip_inp = MIP_coronal_plane(rot_inp)

        # Normalize
        norm_ref = (mip_ref - ref_min) / (ref_max - ref_min + 1e-10)
        norm_inp = (mip_inp - inp_min) / (inp_max - inp_min + 1e-10)

        # Colorize
        c_ref = cm_ref(norm_ref)
        c_inp = cm_inp(norm_inp)
        c_fus = (1 - alpha) * c_ref + alpha * c_inp

        aspect = dz / dx  # physical aspect ratio

        f0 = axes[0].imshow(c_ref, aspect=aspect, animated=True)
        f1 = axes[1].imshow(c_inp, aspect=aspect, animated=True)
        f2 = axes[2].imshow(c_fus, aspect=aspect, animated=True)
        t  = axes[2].set_title(f"Alpha Fusion (α={alpha}) — {angle:.0f}°", animated=True)
        frames.append([f0, f1, f2, t])

    anim = animation.ArtistAnimation(fig, frames, interval=interval, blit=False)
    anim.save(output_path, writer='pillow')
    print(f"GIF saved to {output_path}")
    plt.close()


# =============================================================================
# nnInteractive SEGMENTATION
# =============================================================================

def segment_with_nninteractive(mr_volume, bbox_xy, bbox_slice_z, axis='z',
                                model_dir=None, download_dir="Data/models",
                                device='cuda', verbose=True):
    """
    3D semi-automatic tumor segmentation using nnInteractive.

    The model accepts a 2D bounding box on ONE slice and propagates the
    segmentation through the whole 3D volume via AutoZoom.

    Params
    ----------
    mr_volume : np.ndarray (nz, ny, nx)
        MR volume as returned by pydicom. 
    bbox_xy : dict with keys of the 2D bbox 'x_min', 'x_max', 'y_min', 'y_max'

    bbox_slice_z : int
        Index of the slice where the bbox is drawn (the slice where the tumor
        appears largest).
    axis : 'z' (default)
        Which axis the bbox is drawn on. 'z' = axial slice.
    model_dir : str or None
        Path to nnInteractive weights folder. If None, the model is downloaded
        automatically.
    download_dir : str
        Where to cache the downloaded model.
    device : 'cuda' or 'cpu'
    verbose : bool

    Returns
    -------
    mask : np.ndarray (nz, ny, nx), uint8
        Binary tumor mask, same shape and orientation as the input mr_volume.
    """
    # -----------------------------------------------------------------
    # 1. Download model weights (only the first time)
    REPO_ID = "nnInteractive/nnInteractive"
    MODEL_NAME = "nnInteractive_v1.0"

    if model_dir is None:
        os.makedirs(download_dir, exist_ok=True)
        if verbose:
            print(f"Downloading model weights to {download_dir} (only first time)...")
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=download_dir,
        )
        model_dir = os.path.join(download_dir, MODEL_NAME)

    # -----------------------------------------------------------------
    # 2. Create inference session and load model
    if verbose:
        print(f"Loading model from: {model_dir}")
    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=verbose,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(model_dir)

    # -----------------------------------------------------------------
    # 3. Prepare image
    img_for_model = np.transpose(mr_volume, (2, 1, 0))      # numpy DICOM shape is (Z, Y, X)
    img_for_model = img_for_model[np.newaxis, ...].astype(np.float32)  # nnInteractive expects (1, X, Y, Z) 

    if verbose:
        print(f"Original MR shape (Z,Y,X): {mr_volume.shape}")
        print(f"Reshaped for nnInteractive (1,X,Y,Z): {img_for_model.shape}")

    session.set_image(img_for_model)

    # Output buffer must be 3D matching (X, Y, Z)
    target_tensor = torch.zeros(img_for_model.shape[1:], dtype=torch.uint8)
    session.set_target_buffer(target_tensor)

    # -----------------------------------------------------------------
    # 4. Build the 2D bounding box prompt
    #    BBOX format: [[x1, x2], [y1, y2], [z1, z2]]
    #    For a 2D box on axial slice z, the Z dim is [z, z+1].
    if axis == 'z':
        bbox_coords = [
            [bbox_xy['x_min'], bbox_xy['x_max']],
            [bbox_xy['y_min'], bbox_xy['y_max']],
            [bbox_slice_z, bbox_slice_z + 1],
        ]
    else:
        raise NotImplementedError("Only axis='z' (axial) is implemented.")

    if verbose:
        print(f"Bbox (x,y,z order): {bbox_coords}")

    session.add_bbox_interaction(bbox_coords, include_interaction=True)

    # -----------------------------------------------------------------
    # 5. Retrieve mask and reshape back to (Z, Y, X)
    mask_xyz = session.target_buffer.clone().cpu().numpy().astype(np.uint8)
    mask = np.transpose(mask_xyz, (2, 1, 0))     # back to (Z, Y, X)

    if verbose:
        n_vox = mask.sum()
        print(f"\nDone. Voxels in mask: {n_vox}")

    # Free GPU memory
    session.reset_interactions()
    del session
    torch.cuda.empty_cache()

    return mask
# =============================================================================
# SEGMENTATION VISUALIZATION & EVALUATION
# =============================================================================

def show_mask_overlay(volume, mask, slice_idx=None, axes=('axial', 'coronal', 'sagittal'),
                      cmap_volume='bone', mask_color=(1, 0, 0), alpha=0.4,
                      voxel_spacing=(1.0, 1.0, 1.0), title=""):
    """
    Show the volume with the segmentation mask overlaid as a colored layer.

    If slice_idx is None, uses the slice with the largest mask cross-section
    for each plane (the "most informative" slice).

    Params
    ------
    volume : np.ndarray (nz, ny, nx)
    mask : np.ndarray (nz, ny, nx) — binary or {0, 1}
    slice_idx : dict {'axial': int, 'coronal': int, 'sagittal': int} or None
    axes : tuple of planes to show
    voxel_spacing : (dz, dy, dx) for aspect ratio correction
    """
    dz, dy, dx = voxel_spacing

    # Find the slice with the largest tumor area on each plane
    if slice_idx is None:
        slice_idx = {
            'axial':    int(np.argmax(mask.sum(axis=(1, 2)))),
            'coronal':  int(np.argmax(mask.sum(axis=(0, 2)))),
            'sagittal': int(np.argmax(mask.sum(axis=(0, 1)))),
        }

    fig, ax_list = plt.subplots(1, len(axes), figsize=(5 * len(axes), 5))
    if len(axes) == 1:
        ax_list = [ax_list]
    fig.suptitle(title or "Tumor mask overlay")

    for ax, plane in zip(ax_list, axes):
        if plane == 'axial':
            img_slice = volume[slice_idx['axial']]
            msk_slice = mask[slice_idx['axial']]
            aspect = dy / dx
        elif plane == 'coronal':
            img_slice = np.flipud(volume[:, slice_idx['coronal'], :])
            msk_slice = np.flipud(mask[:, slice_idx['coronal'], :])
            aspect = dz / dx
        elif plane == 'sagittal':
            img_slice = np.flipud(volume[:, :, slice_idx['sagittal']])
            msk_slice = np.flipud(mask[:, :, slice_idx['sagittal']])
            aspect = dz / dy
        else:
            raise ValueError(f"Unknown plane: {plane}")

        # Background
        ax.imshow(img_slice, cmap=cmap_volume, aspect=aspect)

        # Mask as colored overlay (transparent where mask == 0)
        rgba = np.zeros((*msk_slice.shape, 4))
        rgba[..., 0] = mask_color[0]
        rgba[..., 1] = mask_color[1]
        rgba[..., 2] = mask_color[2]
        rgba[..., 3] = msk_slice * alpha
        ax.imshow(rgba, aspect=aspect)

        ax.set_title(f"{plane.capitalize()} (slice {slice_idx[plane]})")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def tumor_geometric_measures(mask, voxel_spacing):
    """
    Compute geometric measures of the segmented tumor.

    Params
    ------
    mask : np.ndarray (nz, ny, nx), binary
    voxel_spacing : (dz, dy, dx) in mm

    Returns
    -------
    dict with: n_voxels, volume_mm3, volume_cm3, voxel_volume_mm3,
               bbox_voxel, bbox_mm, extent_mm
    """
    dz, dy, dx = voxel_spacing
    voxel_vol = dx * dy * dz  # mm cubic

    n_vox = int(mask.sum())
    coords = np.array(np.where(mask > 0))

    if n_vox == 0:
        return {'n_voxels': 0, 'volume_mm3': 0.0}

    z_min, y_min, x_min = coords.min(axis=1)
    z_max, y_max, x_max = coords.max(axis=1)

    extent_mm = (
        (z_max - z_min + 1) * dz,
        (y_max - y_min + 1) * dy,
        (x_max - x_min + 1) * dx,
    )

    return {
        'n_voxels': n_vox,
        'volume_mm3': n_vox * voxel_vol,
        'volume_cm3': n_vox * voxel_vol / 1000.0,
        'voxel_volume_mm3': voxel_vol,
        'bbox_voxel': {
            'z': (int(z_min), int(z_max)),
            'y': (int(y_min), int(y_max)),
            'x': (int(x_min), int(x_max)),
        },
        'extent_mm': {
            'z': extent_mm[0],
            'y': extent_mm[1],
            'x': extent_mm[2],
        },
    }


def tumor_photometric_measures(volume, mask):
    """
    Compute photometric measures of the tumor (intensity statistics inside mask).
    """
    vals = volume[mask > 0]
    if len(vals) == 0:
        return {}
    return {
        'mean':   float(np.mean(vals)),
        'median': float(np.median(vals)),
        'std':    float(np.std(vals)),
        'min':    float(np.min(vals)),
        'max':    float(np.max(vals)),
        'p95':    float(np.percentile(vals, 95)),
        'p99':    float(np.percentile(vals, 99)),
    }

# =============================================================================
# PET CALIBRATION (raw to kBq/cc)
# =============================================================================

def get_pet_rescale_factor(pet_dcm):
    """
    Extract the real-world value mapping (slope, intercept) from a PET DICOM.

    Modern multi-frame PET stores the calibration inside the
    RealWorldValueMappingSequence rather than the classic RescaleSlope /
    RescaleIntercept top-level tags.

    Returns
    -------
    slope, intercept : float, float
        Such that:   value_BqmL = pixel_raw * slope + intercept
        Assumes Units == 'BQML' (Bq per mL).
    """
    if 'RealWorldValueMappingSequence' in pet_dcm:
        rwvm = pet_dcm.RealWorldValueMappingSequence[0]
        slope = float(rwvm.RealWorldValueSlope)
        intercept = float(rwvm.get('RealWorldValueIntercept', 0.0))
    elif hasattr(pet_dcm, 'RescaleSlope'):
        slope = float(pet_dcm.RescaleSlope)
        intercept = float(pet_dcm.get('RescaleIntercept', 0.0))
    else:
        raise ValueError("Could not find PET calibration factors in DICOM headers.")

    # Sanity check: ensure Units are Bq/mL
    units = pet_dcm.get('Units', None)
    if units is not None and units != 'BQML':
        print(f"WARNING: PET Units is '{units}', expected 'BQML'. "
              f"Output may not be in Bq/mL.")

    return slope, intercept


def pet_to_kbq_cc(pet_raw, pet_dcm):
    """
    Convert raw PET pixel values to kBq/cc (= kBq/mL).

    Params
    ------
    pet_raw : np.ndarray
        Raw pixel values from the PET DICOM (any shape).
    pet_dcm : pydicom Dataset
        Source DICOM, used to read the calibration factors.

    Returns
    -------
    pet_kbq_cc : np.ndarray, float32
        Same shape as pet_raw, in kBq/cc.
    """
    slope, intercept = get_pet_rescale_factor(pet_dcm)
    pet_bq_ml = pet_raw.astype(np.float32) * slope + intercept
    pet_kbq_cc = pet_bq_ml / 1000.0
    return pet_kbq_cc

def save_gif_rotating_mip_with_mask(volume, mask, voxel_spacing=(1.0, 1.0, 1.0),
                                     output_path="outputs/mip_rotating_mask.gif",
                                     cmap="bone", mask_color=(1, 0, 0),
                                     mask_alpha=0.45, n_angles=36, interval=80):
    """
    Rotating MIP with the tumour mask silhouette overlaid.

    For each rotation angle:
      - rotate both volume and mask around the axial axis
      - compute MIP of the volume (max along axis 1 -> coronal-sagittal sweep)
      - compute silhouette of the mask the same way (max projection)
      - overlay the silhouette as a coloured semi-transparent layer
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dz, dy, dx = voxel_spacing

    vmin, vmax = np.amin(volume), np.amax(volume)
    cm = matplotlib.colormaps[cmap]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    frames = []

    for angle in np.linspace(0, 360 * (n_angles - 1) / n_angles, num=n_angles):
        rotated_vol  = rotate_on_axial_plane(volume, angle)
        rotated_mask = rotate_on_axial_plane(mask.astype(np.float32), angle)

        # MIP of the volume (background)
        mip_vol  = MIP_coronal_plane(rotated_vol)
        # Silhouette of the mask (overlay)
        mip_mask = MIP_coronal_plane(rotated_mask)
        mip_mask = (mip_mask > 0.3).astype(np.float32)   # crisp silhouette

        # Render background
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        bg = ax.imshow(mip_vol, cmap=cm, norm=norm,
                       aspect=dz / dx, animated=True)

        # Render mask silhouette as a coloured rgba layer
        rgba = np.zeros((*mip_mask.shape, 4))
        rgba[..., 0] = mask_color[0]
        rgba[..., 1] = mask_color[1]
        rgba[..., 2] = mask_color[2]
        rgba[...,  3] = mip_mask * mask_alpha
        ov = ax.imshow(rgba, aspect=dz / dx, animated=True)

        title = ax.set_title(f"MIP + mask", animated=True)
        frames.append([bg, ov, title])

    anim = animation.ArtistAnimation(fig, frames, interval=interval, blit=False)
    anim.save(output_path, writer='pillow')
    print(f"GIF saved to {output_path}")
    plt.close()