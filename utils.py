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


def save_gif_rotating_mip(volume, voxel_spacing=(1.0, 1.0, 1.0),
                           output_path="outputs/mip_rotating.gif",
                           cmap="bone", n_angles=36, interval=80):
    """
    GIF of rotating MIP on the coronal-sagittal plane.
    Rotation is around the Z-axis (axial), projecting along Y (sagittal view)
    Aspect ratio corrected using physical voxel spacing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dz, dy, dx = voxel_spacing

    img_min = np.amin(volume)
    img_max = np.amax(volume)
    cm = matplotlib.colormaps[cmap]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    frames = []

    for alpha in np.linspace(0, 360 * (n_angles - 1) / n_angles, num=n_angles):
        rotated = rotate_on_axial_plane(volume, alpha)
        # MIP along axis=2 (X) gives coronal-like projection; axis=1 gives sagittal-like
        # Rotating then projecting along axis=1 sweeps coronal->sagittal->coronal
        projection = MIP_coronal_plane(rotated)  # coronal-sagittal sweep
        norm = matplotlib.colors.Normalize(vmin=img_min, vmax=img_max)
        f = ax.imshow(projection, cmap=cm, norm=norm, aspect=dz/dx, animated=True)
        title = ax.set_title(f"MIP — {alpha:.0f}°", animated=True)
        frames.append([f, title])

    anim = animation.ArtistAnimation(fig, frames, interval=interval, blit=False)
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
    import SimpleITK as sitk
    dz, dy, dx = voxel_spacing
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing((dx, dy, dz))   # SimpleITK: (x, y, z) order
    return img

def sitk_to_volume(sitk_img):
    """Convert SimpleITK image back to numpy array."""

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

    # normalize to [0,1]
    def norm01(v):
        v = v.astype(np.float64)
        return (v - v.min()) / (v.max() - v.min() + 1e-10)

    mr_n  = norm01(mr_volume)
    pb_n  = norm01(pet_before_resampled)
    pa_n  = norm01(pet_after)

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
    Both volumes must already be in the same physical space (after coregistration).
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
    
    Both volumes are in the same physical space after coregistration.
    The GIF shows:
      - Left panel:  MR MIP (reference)
      - Middle panel: PET MIP (coregistered)
      - Right panel:  Alpha fusion
    
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

