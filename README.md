# OOF 3D - Optimally Oriented Flux for 3D Curvilinear Structure Detection

## Description

**OOF 3D** is a Python implementation of the Optimally Oriented Flux (OOF) filter for detecting 3D curvilinear structures in volumetric images. This implementation is based on the seminal work by M.W.K. Law and A.C.S. Chung from the Hong Kong University of Science and Technology.

The OOF filter is widely used in medical imaging for vessel detection, neurite tracing, and fiber tracking in diffusion MRI. It analyzes the second-order structure tensor to compute the optimally oriented flux through spherical surfaces, enabling robust detection of tubular structures at multiple scales.

This Python implementation is ported from the original MATLAB code and provides efficient computation using FFT-based convolutions for real-world applications.

## Features

- **Multi-scale Detection**: Automatically searches across multiple radii to find optimal structure scales
- **Eigenvalue Analysis**: Computes eigenvalues and eigenvectors of the 3D structure tensor
- **Vesselness Response**: Returns a vesselness measure indicating the likelihood of curvilinear structures
- **Scale Selection**: Automatically determines the optimal scale for each detected structure
- **Eigenvector Fields**: Provides principal direction vectors for tracking and visualization
- **FFT-based Implementation**: Efficient computation using Fourier domain operations
- **Configurable Parameters**: Flexible options for pixel spacing, normalization, and response types

## Installation

```bash
pip install numpy scipy
```

Clone this repository and import the module:

```python
from oof import oof_3d
```

## Quick Start

```python
import numpy as np
from oof import oof_3d

# Load your 3D image (numpy array)
image = np.load('your_volume.npy')

# Define the radii to search
radii = [2.0, 3.0, 4.0, 5.0, 6.0]

# Run OOF detection
vesselness, scale, vx, vy, vz = oof_3d(image, radii)

# vesselness: 3D array of vesselness responses
# scale: 3D array of optimal radii at each voxel
# vx, vy, vz: Components of the principal eigenvector field
```

## Function Signature

```python
oof_3d(image, radii, **options)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | numpy.ndarray | 3D input image (D × H × W) |
| `radii` | list | List of radii (in voxels) to search |
| `spacing` | list (optional) | Pixel spacing [z, x, y]. Default: [1, 1, 1] |
| `sigma` | float (optional) | Gaussian smoothing sigma. Default: min(pixelSpacing) |
| `responsetype` | int (optional) | Response type. Default: 0 |
| `normalizationtype` | int (optional) | Normalization type (0 or 1). Default: 1 |
| `useabsolute` | int (optional) | Use absolute eigenvalues (0 or 1). Default: 1 |

### Return Values

| Output | Type | Description |
|--------|------|-------------|
| `oofv` | numpy.ndarray | 3D vesselness response map |
| `whatScale` | numpy.ndarray | 3D map of optimal radii |
| `Voutx` | numpy.ndarray | X component of eigenvector field |
| `Vouty` | numpy.ndarray | Y component of eigenvector field |
| `Voutz` | numpy.ndarray | Z component of eigenvector field |

## Advanced Usage

### With Custom Pixel Spacing

```python
# For medical images with known voxel sizes (e.g., 0.5mm × 0.5mm × 1.0mm)
vesselness, scale, vx, vy, vz = oof_3d(
    image, 
    radii=[2, 3, 4, 5, 6],
    spacing=[1.0, 0.5, 0.5]  # [z, x, y]
)
```

### Combining with Other Processing

```python
# Apply Gaussian smoothing first
from scipy.ndimage import gaussian_filter

smoothed = gaussian_filter(image.astype(float), sigma=1.0)

# Then run OOF
vesselness, scale, vx, vy, vz = oof_3d(smoothed, radii=[3, 4, 5])
```

## Algorithm Overview

The Optimally Oriented Flux (OOF) filter works by:

1. **Computing the Second-Order Structure Tensor**: For each voxel, the image gradients are used to construct a 3×3 structure tensor.

2. **Optimally Oriented Flux Calculation**: The flux through a sphere of radius r is computed in the frequency domain using FFT. The orientation that maximizes the flux is found analytically.

3. **Eigenvalue Analysis**: The eigenvalues of the optimized tensor reveal the local structure:
   - |λ₁| ≥ |λ₂| ≥ |λ₃|
   - λ₂ ≈ 0, λ₃ < 0 → Tubular structure (vessel)
   - λ₁ ≈ λ₂ ≈ λ₃ > 0 → Blob-like structure
   - λ₁ ≈ λ₂ ≈ λ₃ < 0 → Dark blob

4. **Multi-Scale Response**: The filter is applied at multiple scales (radii), and the scale with maximum response is selected.

5. **Vesselness Measure**: The final vesselness is computed as λ₂ + λ₃, which is positive for tubular structures and negative for other structures.

## References

1. M.W.K. Law and A.C.S. Chung, "Three Dimensional Curvilinear Structure Detection using Optimally Oriented Flux," in *Proceedings of the 10th European Conference on Computer Vision (ECCV 2008)*, Marseille, France, 2008, pp. 368-382.

2. M.W.K. Law, Y. Wang, C. Shu, C. Qin, and A.C.S. Chung, "Dilated Divergence based Scale-Space Representation for Curve Analysis," in *Proceedings of the 12th European Conference on Computer Vision (ECCV 2012)*, Florence, Italy, 2012, pp. 448-461.

## License

This implementation is provided for research purposes. Please cite the original papers when using this code in your research.

## Acknowledgments

- Original MATLAB implementation by Max W.K. Law
- Python port by Jierong Wang (jwangdh@connect.ust.hk)

