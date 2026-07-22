"""
Radiomics feature extraction (F_rad in the paper).

Implements Section 3.3 "Radiomic Intensity and Texture Statistics":
  - First-order intensity statistics: mean, variance, percentiles, range
  - Gray-Level Co-occurrence Matrix (GLCM) texture features: contrast, homogeneity
  - Local Binary Pattern (LBP) texture histogram

These are classical, non-learned image descriptors -- no neural network involved.
They summarize how "bright/dark" and how "textured" the X-ray is.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_intensity_stats(image: np.ndarray) -> dict:
    """
    First-order intensity statistics from a grayscale image array.

    Args:
        image: 2D numpy array (H, W), grayscale, any numeric dtype.

    Returns:
        dict of scalar statistics.
    """
    img = image.astype(np.float64)
    flat = img.flatten()

    return {
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat)),
        "std": float(np.std(flat)),
        "p10": float(np.percentile(flat, 10)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "intensity_range": float(np.max(flat) - np.min(flat)),
    }


def extract_glcm_texture(image: np.ndarray, distances=(1,), angles=(0,)) -> dict:
    """
    Gray-Level Co-occurrence Matrix texture descriptors: contrast and homogeneity,
    as described in Eq. in Section 3.3 of the paper.

    Args:
        image: 2D numpy array, grayscale. Will be rescaled to 8-bit (0-255) and
               quantized to `levels` gray levels for a tractable co-occurrence matrix.
        distances: pixel pair distances to consider.
        angles: pixel pair angles (radians) to consider.

    Returns:
        dict with contrast and homogeneity (averaged across distances/angles).
    """
    img = image.astype(np.float64)
    # Normalize to 0-255 and quantize to 32 gray levels (standard practice --
    # a full 256-level GLCM is noisy and slow for typical radiograph sizes).
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    levels = 32
    img_q = (img_norm * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(
        img_q, distances=list(distances), angles=list(angles),
        levels=levels, symmetric=True, normed=True
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))

    return {"glcm_contrast": contrast, "glcm_homogeneity": homogeneity}


def extract_lbp_texture(image: np.ndarray, n_points: int = 24, radius: int = 3) -> dict:
    """
    Local Binary Pattern texture histogram, summarized into a small set of
    scalar statistics (mean pattern value, histogram entropy, and uniformity).

    Args:
        image: 2D numpy array, grayscale.
        n_points: number of circularly symmetric neighbour points.
        radius: radius of the circle.

    Returns:
        dict of scalar LBP summary statistics.
    """
    img = image.astype(np.float64)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")

    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    hist = hist + 1e-10  # avoid log(0)

    entropy = float(-np.sum(hist * np.log2(hist)))
    uniformity = float(np.sum(hist ** 2))

    return {
        "lbp_mean": float(np.mean(lbp)),
        "lbp_entropy": entropy,
        "lbp_uniformity": uniformity,
    }


def extract_radiomics(image: np.ndarray) -> dict:
    """
    Full radiomics feature set F_rad for one image, combining intensity,
    GLCM, and LBP descriptors. This is the single entry point other modules
    should call.

    Args:
        image: 2D grayscale numpy array (H, W). If given a color image (H, W, 3),
               it will be converted to grayscale first.

    Returns:
        Flat dict of radiomics features, e.g. {"mean": ..., "glcm_contrast": ..., ...}
    """
    if image.ndim == 3:
        # Simple luminance conversion if a color image was passed in.
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    features = {}
    features.update(extract_intensity_stats(image))
    features.update(extract_glcm_texture(image))
    features.update(extract_lbp_texture(image))
    return features


if __name__ == "__main__":
    # Quick self-test with a synthetic "X-ray-like" image so you can confirm
    # the module runs correctly before plugging in real data.
    rng = np.random.default_rng(42)
    synthetic_xray = rng.normal(loc=120, scale=30, size=(256, 256)).clip(0, 255)

    feats = extract_radiomics(synthetic_xray)
    print("Radiomics features on synthetic image:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")
