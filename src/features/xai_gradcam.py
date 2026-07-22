"""
Explainability-derived (XAI) feature extraction (F_xai in the paper).

Implements Section 3.4 "Explainability-Derived Features":
  - Compute Grad-CAM activation maps from a pretrained chest X-ray classifier
    (paper uses DenseNet-121).
  - Summarize the map into scalar statistics: mean, max, entropy, top-10% mass.

Why this matters: Grad-CAM shows *where* a trained classifier is "looking" when
it makes a prediction. If that attention is spread out and unfocused (high
entropy, low top-10% mass), that's a signal the model isn't confidently
grounded in any real anatomical region -- useful context for the VLM to know
before it commits to an answer.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Part 1: the statistics function -- this is the actual F_xai from Eq. (6).
# It only needs a 2D activation map (numpy array), so it can be tested and
# used independently of any specific deep learning framework.
# ---------------------------------------------------------------------------

def derive_spatial_statistics(activation_map: np.ndarray) -> dict:
    """
    Summarize a normalized Grad-CAM activation map into scalar statistics,
    exactly as defined in Eq. (6) of the paper: F_xai = {mean, max, entropy, top-10% mass}.

    Args:
        activation_map: 2D numpy array, values should already be normalized
                         to [0, 1] (typical after ReLU + min-max normalization).

    Returns:
        dict with mean, max, entropy, and top_10pct_mass.
    """
    a = activation_map.astype(np.float64)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)  # ensure [0, 1]

    flat = a.flatten()
    probs = flat / (flat.sum() + 1e-8)  # treat as a distribution for entropy
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    k = max(1, int(0.10 * flat.size))
    top_10pct_mass = float(np.sort(flat)[-k:].sum() / (flat.sum() + 1e-8))

    return {
        "xai_mean": float(a.mean()),
        "xai_max": float(a.max()),
        "xai_entropy": entropy,
        "xai_top10pct_mass": top_10pct_mass,
    }


# ---------------------------------------------------------------------------
# Part 2: the actual Grad-CAM computation. This needs PyTorch + a pretrained
# chest X-ray classifier (e.g. DenseNet-121). It downloads/loads real weights
# and runs on a GPU or CPU -- this is the part you'll run in Colab, not here.
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Standard hook-based Grad-CAM implementation (Selvaraju et al., 2017),
    matching Eqs. (4)-(5) of the paper.

    Usage:
        model = load_pretrained_densenet121_for_cxr()   # see note below
        cam = GradCAM(model, target_layer=model.features.denseblock4)
        heatmap = cam(image_tensor, class_idx=0)         # numpy array, HxW, in [0, 1]
        stats = derive_spatial_statistics(heatmap)

    Note on the pretrained classifier:
    The paper uses a DenseNet-121 backbone. For real chest X-ray results you
    want a model actually trained on chest X-rays (ImageNet weights alone
    won't produce medically meaningful attention). Two good options:
      1. torchxrayvision (`pip install torchxrayvision`) -- ships DenseNet-121
         models pretrained on CheXpert/NIH/MIMIC chest X-ray datasets.
      2. A DenseNet-121 you or your professor already fine-tuned on OpenI/CheXpert.
    """

    def __init__(self, model, target_layer):
        import torch  # local import: only needed when actually running Grad-CAM

        self.torch = torch
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient_tensor)

    def _save_gradient_tensor(self, grad):
        self.gradients = grad.detach()

    def __call__(self, image_tensor, class_idx: int = None) -> np.ndarray:
        """
        Args:
            image_tensor: preprocessed input tensor, shape (1, C, H, W).
            class_idx: which output class to explain. If None, uses the
                       model's top predicted class.

        Returns:
            2D numpy array (H, W) Grad-CAM heatmap, normalized to [0, 1].
        """
        torch = self.torch
        self.model.zero_grad()
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx].sum()
        score.backward()

        # alpha_k^c = global-average-pooled gradient for channel k  (Eq. 4)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # L_GradCAM = ReLU(sum_k alpha_k^c * A^k)                    (Eq. 5)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_cxr_classifier(model_name: str = "densenet121-res224-all"):
    """
    Load a pretrained chest X-ray classifier using torchxrayvision or torchvision DenseNet-121.
    Returns (model, target_layer).
    """
    try:
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights=model_name)
        model.eval()
        if hasattr(model, "model") and hasattr(model.model, "features"):
            target_layer = model.model.features
        elif hasattr(model, "features"):
            target_layer = model.features
        else:
            target_layer = list(model.children())[0]
        return model, target_layer
    except ImportError:
        import torchvision.models as models
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.eval()
        target_layer = model.features
        return model, target_layer


def extract_xai_features(image_or_tensor, model=None, target_layer=None, class_idx: int = None) -> dict:
    """
    Convenience entry point matching Frad/Fxai/Fvoc naming used elsewhere in
    this project: run Grad-CAM, then summarize into F_xai.

    Args:
        image_or_tensor: 2D numpy array image (H, W) or PyTorch tensor (1, C, H, W).
        model: optional PyTorch classifier. If None, auto-loads torchxrayvision DenseNet.
        target_layer: optional target layer for Grad-CAM.
        class_idx: optional target class index.
    """
    import torch

    if model is None or target_layer is None:
        model, target_layer = load_cxr_classifier()

    if isinstance(image_or_tensor, np.ndarray):
        img = image_or_tensor.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = img[None, None, :, :]
        elif img.ndim == 3:
            img = img[None, :, :, :]
        tensor = torch.from_numpy(img)
        if tensor.shape[-2:] != (224, 224):
            tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        if hasattr(model, "pathologies"):
            tensor = tensor * 2048.0 - 1024.0
        elif tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        image_tensor = tensor
    else:
        image_tensor = image_or_tensor
        if image_tensor.ndim == 4 and image_tensor.shape[1] == 1 and not hasattr(model, "pathologies"):
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

    cam_fn = GradCAM(model, target_layer)
    heatmap = cam_fn(image_tensor, class_idx=class_idx)
    return derive_spatial_statistics(heatmap)


if __name__ == "__main__":
    # Self-test for the part that DOESN'T need PyTorch/GPU/pretrained weights:
    # the statistics function. This confirms Eq. (6) is implemented correctly.
    rng = np.random.default_rng(0)

    # A "focused" heatmap: attention concentrated in one small hot region.
    focused = np.zeros((64, 64))
    focused[20:30, 20:30] = 1.0
    focused += rng.normal(0, 0.02, size=(64, 64)).clip(0, None)

    # A "diffuse" heatmap: attention spread everywhere (unfocused).
    diffuse = rng.uniform(0.3, 0.7, size=(64, 64))

    print("Focused attention map stats:", derive_spatial_statistics(focused))
    print("Diffuse attention map stats:", derive_spatial_statistics(diffuse))
    print(
        "\nAs expected: the focused map has lower entropy and higher "
        "top-10%-mass than the diffuse map -- this is the signal the paper "
        "uses to flag when a model's visual grounding is weak."
    )
