# ==========================================
# Project: Image Tampering Detection Pipeline
# Owner & Lead Developer: Vaishnav Anand
# ==========================================
# inference.py — Enhanced with Grad-CAM metrics:
#   Pointing Game Accuracy, Average Drop,
#   Increase in Confidence + saved plots
# ==========================================

import os
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive — saves files without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageChops, ImageEnhance

# ──────────────────────────────────────────────
# 0.  Output directory for all saved plots
# ──────────────────────────────────────────────
RESULTS_DIR = 'inference_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1.  On-the-fly ELA Generator
# ──────────────────────────────────────────────
def generate_ela(image_path, quality=90):
    temp_filename = 'temp_inference_ela.jpg'
    try:
        original = Image.open(image_path).convert('RGB')
        original.save(temp_filename, 'JPEG', quality=quality)
        compressed = Image.open(temp_filename)
        ela_image  = ImageChops.difference(original, compressed)

        extrema  = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0:
            max_diff = 1

        scale     = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image, original
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ──────────────────────────────────────────────
# 2.  Grad-CAM Engine
# ──────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        """Returns a normalised heatmap in [0, 1] of shape (H, W)."""
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        output[0][class_idx].backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations  = self.activations.detach()[0]

        for i in range(activations.size(0)):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        return heatmap


# ──────────────────────────────────────────────
# 3.  Grad-CAM Quality Metrics
# ──────────────────────────────────────────────

def metric_pointing_game(heatmap, gt_mask_np):
    """
    Pointing Game Accuracy
    ─────────────────────
    Checks whether the pixel with the MAXIMUM activation in the heatmap
    falls inside the ground-truth tampered region.

    Args:
        heatmap    : 2-D numpy array [0,1], shape (H, W)
        gt_mask_np : binary numpy array {0,1}, shape (H, W)
                     1 = tampered region, 0 = background

    Returns:
        hit  : bool  — True if max-point is inside GT region
        score: float — 1.0 (hit) or 0.0 (miss)
    """
    h, w  = heatmap.shape
    gt_resized = cv2.resize(gt_mask_np.astype(np.float32), (w, h),
                            interpolation=cv2.INTER_NEAREST)
    gt_binary  = (gt_resized > 0.5).astype(np.uint8)

    max_idx    = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    hit        = bool(gt_binary[max_idx[0], max_idx[1]])
    return hit, float(hit)


def metric_average_drop(model, input_tensor, original_confidence, class_idx):
    """
    Average Drop (AD)
    ─────────────────
    Measures how much the model's confidence DROPS when only the
    Grad-CAM–highlighted pixels (top-50 % of heatmap) are kept.

    AD = max(0, original_conf - masked_conf) / original_conf  × 100 %

    A LOWER average drop means the heatmap faithfully captures the
    truly important region (masking them hurts the model most).

    Returns:
        ad : float — average drop percentage in [0, 100]
    """
    # 1. Generate heatmap for masking
    cam_tmp = GradCAM(model, model.layer4[-1].conv3)
    with torch.enable_grad():
        t = input_tensor.clone().detach().requires_grad_(True)
        hm = cam_tmp.generate_heatmap(t, class_idx)

    hm_resized = cv2.resize(hm, (224, 224))
    threshold  = np.percentile(hm_resized, 50)           # top-50 % mask
    mask       = torch.tensor(
        (hm_resized >= threshold).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(input_tensor.device)  # (1,1,H,W)

    masked_input = input_tensor * mask

    model.eval()
    with torch.no_grad():
        out_masked   = model(masked_input)
        prob_masked  = F.softmax(out_masked, dim=1)[0][class_idx].item()

    drop = max(0.0, original_confidence - prob_masked) / (original_confidence + 1e-8)
    return drop * 100.0


def metric_increase_in_confidence(model, input_tensor,
                                   original_confidence, class_idx):
    """
    Increase in Confidence (IC)
    ───────────────────────────
    Measures the fraction of test cases where MASKING the
    NON-SALIENT region (bottom-50 % of heatmap) actually increases
    model confidence.  A high IC % means the heatmap correctly
    identifies the regions important for the prediction.

    For a SINGLE image this returns 1.0 (increase) or 0.0 (no increase).

    Returns:
        ic_flag     : bool  — True if confidence increased
        ic_delta    : float — absolute change in confidence (can be negative)
    """
    cam_tmp = GradCAM(model, model.layer4[-1].conv3)
    with torch.enable_grad():
        t  = input_tensor.clone().detach().requires_grad_(True)
        hm = cam_tmp.generate_heatmap(t, class_idx)

    hm_resized = cv2.resize(hm, (224, 224))
    threshold  = np.percentile(hm_resized, 50)           # keep top-50 %
    mask       = torch.tensor(
        (hm_resized >= threshold).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(input_tensor.device)

    masked_input = input_tensor * mask

    model.eval()
    with torch.no_grad():
        out_masked  = model(masked_input)
        prob_masked = F.softmax(out_masked, dim=1)[0][class_idx].item()

    delta   = prob_masked - original_confidence
    ic_flag = delta > 0
    return ic_flag, delta


# ──────────────────────────────────────────────
# 4.  Model Loader
# ──────────────────────────────────────────────
def load_trained_model(weights_path, device):
    print("Loading model architecture and weights...")
    model = models.resnet50(weights=None)
    num_ftrs   = model.fc.in_features
    model.fc   = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)


# ──────────────────────────────────────────────
# 5.  Build Superimposed Heatmap (OpenCV)
# ──────────────────────────────────────────────
def superimpose_heatmap(orig_img_pil, heatmap, alpha=0.4):
    """Returns RGB numpy array with JET heatmap overlaid on the image."""
    orig_cv    = cv2.cvtColor(np.array(orig_img_pil), cv2.COLOR_RGB2BGR)
    orig_cv    = cv2.resize(orig_cv, (224, 224))
    hm_resized = cv2.resize(heatmap, (224, 224))
    hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)

    blended = np.clip(hm_colored * alpha + orig_cv * (1 - alpha), 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────
# 6.  Visualise & Save all results
# ──────────────────────────────────────────────
def process_and_visualize(image_path, gt_dir, model, device, save_prefix=None):
    """
    Main pipeline:
      • ELA → model prediction + confidence
      • Grad-CAM heatmap
      • Three Grad-CAM quality metrics (AD, IC, PG if GT mask available)
      • Saves two figures to RESULTS_DIR:
          <prefix>_overview.png   — original / GT / Grad-CAM panel
          <prefix>_metrics.png    — metric bar chart
    """
    classes = ['Authentic', 'Tampered']

    # ── ELA ────────────────────────────────────
    ela_img, orig_img = generate_ela(image_path)

    # ── Ground-truth mask (optional) ───────────
    base_name        = os.path.splitext(os.path.basename(image_path))[0]
    gt_filename_base = base_name + "_gt"
    gt_img_pil       = None
    gt_mask_np       = None    # binary {0,1}

    if os.path.exists(gt_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            temp_path = os.path.join(gt_dir, gt_filename_base + ext)
            if os.path.exists(temp_path):
                gt_img_pil = Image.open(temp_path).convert('RGB')
                gt_gray    = np.array(gt_img_pil.convert('L'))
                gt_mask_np = (gt_gray > 128).astype(np.uint8)
                break

    # ── Prepare input tensor ───────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(ela_img).unsqueeze(0).to(device)

    # ── Prediction ─────────────────────────────
    model.eval()
    with torch.no_grad():
        output        = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    class_idx         = preds.item()
    prediction_label  = classes[class_idx]
    confidence_score  = confidence.item()          # float in [0,1]
    confidence_pct    = confidence_score * 100.0

    print(f"\n{'─'*50}")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {prediction_label}  ({confidence_pct:.2f}%)")

    # ── Grad-CAM heatmap ───────────────────────
    target_layer = model.layer4[-1].conv3
    cam          = GradCAM(model, target_layer)

    with torch.enable_grad():
        t       = input_tensor.clone().detach().requires_grad_(True)
        heatmap = cam.generate_heatmap(t, class_idx)

    superimposed = superimpose_heatmap(orig_img, heatmap)

    # ── Grad-CAM Metrics ───────────────────────
    print("\n  [Grad-CAM Metrics]")

    # 1. Average Drop
    avg_drop = metric_average_drop(model, input_tensor,
                                   confidence_score, class_idx)
    print(f"  Average Drop           : {avg_drop:.2f}%  "
          f"(lower → heatmap focuses on critical regions)")

    # 2. Increase in Confidence
    ic_flag, ic_delta = metric_increase_in_confidence(model, input_tensor,
                                                       confidence_score, class_idx)
    print(f"  Increase in Confidence : {'YES ✓' if ic_flag else 'NO ✗'}  "
          f"(delta = {ic_delta*100:+.2f}%)")

    # 3. Pointing Game (only if GT mask available)
    pg_score = None
    if gt_mask_np is not None:
        pg_hit, pg_score = metric_pointing_game(heatmap, gt_mask_np)
        print(f"  Pointing Game          : {'HIT ✓' if pg_hit else 'MISS ✗'}  "
              f"(max-activation {'inside' if pg_hit else 'outside'} tampered region)")
    else:
        print("  Pointing Game          : N/A (no ground-truth mask found)")

    print(f"{'─'*50}")

    # ── Derive save prefix ─────────────────────
    if save_prefix is None:
        save_prefix = base_name

    # ── Figure 1: Overview panel ───────────────
    n_cols = 3 if gt_img_pil is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in axes:
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    axes[0].imshow(orig_img)
    axes[0].set_title("1. Original Image",
                       color='white', fontsize=13, fontweight='bold', pad=8)

    if gt_img_pil is not None:
        axes[1].imshow(gt_img_pil)
        axes[1].set_title("2. Ground Truth Mask\n(Actual Tampered Region)",
                           color='white', fontsize=13, fontweight='bold', pad=8)
        cam_ax = axes[2]
        cam_col = 3
    else:
        cam_ax = axes[1]
        cam_col = 2

    cam_ax.imshow(superimposed)
    pred_color = '#ff4444' if prediction_label == 'Tampered' else '#44ff88'
    cam_ax.set_title(
        f"{cam_col}. Grad-CAM Heatmap\n"
        f"Prediction: {prediction_label} ({confidence_pct:.2f}%)",
        color=pred_color, fontsize=13, fontweight='bold', pad=8
    )

    # Metric text box inside the Grad-CAM panel
    metric_lines = [
        f"Avg Drop  : {avg_drop:.1f}%",
        f"IC delta  : {ic_delta*100:+.1f}%",
    ]
    if pg_score is not None:
        metric_lines.append(f"Point.Game: {'HIT ✓' if pg_score else 'MISS ✗'}")

    metric_text = "\n".join(metric_lines)
    cam_ax.text(
        0.02, 0.02, metric_text,
        transform=cam_ax.transAxes,
        fontsize=9, color='white',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.65)
    )

    plt.tight_layout(pad=1.5)
    overview_path = os.path.join(RESULTS_DIR, f"{save_prefix}_overview.png")
    plt.savefig(overview_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved overview → {overview_path}")

    # ── Figure 2: Metric Bar Chart ─────────────
    metric_names  = ['Avg Drop (%)\n(lower=better)',
                     'IC Delta (%)\n(higher=better)']
    metric_values = [avg_drop, ic_delta * 100.0]
    bar_colors    = ['#e74c3c', '#2ecc71']

    if pg_score is not None:
        metric_names.append('Pointing Game\n(1=hit, 0=miss)')
        metric_values.append(pg_score * 100.0)
        bar_colors.append('#3498db')

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#1a1a2e')

    bars = ax2.bar(metric_names, metric_values, color=bar_colors,
                   edgecolor='white', linewidth=0.7, width=0.5)

    for bar, val in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}",
                 ha='center', va='bottom',
                 color='white', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Score', color='white')
    ax2.set_title(
        f"Grad-CAM Quality Metrics\n{os.path.basename(image_path)} "
        f"— {prediction_label} ({confidence_pct:.1f}%)",
        color='white', fontsize=12, fontweight='bold'
    )
    ax2.tick_params(colors='white', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555555')
    ax2.yaxis.label.set_color('white')
    ax2.set_ylim(min(0, min(metric_values)) - 5,
                  max(100, max(metric_values)) + 15)
    ax2.axhline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.25, color='white')

    plt.tight_layout()
    metrics_path = os.path.join(RESULTS_DIR, f"{save_prefix}_gradcam_metrics.png")
    plt.savefig(metrics_path, dpi=150, facecolor=fig2.get_facecolor())
    plt.close()
    print(f"  ✅ Saved Grad-CAM metrics chart → {metrics_path}")

    return {
        'prediction':   prediction_label,
        'confidence':   confidence_pct,
        'avg_drop':     avg_drop,
        'ic_delta':     ic_delta,
        'ic_flag':      ic_flag,
        'pg_score':     pg_score,
    }


# ──────────────────────────────────────────────
# 7.  Entry Point
# ──────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    weights_file = 'tampering_model.pth'
    tp_dir  = r"D:\Research\HACKATHON\CASIA2.0_revised\Tp"
    gt_dir  = r"D:\Research\HACKATHON\casia2groundtruth-master\CASIA2.0_Groundtruth"

    try:
        test_images = [
            f for f in os.listdir(tp_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
        ]

        if not test_images:
            print(f"Error: No images found in {tp_dir}")
        else:
            TEST_IMAGE_PATH = os.path.join(tp_dir, random.choice(test_images))
            print(f"Analyzing Image: {TEST_IMAGE_PATH}")

            if os.path.exists(weights_file) and os.path.exists(TEST_IMAGE_PATH):
                model   = load_trained_model(weights_file, device)
                results = process_and_visualize(
                    TEST_IMAGE_PATH, gt_dir, model, device
                )
                print("\n  Summary:")
                for k, v in results.items():
                    print(f"    {k:15s}: {v}")
                print(f"\n✅ All outputs saved to ./{RESULTS_DIR}/")
            else:
                print("Error: Model weights or image path not found.")

    except FileNotFoundError:
        print("Error: Directory not found. Check your folder paths!")
