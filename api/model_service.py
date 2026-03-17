# ==========================================
# Model Service — Wraps inference pipeline
# ==========================================
import io
import os
import base64
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


class GradCAM:
    """Grad-CAM engine for generating activation heatmaps."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0][class_idx].backward()
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        for i in range(activations.size(0)):
            activations[i] *= pooled_grads[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap


def _pil_to_base64(pil_img, fmt="PNG"):
    """Convert PIL image to base64 data-URI string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _numpy_rgb_to_base64(arr):
    """Convert HxWx3 uint8 numpy array (RGB) to base64 PNG."""
    img = Image.fromarray(arr)
    return _pil_to_base64(img, "PNG")


class ModelService:
    """
    Singleton-style service that loads the ResNet50 model once
    and provides an `analyze(image_bytes)` method.
    """

    CLASSES = ["Authentic", "Tampered"]

    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weights_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        print(f"[ModelService] Ready on {self.device}")

    # ── Model loading ──────────────────────────
    def _load_model(self, weights_path: str):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        return model

    # ── ELA generation ─────────────────────────
    @staticmethod
    def _generate_ela(image: Image.Image, quality: int = 90) -> Image.Image:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        ela = ImageChops.difference(image, compressed)

        extrema = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        return ela

    # ── Heatmap overlay ────────────────────────
    @staticmethod
    def _superimpose_heatmap(orig_pil: Image.Image, heatmap, alpha=0.4):
        orig_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
        orig_cv = cv2.resize(orig_cv, (224, 224))
        hm_resized = cv2.resize(heatmap, (224, 224))
        hm_colored = cv2.applyColorMap(
            np.uint8(255 * hm_resized), cv2.COLORMAP_JET
        )
        blended = np.clip(
            hm_colored * alpha + orig_cv * (1 - alpha), 0, 255
        ).astype(np.uint8)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    # ── Grad-CAM metrics ──────────────────────
    def _metric_average_drop(self, input_tensor, original_confidence, class_idx):
        cam = GradCAM(self.model, self.model.layer4[-1].conv3)
        with torch.enable_grad():
            t = input_tensor.clone().detach().requires_grad_(True)
            hm = cam.generate_heatmap(t, class_idx)
        hm_resized = cv2.resize(hm, (224, 224))
        threshold = np.percentile(hm_resized, 50)
        mask = torch.tensor(
            (hm_resized >= threshold).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        masked_input = input_tensor * mask
        self.model.eval()
        with torch.no_grad():
            out = self.model(masked_input)
            prob = F.softmax(out, dim=1)[0][class_idx].item()
        drop = max(0.0, original_confidence - prob) / (original_confidence + 1e-8)
        return drop * 100.0

    def _metric_increase_in_confidence(self, input_tensor, original_confidence, class_idx):
        cam = GradCAM(self.model, self.model.layer4[-1].conv3)
        with torch.enable_grad():
            t = input_tensor.clone().detach().requires_grad_(True)
            hm = cam.generate_heatmap(t, class_idx)
        hm_resized = cv2.resize(hm, (224, 224))
        threshold = np.percentile(hm_resized, 50)
        mask = torch.tensor(
            (hm_resized >= threshold).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        masked_input = input_tensor * mask
        self.model.eval()
        with torch.no_grad():
            out = self.model(masked_input)
            prob = F.softmax(out, dim=1)[0][class_idx].item()
        delta = prob - original_confidence
        return delta > 0, delta

    # ── Main analysis entrypoint ───────────────
    def analyze(self, image_bytes: bytes) -> dict:
        """
        Takes raw image bytes, returns a dict with all results
        and base64-encoded images.
        """
        # Open and ensure RGB
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ELA
        ela_img = self._generate_ela(original)

        # Prepare tensor from ELA
        input_tensor = self.transform(ela_img).unsqueeze(0).to(self.device)

        # Prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, preds = torch.max(probs, 1)

        class_idx = preds.item()
        prediction = self.CLASSES[class_idx]
        confidence_score = confidence.item()
        confidence_pct = confidence_score * 100.0

        # Grad-CAM heatmap
        target_layer = self.model.layer4[-1].conv3
        cam = GradCAM(self.model, target_layer)
        with torch.enable_grad():
            t = input_tensor.clone().detach().requires_grad_(True)
            heatmap = cam.generate_heatmap(t, class_idx)

        superimposed = self._superimpose_heatmap(original, heatmap)

        # Metrics
        avg_drop = self._metric_average_drop(
            input_tensor, confidence_score, class_idx
        )
        ic_flag, ic_delta = self._metric_increase_in_confidence(
            input_tensor, confidence_score, class_idx
        )

        # Encode images
        original_b64 = _pil_to_base64(original.resize((224, 224)))
        ela_b64 = _pil_to_base64(ela_img.resize((224, 224)))
        gradcam_b64 = _numpy_rgb_to_base64(superimposed)

        return {
            "prediction": prediction,
            "confidence": round(confidence_pct, 2),
            "avg_drop": round(avg_drop, 2),
            "ic_flag": ic_flag,
            "ic_delta": round(ic_delta * 100.0, 2),
            "original_image": original_b64,
            "ela_image": ela_b64,
            "gradcam_image": gradcam_b64,
        }
