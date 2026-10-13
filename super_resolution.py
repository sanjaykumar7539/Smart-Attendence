"""Optional Real-ESRGAN-style ONNX super-resolution fallback.
No model is bundled. If the configured ONNX file is absent, this module is disabled.
"""
from pathlib import Path
import cv2
import numpy as np
from loguru import logger


class RealEsrganFaceEnhancer:
    """Use the ai-forever x2 checkpoint for selected recognition crops."""

    def __init__(self, model_path, device="cpu"):
        import sys
        import types
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download

        import torch
        from RealESRGAN import RealESRGAN

        self.device = torch.device(device)
        self.model = RealESRGAN(self.device, scale=2)
        self.model.load_weights(str(model_path), download=False)
        self.calls = 0
        logger.success(f"Real-ESRGAN face enhancer ready: {Path(model_path).name}")

    def enhance(self, crop: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        output = self.model.predict(Image.fromarray(rgb))
        self.calls += 1
        return cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)


class SuperResolutionEngine:
    def __init__(self, cfg):
        self.cfg=cfg; self.net=None
        path=Path(cfg.sr_model_path)
        if cfg.enable_super_resolution and path.exists():
            try:
                self.net=cv2.dnn.readNetFromONNX(str(path))
                logger.success(f"Super-resolution ready: {path.name}")
            except Exception as e:
                logger.warning(f"SR disabled; model load failed: {e}")
        elif cfg.enable_super_resolution:
            logger.warning(f"SR model not found: {path} — adaptive pipeline will continue without SR")

    @property
    def available(self): return self.net is not None

    def enhance(self, crop: np.ndarray) -> np.ndarray:
        if not self.available or crop is None or crop.size == 0: return crop
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        blob=np.transpose(rgb,(2,0,1))[None]
        try:
            self.net.setInput(blob)
            out=self.net.forward()[0]
            out=np.clip(np.transpose(out,(1,2,0)),0,1)
            return cv2.cvtColor((out*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.debug(f"SR inference failed, original crop used: {e}")
            return crop
