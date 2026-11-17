"""Optional low-level YOLO11s adapter for Ultralytics 8.3.58.

Provides forward and detection-loss calls, not a full Trainer/Validator.
"""
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from .fusion import TextureP5Fusion


class TPGYOLO(DetectionModel):
    def __init__(self, nc: int = 12, verbose: bool = False):
        if ultralytics.__version__ != "8.3.58":
            raise RuntimeError("This adapter targets ultralytics==8.3.58")
        # DetectionModel performs a dummy forward during stride initialization.
        super().__init__(cfg="yolo11s.yaml", ch=3, nc=nc, verbose=verbose)
        self.args = get_cfg()
        self.args.box, self.args.cls, self.args.dfl = 7.5, 0.5, 1.5
        channels = self.model[-1].cv2[-1][0].conv.in_channels
        self.texture_fusion = TextureP5Fusion(channels)
        self._active_texture = None
        self.model[-1].register_forward_pre_hook(self._before_detect)

    def _before_detect(self, module, args):
        return (self.texture_fusion(args[0], self._active_texture),)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None, *, z_t=None):
        if not hasattr(self, "texture_fusion"):
            return super().predict(x, profile, visualize, augment, embed)
        if z_t is None:
            raise ValueError("Pass the image-aligned texture prior as z_t=[B,256]")
        if augment or embed is not None:
            raise ValueError("This minimal adapter supports detection without TTA or embeddings")
        self._active_texture = z_t
        try:
            return super().predict(x, profile, visualize, False, None)
        finally:
            self._active_texture = None

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            # Section 3.7: offline restoration outputs are constants during training.
            preds = self.predict(batch["img"].detach(), z_t=batch["z_t"].detach())
        return self.criterion(preds, batch)
