import torch
import torch.nn.functional as F


class VideoResize:
    """Resize video frames so width and height are both divisible by a given alignment (default 16)."""

    ROUNDING_MODES = ["nearest", "floor", "ceil"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alignment": ("INT", {"default": 16, "min": 2, "max": 64, "step": 2}),
                "rounding": (cls.ROUNDING_MODES, {"default": "nearest"}),
                "interpolation": (["bilinear", "bicubic", "nearest", "area"], {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "yaple/video"

    def resize(self, images: torch.Tensor, alignment: int, rounding: str, interpolation: str):
        # images: [B, H, W, C] float32 in [0, 1]
        B, H, W, C = images.shape

        target_h = self._align(H, alignment, rounding)
        target_w = self._align(W, alignment, rounding)

        if target_h == H and target_w == W:
            return (images, target_w, target_h)

        # F.interpolate expects [B, C, H, W]
        x = images.permute(0, 3, 1, 2)

        mode = interpolation
        align_corners = False if mode in ("bilinear", "bicubic") else None

        if align_corners is None:
            x = F.interpolate(x, size=(target_h, target_w), mode=mode)
        else:
            x = F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=align_corners)

        x = x.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        print(f"[VideoResize] {W}x{H} → {target_w}x{target_h} (alignment={alignment}, rounding={rounding})")
        return (x, target_w, target_h)

    @staticmethod
    def _align(value: int, alignment: int, rounding: str) -> int:
        if rounding == "floor":
            return (value // alignment) * alignment
        elif rounding == "ceil":
            return ((value + alignment - 1) // alignment) * alignment
        else:  # nearest
            return round(value / alignment) * alignment


NODE_CLASS_MAPPINGS = {"VideoResize": VideoResize}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoResize": "Video Resize (Align)"}
