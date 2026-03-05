import torch
import numpy as np


class AlphaComposite:
    """
    Composites an image with alpha channel over a solid color background.
    If the input image has no alpha channel, it passes through unchanged.
    Output is always RGB (no alpha).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_red": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Red channel (0-255)"
                }),
                "background_green": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Green channel (0-255)"
                }),
                "background_blue": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Blue channel (0-255)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask/alpha channel (swap opaque and transparent areas)"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional alpha mask. If not provided, will use alpha channel from image if present."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composited", "background")
    FUNCTION = "composite"
    CATEGORY = "image/postprocessing"

    def composite(self, image, background_red, background_green, background_blue, invert_mask, mask=None):
        """
        Composite image over solid color background.

        Args:
            image: Input image tensor (B, H, W, C) where C is 3 (RGB) or 4 (RGBA)
            background_red: Red component (0-255)
            background_green: Green component (0-255)
            background_blue: Blue component (0-255)
            mask: Optional mask tensor (B, H, W) or (H, W)

        Returns:
            Tuple of (composited image, background layer) - both RGB tensors (B, H, W, 3)
        """
        # Convert to float for processing
        img = image.clone()
        batch_size, height, width, channels = img.shape

        # Normalize RGB values to [0, 1] range
        bg_color = torch.tensor(
            [background_red / 255.0, background_green / 255.0, background_blue / 255.0],
            dtype=img.dtype,
            device=img.device
        )

        # Create background layer with same shape as input image
        background = bg_color.view(1, 1, 1, 3).expand(batch_size, height, width, 3)

        # Determine RGB and alpha
        if mask is not None:
            # Use provided mask as alpha
            print("[AlphaComposite] Using provided mask as alpha channel")
            if channels == 4:
                rgb = img[..., :3]
            else:
                rgb = img

            # Handle mask dimensions
            if mask.dim() == 2:
                # (H, W) -> (1, H, W, 1)
                alpha = mask.unsqueeze(0).unsqueeze(-1)
            elif mask.dim() == 3:
                # (B, H, W) -> (B, H, W, 1)
                alpha = mask.unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected mask dimensions: {mask.shape}")

        elif channels == 4:
            # Use alpha channel from image
            print("[AlphaComposite] Using alpha channel from image")
            rgb = img[..., :3]  # (B, H, W, 3)
            alpha = img[..., 3:4]  # (B, H, W, 1)

        else:
            # No alpha available - pass through unchanged, still return background
            print("[AlphaComposite] No alpha channel or mask provided, passing through unchanged")
            return (img, background)

        # Debug info
        print(f"[AlphaComposite] Image shape: {image.shape}")
        print(f"[AlphaComposite] RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"[AlphaComposite] Alpha range BEFORE invert: [{alpha.min():.3f}, {alpha.max():.3f}]")
        print(f"[AlphaComposite] Alpha shape: {alpha.shape}")

        # Invert mask if requested
        if invert_mask:
            print("[AlphaComposite] Inverting mask/alpha")
            alpha = 1.0 - alpha
            print(f"[AlphaComposite] Alpha range AFTER invert: [{alpha.min():.3f}, {alpha.max():.3f}]")

        # Alpha compositing: result = foreground * alpha + background * (1 - alpha)
        composited = rgb * alpha + background * (1 - alpha)
        composited = composited.clamp(0, 1)

        print(f"[AlphaComposite] Composited {batch_size} image(s) of {height}x{width} "
              f"over RGB({background_red}, {background_green}, {background_blue})")

        return (composited, background)


NODE_CLASS_MAPPINGS = {
    "AlphaComposite": AlphaComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaComposite": "Alpha Composite",
}
