import os
import sys
import torch

# ---------------------------------------------------------------------------
# Lazy RIFE imports — comfyui-frame-interpolation must be installed alongside
# ---------------------------------------------------------------------------
_rife_pkg_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "comfyui-frame-interpolation")
)
_IFNet = None
_load_file_from_github_release = None


def _ensure_rife():
    global _IFNet, _load_file_from_github_release
    if _IFNet is not None:
        return
    if _rife_pkg_path not in sys.path:
        sys.path.insert(0, _rife_pkg_path)
    try:
        from vfi_models.rife.rife_arch import IFNet
        from vfi_utils import load_file_from_github_release
        _IFNet = IFNet
        _load_file_from_github_release = load_file_from_github_release
    except ImportError as e:
        raise RuntimeError(
            "[VideoSeamRIFE] Cannot import RIFE — make sure comfyui-frame-interpolation "
            f"is installed as a sibling custom node ({_rife_pkg_path}). Error: {e}"
        ) from e


# Maps checkpoint filename → IFNet arch version string
CKPT_VER = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0",
    "rife42.pth": "4.2",
    "rife43.pth": "4.3",
    "rife44.pth": "4.3",
    "rife45.pth": "4.5",
    "rife46.pth": "4.6",
    "rife47.pth": "4.7",
    "rife48.pth": "4.7",
    "rife49.pth": "4.7",
}


class VideoSeamRIFE:
    """
    Smooths InfiniteTalk frame-window boundary jumps using RIFE motion-compensated
    interpolation, without changing the frame count (preserving audio sync).

    For each seam between window N and window N+1, the frames immediately surrounding
    the boundary are replaced with RIFE-interpolated frames generated between two
    anchor frames that lie outside the affected region.

    With replace_radius=1 and a seam at frame 144→145:
      - Anchor frames: 143 and 146
      - RIFE generates 2 interpolated frames (t=1/3 and t=2/3)
      - Frames 144 and 145 are replaced with those 2 RIFE frames
      - Total frame count unchanged

    Increasing replace_radius widens the zone — more frames replaced, smoother
    transition but more RIFE warping visible on either side.

    Requires comfyui-frame-interpolation to be installed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_window_size": ("INT", {
                    "default": 145, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Connect to frame_window_size output of Optimal Frame Window Calculator.",
                }),
                "seam_motion_frame": ("INT", {
                    "default": 16, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Match the motion_frame value on your InfiniteTalk sampler. Used to compute window stride.",
                }),
                "ckpt_name": (sorted(CKPT_VER.keys()), {"default": "rife47.pth"}),
                "replace_radius": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": (
                        "Number of frames replaced on each side of the seam boundary. "
                        "radius=1 replaces 2 frames (one from each window); "
                        "radius=2 replaces 4 frames. Larger = smoother but more RIFE warping."
                    ),
                }),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "infinitetalk/utils"

    def apply(
        self,
        images,
        frame_window_size,
        seam_motion_frame,
        ckpt_name,
        replace_radius,
        scale_factor,
        fast_mode,
        ensemble,
    ):
        import comfy.model_management as model_management

        n_frames = images.shape[0]
        stride = frame_window_size - seam_motion_frame

        if stride <= 0:
            print("[VideoSeamRIFE] stride <= 0 — nothing to do.")
            return (images,)

        # Collect seam left-frame indices (last frame of each window except the last)
        # The hard boundary is between frame seam_l and seam_l+1.
        seam_lefts = []
        s = frame_window_size - 1  # 0-indexed last frame of window 1
        while s + 1 < n_frames:
            seam_lefts.append(s)
            s += stride

        if not seam_lefts:
            print("[VideoSeamRIFE] No seams found within the video length.")
            return (images,)

        print(
            f"[VideoSeamRIFE] {len(seam_lefts)} seam(s) at: "
            f"{[f'{sl}→{sl+1}' for sl in seam_lefts]}"
        )

        # Load RIFE
        _ensure_rife()
        arch_ver = CKPT_VER[ckpt_name]
        model_path = _load_file_from_github_release("rife", ckpt_name)
        rife = _IFNet(arch_ver=arch_ver)
        rife.load_state_dict(torch.load(model_path, map_location="cpu"))
        rife.eval()
        device = model_management.get_torch_device()
        rife = rife.to(device)

        scale_list = [8 / scale_factor, 4 / scale_factor,
                      2 / scale_factor, 1 / scale_factor]

        result = images.clone()

        def to_rife(idx):
            """(H,W,C) float32 → (1,C,H,W) on device."""
            return images[idx].permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)

        def from_rife(t):
            """(1,C,H,W) → (H,W,C) float32 on CPU."""
            return t.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0)

        for seam_l in seam_lefts:
            seam_r = seam_l + 1

            # Anchor frames sit just outside the replacement zone
            anchor_l_idx = max(0, seam_l - replace_radius)
            anchor_r_idx = min(n_frames - 1, seam_r + replace_radius)

            # Frames to replace: everything strictly between the two anchors
            replace_start = anchor_l_idx + 1
            replace_end   = anchor_r_idx        # exclusive
            n_replace      = replace_end - replace_start

            if n_replace <= 0:
                print(f"[VideoSeamRIFE] Seam {seam_l}→{seam_r}: "
                      "not enough frames to replace — skipping.")
                continue

            f0 = to_rife(anchor_l_idx)
            f1 = to_rife(anchor_r_idx)

            # multiplier = n_replace + 1 gives exactly n_replace middle frames
            # at timesteps 1/M, 2/M, ..., n_replace/M  where M = n_replace+1
            M = n_replace + 1
            with torch.no_grad():
                for i in range(n_replace):
                    t = (i + 1) / M
                    interp = rife(f0, f1, t, scale_list, fast_mode, ensemble)
                    result[replace_start + i] = from_rife(interp)

            print(
                f"[VideoSeamRIFE] Seam {seam_l}→{seam_r}: "
                f"replaced frames {replace_start}–{replace_end - 1} "
                f"(anchors: {anchor_l_idx} and {anchor_r_idx})"
            )

        del rife
        torch.cuda.empty_cache()
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VideoSeamRIFE": VideoSeamRIFE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSeamRIFE": "Video Seam RIFE",
}
