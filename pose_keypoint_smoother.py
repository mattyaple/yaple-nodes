import copy
import math
import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Rendering helpers are imported lazily inside _render() so that the
# comfyui_controlnet_aux src/ path is guaranteed to be on sys.path by the
# time the node actually executes (all custom nodes finish loading first).
_draw_poses = None
_decode_json_as_poses = None


# ---------------------------------------------------------------------------
# Smoothing primitives — all operate on (N, 2) float arrays (x, y over time)
# ---------------------------------------------------------------------------

def _interpolate_gaps(coords, mask):
    """
    Linearly interpolate over frames where mask is False.
    coords : (N, 2) float array
    mask   : (N,) bool array — True where keypoint was detected
    Returns a new (N, 2) array with gaps filled.
    """
    n = len(coords)
    indices = np.arange(n)
    result = coords.astype(float).copy()
    if not mask.any():
        return result
    for dim in range(2):
        result[:, dim] = np.interp(indices, indices[mask], coords[mask, dim])
    return result


def _smooth_gaussian(coords, sigma):
    """Gaussian temporal smoothing (offline, symmetric)."""
    if SCIPY_AVAILABLE:
        return gaussian_filter1d(coords.astype(float), sigma=sigma, axis=0, mode='nearest')
    # Fallback: box filter approximation
    window = max(3, int(sigma * 2) | 1)
    return _smooth_moving_average(coords, window)


def _smooth_moving_average(coords, window_size):
    """Symmetric moving-average (offline)."""
    n = len(coords)
    half = window_size // 2
    result = np.empty_like(coords, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = coords[lo:hi].mean(axis=0)
    return result


def _smooth_exponential(coords, alpha):
    """
    Forward-backward exponential moving average (offline, zero phase-lag).
    alpha: smoothing factor — lower = more smoothing.
    """
    coords = coords.astype(float)
    n = len(coords)
    # Forward pass
    fwd = np.empty_like(coords)
    fwd[0] = coords[0]
    for i in range(1, n):
        fwd[i] = alpha * coords[i] + (1.0 - alpha) * fwd[i - 1]
    # Backward pass
    bwd = np.empty_like(coords)
    bwd[-1] = fwd[-1]
    for i in range(n - 2, -1, -1):
        bwd[i] = alpha * fwd[i] + (1.0 - alpha) * bwd[i + 1]
    return bwd


def _one_euro_alpha(cutoff, fps):
    """Compute the one-euro filter alpha from a cutoff frequency array or scalar."""
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=float))
    te = 1.0 / fps
    return 1.0 / (1.0 + tau / te)


def _smooth_one_euro(coords, fps, min_cutoff, beta, d_cutoff=1.0):
    """
    One Euro Filter — designed specifically for pose jitter reduction.
    Adapts the cutoff frequency to signal speed so slow motion is smoothed
    heavily while fast motion passes through with minimal lag.

    fps        : frames per second
    min_cutoff : minimum cutoff frequency (Hz) — lower = smoother for slow motion
    beta       : speed coefficient — higher = less lag on fast motion
    d_cutoff   : derivative low-pass cutoff (Hz)
    """
    coords = coords.astype(float)
    n, dims = coords.shape
    result = np.empty_like(coords)

    d_alpha = _one_euro_alpha(d_cutoff, fps)

    x_hat = coords[0].copy()
    dx_hat = np.zeros(dims)
    result[0] = x_hat

    for i in range(1, n):
        dx = (coords[i] - x_hat) * fps
        dx_hat = d_alpha * dx + (1.0 - d_alpha) * dx_hat

        # Per-dimension adaptive cutoff
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = _one_euro_alpha(cutoff, fps)

        x_hat = a * coords[i] + (1.0 - a) * x_hat
        result[i] = x_hat

    return result


# ---------------------------------------------------------------------------
# Core smoothing logic over a list of per-frame keypoint arrays
# ---------------------------------------------------------------------------

def _smooth_keypoints(kps_frames, method, sigma, window_size, alpha, min_cutoff, beta, fps):
    """
    Smooth keypoints across N frames.

    kps_frames : list of N items; each item is a flat list of floats
                 [x1, y1, c1, x2, y2, c2, ...] as produced by compress_keypoints(),
                 or None for frames where this person/part is absent.
    Returns    : list of N items in the same flat-float format, x/y smoothed.
                 Confidence values and undetected keypoints are preserved unchanged.
    """
    n_frames = len(kps_frames)
    if n_frames < 2:
        return kps_frames

    # Determine keypoint count from the first non-None, non-empty frame.
    # The flat list has length = n_kps * 3  (x, y, confidence per keypoint).
    raw_len = next((len(f) for f in kps_frames if f is not None and len(f) > 0), None)
    if raw_len is None or raw_len % 3 != 0:
        return kps_frames
    n_kps = raw_len // 3

    # Build (N, K, 2) coordinate array and (N, K) confidence array
    coords = np.zeros((n_frames, n_kps, 2), dtype=float)
    conf = np.zeros((n_frames, n_kps), dtype=float)

    for f, frame in enumerate(kps_frames):
        if frame is None:
            continue
        for k in range(n_kps):
            coords[f, k, 0] = frame[k * 3]        # x
            coords[f, k, 1] = frame[k * 3 + 1]    # y
            conf[f, k]       = frame[k * 3 + 2]   # confidence

    smoothed = coords.copy()

    for k in range(n_kps):
        mask = conf[:, k] > 0
        if mask.sum() < 2:
            continue

        # Fill detection gaps via linear interpolation before smoothing
        filled = _interpolate_gaps(coords[:, k, :], mask)

        if method == "gaussian":
            result = _smooth_gaussian(filled, sigma)
        elif method == "moving_average":
            result = _smooth_moving_average(filled, window_size)
        elif method == "exponential":
            result = _smooth_exponential(filled, alpha)
        elif method == "one_euro":
            result = _smooth_one_euro(filled, fps, min_cutoff, beta)
        else:
            result = filled

        # Only write back to frames that originally had this keypoint detected
        smoothed[mask, k] = result[mask]

    # Rebuild output as flat float lists matching the input format
    output = []
    for f in range(n_frames):
        if kps_frames[f] is None:
            output.append(None)
            continue
        flat = []
        for k in range(n_kps):
            orig_c = kps_frames[f][k * 3 + 2]
            if conf[f, k] > 0:
                flat += [float(smoothed[f, k, 0]), float(smoothed[f, k, 1]), float(orig_c)]
            else:
                # Preserve original undetected triplet (typically 0.0, 0.0, 0.0)
                flat += [float(kps_frames[f][k * 3]),
                         float(kps_frames[f][k * 3 + 1]),
                         float(orig_c)]
        output.append(flat)

    return output


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class PoseKeypointSmoother:
    """
    Smooths pose keypoints temporally to reduce jitter while preserving
    intentional motion.

    Four methods are available:
      gaussian      — symmetric Gaussian kernel (recommended for batch video)
      moving_average — simple sliding window average
      exponential   — forward-backward exponential moving average (zero phase)
      one_euro      — One Euro Filter, adapts to motion speed; best for reducing
                      jitter without adding lag on fast movements

    Only detected keypoints (confidence > 0) are updated. Gaps are filled by
    linear interpolation before smoothing, then undetected frames are restored
    to their original values so downstream rendering nodes still skip them.

    Coordinate format expected: POSE_KEYPOINT as produced by DWPose / OpenPose
    preprocessors in comfyui_controlnet_aux.
    """

    SMOOTH_METHODS = ["gaussian", "moving_average", "exponential", "one_euro"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "method": (cls.SMOOTH_METHODS, {"default": "gaussian"}),
                "sigma": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "[gaussian] Temporal std dev in frames. Higher = more smoothing.",
                }),
                "window_size": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 31,
                    "step": 2,
                    "tooltip": "[moving_average] Window size in frames (odd).",
                }),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "[exponential] Smoothing factor — lower = more smoothing.",
                }),
                "min_cutoff": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "[one_euro] Min cutoff frequency (Hz). Lower = smoother for slow motion.",
                }),
                "beta": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "[one_euro] Speed coefficient. Higher = less lag on fast movements.",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.5,
                    "tooltip": "[one_euro] Frames per second of the pose sequence.",
                }),
                "smooth_body": ("BOOLEAN", {"default": True}),
                "smooth_face": ("BOOLEAN", {"default": True}),
                "smooth_hands": ("BOOLEAN", {"default": False}),
                "render_body": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": True}),
                "render_hands": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("pose_keypoint", "images")
    FUNCTION = "smooth"
    CATEGORY = "infinitetalk/utils"

    def smooth(
        self,
        pose_keypoint,
        method,
        sigma,
        window_size,
        alpha,
        min_cutoff,
        beta,
        fps,
        smooth_body,
        smooth_face,
        smooth_hands,
        render_body,
        render_face,
        render_hands,
    ):
        n_frames = len(pose_keypoint)
        if n_frames < 2:
            print("[PoseKeypointSmoother] Less than 2 frames — nothing to smooth.")
            return (pose_keypoint,)

        # Deep copy so we never mutate the upstream node's output
        result = copy.deepcopy(pose_keypoint)

        is_ap10k = result[0].get("version") == "ap10k"

        if is_ap10k:
            n_animals = max(len(f.get("animals", [])) for f in result)
            for animal_idx in range(n_animals):
                animal_kps = [
                    f["animals"][animal_idx]
                    if f.get("animals") and animal_idx < len(f["animals"])
                    else None
                    for f in result
                ]
                smoothed = _smooth_keypoints(
                    animal_kps, method, sigma, window_size, alpha, min_cutoff, beta, fps
                )
                for f_idx, frame in enumerate(result):
                    if frame.get("animals") and animal_idx < len(frame["animals"]) and smoothed[f_idx] is not None:
                        frame["animals"][animal_idx] = smoothed[f_idx]

        else:
            # OpenPose / DWPose human format
            n_people = max(len(f.get("people", [])) for f in result)

            parts_to_smooth = []
            if smooth_body:
                parts_to_smooth.append("pose_keypoints_2d")
            if smooth_face:
                parts_to_smooth.append("face_keypoints_2d")
            if smooth_hands:
                parts_to_smooth += ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]

            for person_idx in range(n_people):
                for part_key in parts_to_smooth:
                    part_kps = [
                        f["people"][person_idx].get(part_key)
                        if f.get("people") and person_idx < len(f["people"])
                        else None
                        for f in result
                    ]
                    smoothed = _smooth_keypoints(
                        part_kps, method, sigma, window_size, alpha, min_cutoff, beta, fps
                    )
                    for f_idx, frame in enumerate(result):
                        if (
                            frame.get("people")
                            and person_idx < len(frame["people"])
                            and smoothed[f_idx] is not None
                        ):
                            frame["people"][person_idx][part_key] = smoothed[f_idx]

        parts_label = ", ".join(
            p for p, flag in [("body", smooth_body), ("face", smooth_face), ("hands", smooth_hands)] if flag
        )
        print(
            f"[PoseKeypointSmoother] {n_frames} frames | method={method} | "
            f"parts={parts_label or 'none'}"
        )

        images = self._render(result, render_body, render_face, render_hands)
        return (result, images)

    def _render(self, pose_keypoint, render_body, render_face, render_hands):
        global _draw_poses, _decode_json_as_poses
        if _draw_poses is None:
            try:
                from custom_controlnet_aux.dwpose import draw_poses, decode_json_as_poses
                _draw_poses = draw_poses
                _decode_json_as_poses = decode_json_as_poses
            except ImportError as e:
                raise RuntimeError(
                    "[PoseKeypointSmoother] Cannot render: comfyui_controlnet_aux dwpose module "
                    f"could not be imported ({e}). Make sure comfyui_controlnet_aux is installed."
                ) from e

        frames = []
        for frame_dict in pose_keypoint:
            poses, animals, H, W = _decode_json_as_poses(frame_dict)
            if poses:
                np_img = _draw_poses(poses, H, W,
                                     draw_body=render_body,
                                     draw_hand=render_hands,
                                     draw_face=render_face)
            else:
                # Animal / empty frame — blank canvas at canvas dimensions
                np_img = np.zeros((H, W, 3), dtype=np.uint8)
            frames.append(np_img.astype(np.float32) / 255.0)

        # Stack into (N, H, W, C) tensor that ComfyUI IMAGE nodes expect
        return torch.from_numpy(np.stack(frames, axis=0))


NODE_CLASS_MAPPINGS = {
    "PoseKeypointSmoother": PoseKeypointSmoother,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseKeypointSmoother": "Pose Keypoint Smoother",
}
