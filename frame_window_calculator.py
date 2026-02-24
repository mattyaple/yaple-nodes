import math


class OptimalFrameWindowCalculator:
    """
    Calculates the optimal frame window size for InfiniteTalk generation.

    Valid frame window sizes must satisfy: size % 4 == 1 (i.e. 81, 85, 89, ..., 181)
    The stride between windows is: size - motion_frame
    (matching the audio_start_idx advance in multitalk_loop.py)

    Strategy: minimize number of windows first, then minimize padding.
    If target_windows > 0, targets that exact number of windows with no size upper limit.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 120,
                    "tooltip": "Frames per second of the output video"
                }),
                "motion_frame": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Must match the motion_frame value on your InfiniteTalk sampler node"
                }),
                "target_windows": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "tooltip": "0 = auto (minimize windows, respect max_frame_window_size). >0 = target exactly this many windows with no size upper limit (experimental, may OOM)"
                }),
                "max_frame_window_size": ("INT", {
                    "default": 181,
                    "min": 81,
                    "max": 500,
                    "step": 4,
                    "tooltip": "Upper limit for frame window size in auto mode (target_windows=0). Default 181 is tuned for H100 (80GB VRAM). Lower this on GPUs with less VRAM — 81 is a safe starting point for less powerful GPUs. Larger windows use significantly more VRAM."
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("frame_window_size", "total_frames", "num_windows", "padding_frames")
    FUNCTION = "calculate"
    CATEGORY = "infinitetalk/utils"

    def calculate(self, audio, fps, motion_frame, target_windows, max_frame_window_size):
        waveform = audio["waveform"]  # shape: (batch, channels, samples)
        sample_rate = audio["sample_rate"]
        num_samples = waveform.shape[-1]
        duration = num_samples / sample_rate

        # Use int() to match InfiniteTalk's own frame counting: int(audio_duration * fps)
        total_frames = int(duration * fps)

        if target_windows == 0:
            # Normal mode: cap at max_frame_window_size
            valid_sizes = list(range(81, max_frame_window_size + 1, 4))
        else:
            # Experimental mode: no upper limit — generate sizes up to the minimum
            # needed to fit total_frames in a single window (size >= total_frames)
            upper = total_frames
            while upper % 4 != 1:
                upper += 1
            upper = max(upper, 81)  # ensure at least one candidate exists
            valid_sizes = list(range(81, upper + 1, 4))

        candidates = []
        for size in valid_sizes:
            stride = size - motion_frame
            if stride <= 0:
                continue
            if total_frames <= size:
                n = 1
            else:
                n = math.ceil((total_frames - size) / stride) + 1
            coverage = size + (n - 1) * stride
            padding = coverage - total_frames

            if target_windows == 0 or n == target_windows:
                candidates.append((n, padding, size))

        if not candidates:
            print(f"[FrameWindowCalc] WARNING: Cannot achieve target_windows={target_windows} "
                  f"for {total_frames} frames with motion_frame={motion_frame}. Falling back to auto.")
            for size in valid_sizes:
                stride = size - motion_frame
                if stride <= 0:
                    continue
                n = 1 if total_frames <= size else math.ceil((total_frames - size) / stride) + 1
                coverage = size + (n - 1) * stride
                padding = coverage - total_frames
                candidates.append((n, padding, size))

        if target_windows == 0:
            # Minimize windows first, then minimize padding
            candidates.sort(key=lambda x: (x[0], x[1]))
        else:
            # Window count is fixed — just minimize padding
            candidates.sort(key=lambda x: x[1])

        best_n, best_padding, best_size = candidates[0]

        print(f"[FrameWindowCalc] Audio: {duration:.3f}s @ {fps}fps = {total_frames} frames")
        print(f"[FrameWindowCalc] motion_frame={motion_frame}, stride={best_size - motion_frame}")
        if target_windows > 0:
            print(f"[FrameWindowCalc] Target windows: {target_windows} (experimental)")
        else:
            print(f"[FrameWindowCalc] Max frame window size: {max_frame_window_size}")
        print(f"[FrameWindowCalc] Best window size: {best_size} | Windows: {best_n} | Padding: {best_padding} frames")

        return (best_size, total_frames, best_n, best_padding)


NODE_CLASS_MAPPINGS = {
    "OptimalFrameWindowCalculator": OptimalFrameWindowCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OptimalFrameWindowCalculator": "Optimal Frame Window Calculator",
}
