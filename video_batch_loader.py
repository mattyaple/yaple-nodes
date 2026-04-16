import os
import re
import sys

# Add VHS directory to sys.path so we can import load_video
_VHS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'comfyui-videohelpersuite')
if os.path.isdir(_VHS_DIR) and _VHS_DIR not in sys.path:
    sys.path.insert(0, _VHS_DIR)

try:
    from videohelpersuite.load_video_nodes import load_video, video_extensions, get_load_formats
except ImportError as e:
    raise ImportError(f"VideoBatchLoader requires comfyui-videohelpersuite to be installed. Error: {e}")

_VIDEO_EXTENSIONS = set(video_extensions)


class VideoBatchLoader:
    """
    Loads a single video from a directory, selected by index.
    Optionally auto-increments the index on each run to cycle through all videos.
    """

    # Per-node-instance state: {unique_id: {'stored_index': int, 'last_widget_index': int}}
    _state = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1,
                                  "tooltip": "Which video to load (0-based). Wraps around total file count."}),
                "auto_increment": ("BOOLEAN", {"default": True,
                                               "tooltip": "Increment index automatically on each run"}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1,
                                         "tooltip": "0 = use source FPS"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                          "tooltip": "0 = use source width"}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                           "tooltip": "0 = use source height"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1,
                                            "tooltip": "0 = load all frames"}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "filename")
    FUNCTION = "load_batch_video"
    CATEGORY = "yaple/video"

    @classmethod
    def IS_CHANGED(cls, auto_increment, **kwargs):
        # Always re-execute when auto-incrementing so each queue run loads the next video
        if auto_increment:
            return float("NaN")
        return False

    def _get_video_files(self, directory):
        files = []
        for f in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, f)):
                ext = f.rsplit('.', 1)[-1].lower() if '.' in f else ''
                if ext in _VIDEO_EXTENSIONS:
                    files.append(f)

        def natural_key(name):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', name)]

        return sorted(files, key=natural_key)

    def load_batch_video(self, directory, index, auto_increment,
                         force_rate, custom_width, custom_height,
                         frame_load_cap, skip_first_frames, select_every_nth,
                         meta_batch=None, vae=None, format='None', unique_id=None):
        directory = directory.strip().strip('"').strip("'")
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        files = self._get_video_files(directory)
        if not files:
            raise ValueError(f"No video files found in: {directory}")

        total = len(files)
        state = self.__class__._state.get(unique_id)

        if state is None or state['last_widget_index'] != index:
            # First run, or user manually changed the index widget — honour widget value
            effective_index = index % total
        else:
            # Continuing an auto-increment sequence
            effective_index = state['stored_index'] % total

        next_index = (effective_index + 1) % total if auto_increment else effective_index
        self.__class__._state[unique_id] = {
            'stored_index': next_index,
            'last_widget_index': index,
        }

        filename = files[effective_index]
        video_path = os.path.join(directory, filename)

        print(f"[VideoBatchLoader] [{effective_index + 1}/{total}] {filename}")

        result = load_video(
            video=video_path,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            meta_batch=meta_batch,
            vae=vae,
            format=format,
            unique_id=unique_id,
        )

        return result + (filename,)


NODE_CLASS_MAPPINGS = {
    "VideoBatchLoader": VideoBatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoBatchLoader": "Video Batch Loader",
}
