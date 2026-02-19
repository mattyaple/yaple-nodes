import torch


class VideoFramePadder:
    """
    Pads a sequence of video frames by repeating the last frame.

    Padding amount = padding_frames + 3, where the extra 3 matches the
    hardcoded safety buffer InfiniteTalk/MultiTalk adds when padding audio
    (see multitalk_loop.py: miss_length = audio_end_idx - len(audio_embedding) + 3).

    Connect padding_frames from OptimalFrameWindowCalculator to keep the
    pose/reference video in sync with the padded audio embedding length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "padding_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Connect to padding_frames output of OptimalFrameWindowCalculator"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "pad"
    CATEGORY = "infinitetalk/utils"

    def pad(self, images, padding_frames):
        # images shape: (frames, height, width, channels)
        total_padding = padding_frames + 3  # +3 matches InfiniteTalk's safety buffer

        if total_padding <= 0:
            return (images,)

        last_frame = images[-1:] # shape: (1, H, W, C)
        padding = last_frame.repeat(total_padding, 1, 1, 1)
        padded = torch.cat([images, padding], dim=0)

        print(f"[VideoFramePadder] Input frames: {images.shape[0]} | "
              f"Padding: {padding_frames} + 3 safety = {total_padding} | "
              f"Output frames: {padded.shape[0]}")

        return (padded,)


NODE_CLASS_MAPPINGS = {
    "VideoFramePadder": VideoFramePadder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFramePadder": "Video Frame Padder",
}
