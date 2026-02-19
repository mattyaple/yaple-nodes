# yaple-nodes

Custom ComfyUI nodes for InfiniteTalk / LipSync workflows.

## Nodes

### Optimal Frame Window Calculator
`infinitetalk/utils`

Calculates the optimal `frame_window_size` for InfiniteTalk generation given an audio clip, target FPS, and motion frame value. Minimises the number of generation windows (and therefore render time) while keeping padding to a minimum.

**Inputs:** `audio`, `fps`, `motion_frame`, `target_windows`
**Outputs:** `frame_window_size`, `total_frames`, `num_windows`, `padding_frames`

---

### Video Frame Padder
`infinitetalk/utils`

Pads a video frame sequence by repeating the last frame. Connect `padding_frames` from the Optimal Frame Window Calculator to keep pose/reference video length in sync with the padded audio embedding.

**Inputs:** `images`, `padding_frames`
**Outputs:** `images`

---

### Pose Keypoint Smoother
`infinitetalk/utils`

Smooths DWPose / OpenPose keypoints temporally across a video sequence to reduce jitter and smooth out frame-window boundary jumps. Outputs both the smoothed `POSE_KEYPOINT` data and a rendered `IMAGE` sequence.

**Inputs:** `pose_keypoint`, `method`, `sigma`, `window_size`, `alpha`, `min_cutoff`, `beta`, `fps`, `smooth_body`, `smooth_face`, `smooth_hands`, `render_body`, `render_face`, `render_hands`
**Outputs:** `pose_keypoint`, `images`

**Methods:**
| Method | Key Parameter | Best For |
|---|---|---|
| `gaussian` | `sigma` (frames) | Frame-window boundary jumps; general batch smoothing |
| `moving_average` | `window_size` (frames) | Simple uniform smoothing |
| `exponential` | `alpha` (0→1) | Forward-backward pass, zero phase lag |
| `one_euro` | `min_cutoff` + `beta` + `fps` | High-frequency jitter while preserving fast motion |

**Tip for frame-window boundary jumps:** use `gaussian` with `sigma` 4–6.

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) (required by Pose Keypoint Smoother for rendering)
