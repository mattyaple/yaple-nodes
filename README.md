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

**Inputs:** `pose_keypoint`, `method`, `sigma`, `window_size`, `alpha`, `min_cutoff`, `beta`, `fps`, `smooth_body`, `smooth_face`, `smooth_hands`, `render_body`, `render_face`, `render_hands`, `seam_stabilize`, `seam_frame_window_size`, `seam_motion_frame`, `stabilize_sigma`, `stabilize_radius`, `stabilize_strength`
**Outputs:** `pose_keypoint`, `images`

**Methods:**
| Method | Key Parameter | Best For |
|---|---|---|
| `gaussian` | `sigma` (frames) | Frame-window boundary jumps; general batch smoothing |
| `moving_average` | `window_size` (frames) | Simple uniform smoothing |
| `exponential` | `alpha` (0→1) | Forward-backward pass, zero phase lag |
| `one_euro` | `min_cutoff` + `beta` + `fps` | High-frequency jitter while preserving fast motion |

**Tip for frame-window boundary jumps:** use `gaussian` with `sigma` 4–6.

**Seam Stabilization** (`seam_stabilize` toggle):

Applies extra smoothing specifically around InfiniteTalk's frame-window stitch points. The weight follows a Gaussian bell curve centred on each seam so the pose eases naturally into a more stable position approaching the boundary and eases back out the other side — no sudden freeze or unfreeze.

| Parameter | Default | Description |
|---|---|---|
| `seam_frame_window_size` | 145 | Connect to `frame_window_size` output of Optimal Frame Window Calculator |
| `seam_motion_frame` | 16 | Match the `motion_frame` value on your InfiniteTalk sampler node |
| `stabilize_sigma` | 8.0 | Gaussian sigma for heavy smoothing at the seam. Higher = pose held more rigidly at boundary |
| `stabilize_radius` | 8.0 | Std dev (frames) of the bell-curve blend weight. Controls how wide the ease-in/out region is |
| `stabilize_strength` | 1.0 | Peak blend weight at seam centre. 1.0 = fully stabilized; 0.5 = half normal, half stabilized |

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) (required by Pose Keypoint Smoother for rendering)
