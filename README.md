# yaple-nodes

Custom ComfyUI nodes for InfiniteTalk / LipSync workflows and general utilities.

## Nodes

---

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

---

### Video Seam RIFE
`infinitetalk/utils`

Smooths InfiniteTalk frame-window boundary jumps using RIFE motion-compensated interpolation without changing the frame count (preserving audio sync). For each seam, frames immediately surrounding the boundary are replaced with RIFE-interpolated frames generated between two anchor frames lying outside the affected region.

> **Note:** Built and registered but not currently used in the active workflow.

**Inputs:** `images`, `frame_window_size`, `seam_motion_frame`, `ckpt_name`, `replace_radius`, `scale_factor`, `fast_mode`, `ensemble`
**Outputs:** `images`

**Requirements:** [comfyui-frame-interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) installed as a sibling custom node.

---

### Auto Switch
`yaple/utils`

Smart A/B switch that auto-selects the active input — no manual toggling needed when one input is muted or disconnected. Accepts any data type.

**Priority:**
1. If only one of A/B is connected (or the other's source node is muted/bypassed), automatically routes the active input.
2. If both are active, the `select_b` toggle decides.
3. If neither is active, returns `None`.

**Inputs:** `select_b` (toggle), `a` (optional, any type), `b` (optional, any type)
**Outputs:** `output` (any type)

---

### Alpha Composite
`image/postprocessing`

Composites an RGBA image (or IMAGE + separate MASK) over a solid colour background, outputting an RGB result with no alpha channel. If the input has no alpha and no mask is provided, the image passes through unchanged.

**Inputs:** `image`, `background_red`, `background_green`, `background_blue`, `invert_mask`, `mask` (optional)
**Outputs:** `composited` (IMAGE), `background` (IMAGE)

---

### LLM Prompt Builder
`llm/prompting`

Builds prompts for LLM API nodes. Combines a user prompt with a system prompt loaded from a `.md` or `.txt` file in the `ComfyUI/LLM_input/` directory. Optionally prepends a creative enhancement instruction to the prompt.

**Inputs:** `prompt`, `system_prompt_file`, `creatively_enhance`
**Outputs:** `prompt` (STRING), `system_prompt` (STRING)

---

### Video Resize (Align)
`yaple/video`

Resizes a video frame batch so that width and height are both divisible by a given alignment value (default 16). Useful for ensuring model-compatible dimensions before encoding.

**Inputs:** `images`, `alignment` (default 16), `rounding` (`nearest`/`floor`/`ceil`), `interpolation` (`bilinear`/`bicubic`/`nearest`/`area`)
**Outputs:** `images`, `width` (INT), `height` (INT)

---

### Qwen Camera Prompt
`yaple/camera`

Interactive 3D camera-angle selector for Qwen multi-angle image editing LoRAs. A drag-handle viewport in the node UI lets you set azimuth, elevation, and distance, then outputs the corresponding prompt string for the selected LoRA format.

**LoRA formats (toggled in the node UI):**
- **fal** — `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`: outputs `<sks> {azimuth} {elevation} {distance}`
- **dx8152** — `dx8152/Qwen-Edit-2509-Multiple-angles`: outputs bilingual natural-language camera commands (also used by the linoyts/Qwen-Image-Edit-Angles HuggingFace space)

**Pose space:** 8 azimuths (0°–315° in 45° steps) × 4 elevations (−30°, 0°, 30°, 60°) × 4 distances (close-up, forward, medium, wide) = 128 poses.

**Inputs:** `camera_state` (JSON, written by the JS viewport UI)
**Outputs:** `prompt` (STRING)

---

### Nano Banana Batch
`api node/image/Gemini`

Generates a batch of images with **Nano Banana** (Google Gemini Image) in a single workflow run. All `batch_size` requests are fired concurrently via `asyncio.gather`, so total wall-clock time ≈ one image. Cost = `batch_size` × per-image price.

**Inputs:** `prompt`, `batch_size` (1–8), `model`, `seed`, `images` (optional), `files` (optional), `aspect_ratio`, `response_modalities`, `system_prompt`
**Outputs:** `IMAGE`, `STRING` (generated text)

---

### Nano Banana 2 Batch
`api node/image/Gemini`

Same concurrent-batch pattern as Nano Banana Batch but targets **Nano Banana 2** (Gemini 3 Pro / 3.1 Flash Image). Adds `resolution` (1K / 2K / 4K) and `thinking_level` controls.

**Inputs:** `prompt`, `batch_size` (1–8), `model`, `seed`, `aspect_ratio`, `resolution`, `response_modalities`, `thinking_level`, `images` (optional), `files` (optional), `system_prompt`
**Outputs:** `IMAGE`, `STRING` (generated text)

---

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) (required by Pose Keypoint Smoother for rendering)
- [comfyui-frame-interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (required by Video Seam RIFE only)
