# yaple-nodes — Project Context for Claude Code

## What This Repo Is

Custom ComfyUI nodes for a WAN 2.1 / InfiniteTalk lipsync pipeline:

- **OptimalFrameWindowCalculator** — calculates frame window size and total frames from audio length, FPS, and motion frames
- **VideoFramePadder** — pads video frames to fill the required frame count
- **PoseKeypointSmoother** — smooths DWPose keypoints across frames with multiple filter options; includes seam stabilization for InfiniteTalk window boundaries
- **VideoSeamRIFE** — replaces seam frames with RIFE-interpolated frames (on hold, not currently used)

## Environment

- ComfyUI root: `/home/ubuntu/ComfyUI`
- This repo: `/home/ubuntu/ComfyUI/custom_nodes/yaple-nodes`
- Key custom node dependency: `/home/ubuntu/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper` (WAN models, InfiniteTalk/MultiTalk, UniAnimate)
- Workflows live at: `/home/ubuntu/ComfyUI/user/default/workflows/Video Workflows/`
- GitHub remote: `git@github.com:mattyaple/yaple-nodes.git` (SSH key at `~/.ssh/github_yaple`)

## Installed Models (relevant)

All under `/home/ubuntu/ComfyUI/models/diffusion_models/Wan2.1/`:
- `wan2.1_i2v_720p_14B_fp16.safetensors` — primary InfiniteTalk base model
- `wan2.1_t2v_14B_fp16.safetensors` — T2V base (available but not currently used)
- `wan2.1_vace_14B_fp16.safetensors` — VACE module (available but see VACE note below)
- WAN 2.2 Animate model also installed

---

## Active Work: LipSync V2V InfiniteTalk Workflow

### The Pipeline (2-stage)

**Stage 1 — Motion Pass (WAN 2.2 Animate)**
- Generates a video with good body motion and mouth shapes
- Uses UniAnimate for pose-driven generation
- Output is used as the reference for Stage 2

**Stage 2 — InfiniteTalk Pass (WAN 2.1 I2V + InfiniteTalk)**
- Audio-driven lipsync baked in via InfiniteTalk/MultiTalk sampling loop
- Uses Stage 1 output as: init latent (WanVideoEncode), clip vision reference (first frame), and UniAnimate pose reference (DWPose skeleton extraction)
- Output is the final lipsync video

### Active Workflow File

`LipSync_Stage2_2.1-V2V_InfiniteTalk.json`

Key node IDs in the workflow:
- `122` WanVideoModelLoader — loads I2V model + UniAnimate LoRA chain
- `128` WanVideoSampler — main sampler, runs InfiniteTalk loop
- `192` WanVideoImageToVideoMultiTalk — constructs image_embeds for InfiniteTalk
- `194` MultiTalkWav2VecEmbeds — processes audio → multitalk_embeds
- `398` OptimalFrameWindowCalculator
- `440` VideoFramePadder
- `443` DWPreprocessor — pose extraction (detect_face=OFF — see below)
- `444` PoseKeypointSmoother — seam stabilization enabled
- `407` WanVideoUniAnimatePoseInput — UniAnimate pose conditioning

### Current Parameter State (Stage 2 / InfiniteTalk pass)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `wan2.1_i2v_720p_14B_fp16` | **About to test swapping to WAN 2.2 Animate** |
| shift | 6 | Was tuned for WAN 2.1; see shift notes below |
| audio_scale | 2.0 | Multiplies audio embedding tensor |
| audio_cfg_scale | 1.0 | Keep at 1.0 unless willing to double render time |
| UniAnimate strength | 0.7 | |
| motion_frames | 25 | Frame window overlap |
| LoRA | Lightx2v rank64 @ 0.6 + UniAnimate LoRA | |
| detect_face (DWPose) | OFF | Enabling throws off lipsync — keep OFF |
| seam_stabilize | ON | seam_frame_window_size and seam_motion_frame from Get_ nodes |

---

## Next Test to Run

**Swap WAN 2.2 Animate into the InfiniteTalk workflow.** Goal: get WAN 2.2's better mouth shapes while keeping InfiniteTalk's audio-driven timing.

Changes to make vs current (ALT) workflow:
1. **Model**: swap `wan2.1_i2v_720p_14B_fp16` → WAN 2.2 Animate model in WanVideoModelLoader
2. **Seed**: match the seed being used in the standalone WAN 2.2 motion pass workflow
3. **Shift**: use **8 or 9** (not 6, not 11)
   - Shift=6 was a WAN 2.1 workaround; WAN 2.2 produces better mouths natively at higher shift
   - Shift=6 with WAN 2.2 causes color washout and stiff motion
   - Shift=11 (WAN 2.2 native default) is fine for the motion pass but probably too high for fine-detail InfiniteTalk work; 8–9 is the middle ground
4. **LoRA stack**: replace Lightx2v rank64 with:
   - UniAnimate LoRA (required, already there)
   - WANAnimate relight LoRA @ 0.5–0.7 (fixes color washout observed when using WAN 2.2 without its normal LoRA stack)
   - LightX2V elite **animate_face** LoRA @ 0.5–0.6 (prefer over rank64 since face/mouth quality is the priority)
   - Skip motion/dynamics LoRAs — UniAnimate is providing pose guidance from the motion pass reference

---

## Key Technical Findings

### Why WAN 2.2 has better mouth shapes
WAN 2.2 Animate was trained on animation content; it maps audio-style inputs to better 2D visemes for flat character art. WAN 2.1 InfiniteTalk is optimized for audio-driven generation but has a different mouth aesthetic. This is a model-level difference, not fixable purely by parameter tuning.

### VACE + InfiniteTalk are architecturally incompatible
- VACE runs in the sampler's **T2V branch** and requires a T2V base model
- InfiniteTalk runs in the sampler's **I2V branch** via `multitalk_loop()` — a completely separate code path with zero VACE handling
- `vace_data` is only populated in the T2V branch; it's `None` during the entire InfiniteTalk sampling loop
- A 3-stage VACE post-processing refinement pass (T2V + denoise_strength=0.15–0.25) is theoretically possible using `wan2.1_t2v_14B_fp16` + VACE module, but was decided against due to added complexity and uncertain payoff

### Shift parameter behavior
- `σ' = s·σ/(1+(s-1)·σ)` — higher shift = more steps at high noise (global structure, eye position); lower shift = more steps at low noise (fine details, mouth shapes)
- Too low (< 5) with WAN 2.1 causes eye position jumping between windows
- WAN 2.2 native default is ~11; using shift=6 causes color artifacts with WAN 2.2
- For WAN 2.1 InfiniteTalk: shift=6 is a good balance

### audio_scale vs audio_cfg_scale
- `audio_scale`: multiplies the audio embedding tensor before entering transformer — affects both amplitude AND viseme accuracy
- `audio_cfg_scale`: extra forward pass with zeroed audio; `noise_pred = uncond + cfg*(cond - uncond)`; more surgical for viseme contrast but **doubles render time**; keep at 1.0 unless render time is acceptable

### Face pose in DWPose
- Enabling `detect_face` in DWPreprocessor + face rendering in PoseKeypointSmoother improved mouth shapes in some tests but consistently hurt lipsync accuracy across multiple clips
- **Keep detect_face=OFF and render_face=OFF**

### Seam stabilization (PoseKeypointSmoother)
- `stabilize_radius`: controls WIDTH of Gaussian bell-curve effect (how many frames the stabilization spreads from each seam)
- `stabilize_sigma`: controls RIGIDITY of heavy smoothing pass within the zone (how frozen the pose is)
- Seam centers are at `frame_window_size - 0.5`, repeating every `frame_window_size - seam_motion_frame` frames

### InfiniteTalk architecture
- VAE 4:1 temporal compression; 16 motion frames = 4 latent frames
- Seam at `frame_window_size - 0.5`; `multitalk_loop.py` decodes per window
- `motion_frames=25` is the current setting (higher than the typical default)

---

## Workflow Files in Repo

- `workflows/LipSync_Stage1_MotionPass_2.2-I2V_InfiniteTalk.json` — Stage 1 motion pass (WAN 2.2 I2V)
- `workflows/LipSync_Stage2_2.1-V2V_InfiniteTalk.json` — Stage 2 final lipsync pass (WAN 2.1 InfiniteTalk, currently being tuned)

---

## Node Code Notes

- `pose_keypoint_smoother.py`: main file; `_compute_seam_weight()` is the Gaussian seam blending function
- `video_seam_rife.py`: RIFE-based seam replacement node (built, registered, not currently used in workflow)
- `__init__.py`: registers all 4 nodes
